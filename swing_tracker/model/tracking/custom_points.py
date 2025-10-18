from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Callable, Deque, Dict, Iterable, List, Optional, Set, Tuple
import threading
import logging

import cv2

from ..entities import Point2D, TrackIssue, TrackedPoint
from ..settings import TrackingSettings
from .provisional import AnchorMetrics, ProvisionalChain
from .segments import FrameSample, SegmentBuildResult, TrackingSegment, KeyFrame


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb


@dataclass
class CustomPointFrameResult:
    positions: Dict[str, Point2D]
    issues: List[TrackIssue]
    resolved_points: List[str]


class CustomPointTracker:
    def __init__(self, max_history_frames: int = 200) -> None:
        self._log = logging.getLogger(__name__)
        self.max_history_frames = max_history_frames
        self.points: Dict[str, TrackedPoint] = {}
        self.current_positions: Dict[str, Point2D] = {}
        self.truncate_future_on_manual_set: bool = True
        self.interpolation_mode: str = "linear"
        self.span_length: int = 50
        self.threshold_confidence: float = 0.6
        self.min_threshold_pixels: float = 6.0
        self.threshold_scale: float = 0.04
        self.nms_window: int = 8
        self.min_segment_len: int = 6
        self.max_provisionals_per_span: int = 5
        self.min_improvement_px: float = 2.0
        self.min_improvement_ratio: float = 0.2
        self.smoothing_window: int = 5
        self.lk_window_size: int = 21
        self.lk_max_level: int = 3
        self.lk_term_count: int = 30
        self.lk_term_epsilon: float = 0.01
        self.lk_feature_quality: float = 0.4
        self.lk_min_distance: float = 1.5
        self.lk_min_eig_threshold: float = 1e-4
        self.lk_batch_size: int = 8
        self.performance_mode: str = "Balanced"
        self.thread_priority: str = "Normal"
        self.cache_frames_enabled: bool = True
        self._provisional_cache: Dict[str, Dict[Tuple[int, int], ProvisionalChain]] = {}
        self._provisional_rejections: Dict[str, Dict[Tuple[int, int], Set[int]]] = {}
        self._provisional_dirty: Dict[str, Set[Tuple[int, int]]] = {}
        self._frame_samples: Dict[int, FrameSample] = {}
        self._frame_order: Deque[int] = deque()
        self._segment_results: Dict[str, Dict[Tuple[int, int], SegmentBuildResult]] = {}
        self._segment_dirty: Dict[str, Set[Tuple[int, int]]] = {}
        self._frame_loader: Optional[Callable[[int], Optional[object]]] = None
        self._frame_lock = threading.RLock()

    def configure_points(self, point_definitions: Dict[str, Tuple[int, int, int]]) -> None:
        self.points = {
            name: TrackedPoint(name=name, color=color)
            for name, color in point_definitions.items()
        }
        self.current_positions.clear()
        self._reset_provisional_state()
        with self._frame_lock:
            self._frame_samples.clear()
            self._frame_order.clear()

    def reset(self) -> None:
        for tracked_point in self.points.values():
            tracked_point.clear()
        self.current_positions.clear()
        self._reset_provisional_state()
        with self._frame_lock:
            self._frame_samples.clear()
            self._frame_order.clear()

    def serialize_state(self) -> Dict[str, Dict[str, object]]:
        payload: Dict[str, Dict[str, object]] = {}
        for name, tracked_point in self.points.items():
            if not tracked_point.keyframes:
                continue
            keyframes = {
                str(frame): [float(pos[0]), float(pos[1])]
                for frame, pos in tracked_point.keyframes.items()
            }
            payload[name] = {
                "keyframes": keyframes,
                "accepted": list(tracked_point.accepted_keyframes),
                "absences": [[int(start), int(end)] for start, end in tracked_point.absent_ranges],
                "open_absence": tracked_point.open_absence_start,
            }
        return payload

    def load_state(self, data: Dict[str, Dict[str, object]]) -> None:
        self.reset()
        for name, payload in data.items():
            tracked_point = self.points.get(name)
            if not tracked_point:
                continue
            keyframes = payload.get("keyframes", {})
            accepted = set(payload.get("accepted", []))
            for frame_str, pos in keyframes.items():
                try:
                    frame = int(frame_str)
                    x, y = pos
                    tracked_point.set_keyframe(frame, (float(x), float(y)), accepted=frame in accepted)
                except Exception:
                    continue
            absences = payload.get("absences", [])
            tracked_point.clear_absence_ranges()
            if isinstance(absences, list):
                for entry in absences:
                    try:
                        start, end = int(entry[0]), int(entry[1])
                    except Exception:
                        continue
                    tracked_point.add_absence(start, end)
                    self._apply_absence_range(tracked_point, start, end)
            open_absence = payload.get("open_absence")
            tracked_point.open_absence_start = int(open_absence) if isinstance(open_absence, int) else None
        self._invalidate_point_cache()

    def remove_absence_segment(self, point_name: str, frame_index: int) -> bool:
        tracked_point = self.points.get(point_name)
        if not tracked_point:
            return False

        frame_index = int(frame_index)
        removed = False
        new_ranges: List[Tuple[int, int]] = []
        for start, end in tracked_point.absent_ranges:
            if start <= frame_index <= end or start <= frame_index - 1 <= end:
                removed = True
                continue
            new_ranges.append((start, end))
        if removed:
            tracked_point.absent_ranges = sorted(new_ranges)

        open_start = tracked_point.open_absence_start
        if open_start is not None and frame_index >= open_start:
            tracked_point.open_absence_start = None
            removed = True

        if removed:
            self.current_positions.pop(point_name, None)
            self._invalidate_point_cache(point_name)
        return removed

    def split_absence_segment(self, point_name: str, frame_index: int) -> bool:
        tracked_point = self.points.get(point_name)
        if not tracked_point:
            return False

        frame_index = int(frame_index)
        for start, end in list(tracked_point.absent_ranges):
            if start < frame_index <= end:
                if frame_index == start:
                    return False
                tracked_point.absent_ranges.remove((start, end))
                if start <= frame_index - 1:
                    tracked_point.add_absence(start, frame_index - 1)
                if frame_index <= end:
                    tracked_point.add_absence(frame_index, end)
                self.current_positions.pop(point_name, None)
                self._invalidate_point_cache(point_name)
                return True

        open_start = tracked_point.open_absence_start
        if open_start is not None and frame_index > open_start + 1:
            tracked_point.add_absence(open_start + 1, frame_index - 1)
            tracked_point.open_absence_start = frame_index
            self.current_positions.pop(point_name, None)
            self._invalidate_point_cache(point_name)
            return True
        return False

    def set_frame_loader(self, loader: Optional[Callable[[int], Optional[object]]]) -> None:
        """Register a callback that returns a BGR frame for the given index."""

        self._frame_loader = loader

    def get_frame_loader(self) -> Optional[Callable[[int], Optional[object]]]:
        return self._frame_loader

    def ensure_frame_sample(self, frame_index: int) -> Optional[FrameSample]:
        loader = self._frame_loader
        if loader is None:
            return None
        if self.cache_frames_enabled:
            with self._frame_lock:
                cached = self._frame_samples.get(frame_index)
            if cached is not None:
                return cached
        frame_bgr = loader(frame_index)
        if frame_bgr is None:
            return None
        return self._store_frame(frame_index, frame_bgr)

    def fetch_frame_samples(self, span_start: int, span_end: int) -> Dict[int, FrameSample]:
        frames: Dict[int, FrameSample] = {}
        if span_end < span_start:
            return frames
        for frame_index in range(span_start, span_end + 1):
            sample = self.ensure_frame_sample(frame_index)
            if sample is None:
                return {}
            frames[frame_index] = sample
        return frames

    def cache_frame_sample(self, frame_index: int, frame_bgr) -> Optional[FrameSample]:
        return self._store_frame(frame_index, frame_bgr)

    # ------------------------------------------------------------------
    # Provisional refinement helpers
    # ------------------------------------------------------------------
    def _reset_provisional_state(self) -> None:
        self._provisional_cache = {}
        self._provisional_rejections = {}
        self._provisional_dirty = {}
        self._segment_results = {}
        self._segment_dirty = {}

    def _invalidate_point_cache(self, point_name: Optional[str] = None) -> None:
        if point_name is None:
            self._provisional_cache.clear()
            self._provisional_dirty.clear()
            self._segment_results.clear()
            self._segment_dirty.clear()
            for name in self.points:
                self._mark_all_spans_dirty(name)
            return
        self._provisional_cache.pop(point_name, None)
        self._provisional_dirty.pop(point_name, None)
        self._segment_results.pop(point_name, None)
        self._segment_dirty.pop(point_name, None)
        self._mark_all_spans_dirty(point_name)

    def _invalidate_segment_cache(self, point_name: Optional[str] = None) -> None:
        if point_name is None:
            self._segment_results.clear()
            self._segment_dirty.clear()
            return
        self._segment_results.pop(point_name, None)
        self._segment_dirty.pop(point_name, None)

    def _rejections_for(self, point_name: str, span: Tuple[int, int]) -> Set[int]:
        point_map = self._provisional_rejections.setdefault(point_name, {})
        return point_map.setdefault(span, set())

    def _mark_span_dirty(self, point_name: str, span: Tuple[int, int]) -> None:
        self._provisional_dirty.setdefault(point_name, set()).add(span)
        self._segment_dirty.setdefault(point_name, set()).add(span)
        result_map = self._segment_results.get(point_name)
        if result_map is not None:
            result_map.pop(span, None)
        cache = self._provisional_cache.get(point_name)
        if cache is not None:
            cache.pop(span, None)

    def _mark_all_spans_dirty(self, point_name: str) -> None:
        tracked_point = self.points.get(point_name)
        if not tracked_point:
            return
        spans = list(self._iter_span_pairs(tracked_point))
        if not spans:
            return
        segment_dirty = self._segment_dirty.setdefault(point_name, set())
        provisional_dirty = self._provisional_dirty.setdefault(point_name, set())
        cache = self._provisional_cache.setdefault(point_name, {})
        results = self._segment_results.setdefault(point_name, {})
        for span in spans:
            segment_dirty.add(span)
            provisional_dirty.add(span)
            cache.pop(span, None)
            results.pop(span, None)

    def _segment_result(
        self, point_name: str, span: Tuple[int, int], tracked_point: TrackedPoint
    ) -> Optional[SegmentBuildResult]:
        if span in self._segment_dirty.get(point_name, set()):
            return None
        result_map = self._segment_results.get(point_name, {})
        return result_map.get(span)

    def _iter_span_pairs(self, tracked_point: TrackedPoint) -> Iterable[Tuple[int, int]]:
        confirmed = tracked_point.accepted_keyframe_frames()
        if len(confirmed) < 2:
            confirmed = tracked_point.keyframe_frames()
        confirmed = sorted(confirmed)
        for idx in range(len(confirmed) - 1):
            start = confirmed[idx]
            end = confirmed[idx + 1]
            if end <= start:
                continue
            yield (start, end)

    def _span_for_frame(self, tracked_point: TrackedPoint, frame_index: int) -> Optional[Tuple[int, int]]:
        spans = list(self._iter_span_pairs(tracked_point))
        if not spans:
            return None
        for start, end in spans:
            if start <= frame_index <= end:
                return (start, end)
        # Fallback: closest span
        if frame_index < spans[0][0]:
            return spans[0]
        return spans[-1]

    def _get_chain(self, point_name: str, span: Tuple[int, int]) -> Optional[ProvisionalChain]:
        cache = self._provisional_cache.setdefault(point_name, {})
        dirty_spans = self._provisional_dirty.setdefault(point_name, set())
        if span in cache and span not in dirty_spans:
            return cache[span]
        result_map = self._segment_results.get(point_name, {})
        result = result_map.get(span)
        if result is None or span in dirty_spans:
            return None
        cache[span] = result.chain
        return result.chain

    def ensure_all_spans(self, point_name: str) -> None:
        tracked_point = self.points.get(point_name)
        if not tracked_point:
            return
        for span in self._iter_span_pairs(tracked_point):
            self._get_chain(point_name, span)

    def provisional_chain_for_frame(
        self, point_name: str, frame_index: int
    ) -> Optional[ProvisionalChain]:
        tracked_point = self.points.get(point_name)
        if not tracked_point:
            return None
        span = self._span_for_frame(tracked_point, frame_index)
        if span is None:
            return None
        return self._get_chain(point_name, span)

    def provisional_chain_for_span(
        self, point_name: str, span: Tuple[int, int]
    ) -> Optional[ProvisionalChain]:
        return self._get_chain(point_name, span)

    def all_provisional_candidates(self, point_name: str) -> List[AnchorMetrics]:
        self.ensure_all_spans(point_name)
        cache = self._provisional_cache.get(point_name, {})
        anchors: List[AnchorMetrics] = []
        for chain in cache.values():
            anchors.extend([c.anchor for c in chain.provisionals()])
        anchors.sort(key=lambda anchor: anchor.frame)
        return anchors

    def provisional_markers(self, point_name: str) -> List[AnchorMetrics]:
        return self.all_provisional_candidates(point_name)

    def count_provisionals(self, point_name: str) -> int:
        self.ensure_all_spans(point_name)
        cache = self._provisional_cache.get(point_name, {})
        return sum(chain.total_provisionals() for chain in cache.values())

    def tracking_segments(self, point_name: str) -> List[TrackingSegment]:
        tracked_point = self.points.get(point_name)
        if not tracked_point:
            return []
        segments: List[TrackingSegment] = []
        for span in self._iter_span_pairs(tracked_point):
            result = self._segment_result(point_name, span, tracked_point)
            if result:
                segments.extend(result.segments)
            else:
                fallback = self._build_fallback_segment(point_name, span, tracked_point)
                if fallback:
                    segments.append(fallback)
        segments.sort(key=lambda segment: (segment.start_key.frame, segment.end_key.frame, segment.point_name or ""))
        return segments

    def all_tracking_segments(self) -> List[TrackingSegment]:
        segments: List[TrackingSegment] = []
        for point_name in self.points.keys():
            segments.extend(self.tracking_segments(point_name))
        segments.sort(key=lambda segment: (segment.start_key.frame, segment.end_key.frame, segment.point_name or ""))
        return segments

    def _build_fallback_segment(
        self,
        point_name: str,
        span: Tuple[int, int],
        tracked_point: TrackedPoint,
    ) -> Optional[TrackingSegment]:
        start_frame, end_frame = span
        if end_frame <= start_frame:
            return None
        start_pos = tracked_point.keyframes.get(start_frame)
        end_pos = tracked_point.keyframes.get(end_frame)
        if start_pos is None or end_pos is None:
            return None

        def interpolate(frame: int) -> Point2D:
            if frame in tracked_point.positions:
                return tracked_point.positions[frame]
            ratio = (frame - start_frame) / (end_frame - start_frame)
            x = start_pos[0] + (end_pos[0] - start_pos[0]) * ratio
            y = start_pos[1] + (end_pos[1] - start_pos[1]) * ratio
            return (x, y)

        tracked_positions: Dict[int, Point2D] = {}
        for frame in range(start_frame, end_frame + 1):
            tracked_positions[frame] = interpolate(frame)

        entity_colour = None
        if tracked_point.color:
            entity_colour = "#%02x%02x%02x" % tracked_point.color

        gradient_colour = "#66ff00"
        gradient_colours = {frame: gradient_colour for frame in range(start_frame, end_frame + 1)}

        start_key = KeyFrame(frame=start_frame, pos=start_pos, type="confirmed", conf=1.0)
        end_key = KeyFrame(frame=end_frame, pos=end_pos, type="confirmed", conf=1.0)

        segment = TrackingSegment(
            start_key=start_key,
            end_key=end_key,
            tracked_positions=tracked_positions,
            residuals={},
            gradient_colours=gradient_colours,
            provisional_points=[],
            accepted=False,
            entity_colour=entity_colour,
        )
        segment.point_name = point_name
        return segment

    def span_pairs(self, point_name: str) -> List[Tuple[int, int]]:
        tracked_point = self.points.get(point_name)
        if not tracked_point:
            return []
        return list(self._iter_span_pairs(tracked_point))

    def requires_tracking(self, point_name: str, span: Tuple[int, int]) -> bool:
        dirty_spans = self._segment_dirty.get(point_name, set())
        if span in dirty_spans:
            return True
        result_map = self._segment_results.get(point_name, {})
        return span not in result_map

    def get_frame_loader(self):
        return self._frame_loader

    def point_colour_hex(self, point_name: str) -> Optional[str]:
        tracked_point = self.points.get(point_name)
        if not tracked_point:
            return None
        return _rgb_to_hex(tracked_point.color)

    def rejected_frames_for_span(self, point_name: str, span: Tuple[int, int]) -> Set[int]:
        return self._rejections_for(point_name, span).copy()

    def apply_segment_result(
        self,
        point_name: str,
        span: Tuple[int, int],
        result: Optional[SegmentBuildResult],
    ) -> None:
        tracked_point = self.points.get(point_name)
        if not tracked_point:
            return

        if result is None:
            # Leave the span dirty so the manager can retry later.
            self._mark_span_dirty(point_name, span)
            self._log.debug("apply_segment_result: missing result point=%s span=%s", point_name, span)
            return

        self._log.debug(
            "apply_segment_result: point=%s span=%s positions=%d",
            point_name,
            span,
            len(result.optical_flow.positions),
        )
        for frame_index, position in result.optical_flow.positions.items():
            if frame_index in tracked_point.keyframes:
                continue
            tracked_point.positions[frame_index] = position
            tracked_point.confidence[frame_index] = result.optical_flow.confidences.get(frame_index, 0.0)
            tracked_point.fb_errors[frame_index] = result.optical_flow.fb_errors.get(frame_index, 0.0)

        for segment in result.segments:
            segment.point_name = point_name

        cache = self._provisional_cache.setdefault(point_name, {})
        cache[span] = result.chain
        self._provisional_dirty.setdefault(point_name, set()).discard(span)

        result_map = self._segment_results.setdefault(point_name, {})
        result_map[span] = result
        self._segment_dirty.setdefault(point_name, set()).discard(span)
        self._log.debug(
            "apply_segment_result: stored point=%s span=%s segments=%d",
            point_name,
            span,
            len(result.segments),
        )


    def accept_provisional(self, point_name: str, span: Tuple[int, int], frame: int) -> bool:
        tracked_point = self.points.get(point_name)
        if not tracked_point:
            return False
        actual_span = span
        inferred_span = self._span_for_frame(tracked_point, frame)
        if inferred_span is not None:
            actual_span = inferred_span
        chain = self._get_chain(point_name, actual_span)
        if not chain:
            return False
        candidate = next((c for c in chain.provisionals() if c.anchor.frame == frame), None)
        if not candidate:
            return False
        tracked_point.set_keyframe(frame, candidate.anchor.position, accepted=True)
        tracked_point.smoothed_position = candidate.anchor.position
        tracked_point.last_frame_index = frame
        self._mark_span_dirty(point_name, actual_span)
        self._invalidate_point_cache(point_name)
        rejections = self._rejections_for(point_name, actual_span)
        rejections.discard(frame)
        if not rejections:
            point_rejections = self._provisional_rejections.get(point_name)
            if point_rejections is not None:
                point_rejections.pop(actual_span, None)
        return True

    def accept_all_provisionals(self, point_name: str, span: Tuple[int, int]) -> bool:
        chain = self._get_chain(point_name, span)
        if not chain:
            return False
        frames = [candidate.anchor.frame for candidate in chain.provisionals()]
        changed = False
        for frame in frames:
            changed |= self.accept_provisional(point_name, span, frame)
        return changed

    def reject_provisional(self, point_name: str, span: Tuple[int, int], frame: int) -> bool:
        tracked_point = self.points.get(point_name)
        if not tracked_point:
            return False
        actual_span = span
        inferred_span = self._span_for_frame(tracked_point, frame)
        if inferred_span is not None:
            actual_span = inferred_span
        chain = self._get_chain(point_name, actual_span)
        if not chain:
            return False
        candidate = next((c for c in chain.provisionals() if c.anchor.frame == frame), None)
        if not candidate:
            return False
        self._rejections_for(point_name, actual_span).add(frame)
        self._mark_span_dirty(point_name, actual_span)
        self._invalidate_point_cache(point_name)
        return True

    def reject_all_provisionals(self, point_name: str, span: Tuple[int, int]) -> bool:
        chain = self._get_chain(point_name, span)
        if not chain:
            return False
        frames = [candidate.anchor.frame for candidate in chain.provisionals()]
        changed = False
        for frame in frames:
            changed |= self.reject_provisional(point_name, span, frame)
        return changed

    def process_frame(self, frame_bgr, frame_index: int, record: bool) -> CustomPointFrameResult:
        issues: List[TrackIssue] = []
        resolved: Set[str] = set()
        new_positions: Dict[str, Point2D] = {}

        if frame_bgr is not None:
            self._store_frame(frame_index, frame_bgr)

        for point_name, tracked_point in self.points.items():
            if not tracked_point.keyframes:
                continue
            if tracked_point.is_absent(frame_index):
                tracked_point.positions.pop(frame_index, None)
                tracked_point.confidence.pop(frame_index, None)
                tracked_point.fb_errors.pop(frame_index, None)
                tracked_point.low_confidence_frames.discard(frame_index)
                tracked_point.interpolation_cache.pop(frame_index, None)
                tracked_point.smoothed_position = None
                tracked_point.last_frame_index = frame_index
                continue

            expected = self._expected_position(tracked_point, frame_index)
            if expected is None:
                continue

            new_positions[point_name] = expected
            tracked_point.smoothed_position = expected
            tracked_point.last_frame_index = frame_index

        self.current_positions = new_positions

        return CustomPointFrameResult(
            positions=new_positions,
            issues=issues,
            resolved_points=sorted(resolved),
        )

    def set_manual_point(self, frame_index: int, point_name: str, position: Point2D) -> None:
        self._log.debug(
            "set_manual_point start point=%s frame=%s position=%s", point_name, frame_index, position
        )
        tracked_point = self.points[point_name]
        was_absent = tracked_point.is_absent(frame_index)
        open_start = tracked_point.open_absence_start
        tracked_point.remove_absence_at(frame_index)
        if was_absent and (open_start is None or frame_index < open_start):
            tracked_point.end_absence_at(frame_index)
        tracked_point.set_keyframe(frame_index, position, accepted=True)
        self._interpolate_neighbors(tracked_point, frame_index)
        tracked_point.smoothed_position = position
        tracked_point.last_frame_index = frame_index
        self.current_positions[point_name] = position
        self._invalidate_point_cache(point_name)
        self._log.debug(
            "set_manual_point complete point=%s frame=%s total_keyframes=%s",
            point_name,
            frame_index,
            len(tracked_point.keyframes),
        )

    def clear_point_history(self, point_name: str) -> None:
        if point_name not in self.points:
            return
        self.points[point_name].clear()
        self.current_positions.pop(point_name, None)
        self._invalidate_point_cache(point_name)

    def mark_point_absent(self, point_name: str, start_frame: int, end_frame: int) -> None:
        tracked_point = self.points.get(point_name)
        if not tracked_point:
            return
        start = max(0, min(start_frame, end_frame))
        end = max(start, max(start_frame, end_frame))
        tracked_point.add_absence(start, end)
        self._apply_absence_range(tracked_point, start, end)
        self._invalidate_point_cache(point_name)

    def set_point_absences(self, point_name: str, ranges: List[Tuple[int, int]]) -> None:
        tracked_point = self.points.get(point_name)
        if not tracked_point:
            return
        tracked_point.clear_absence_ranges()
        tracked_point.open_absence_start = None
        for start, end in sorted(ranges):
            start = max(0, start)
            end = max(start, end)
            tracked_point.add_absence(start, end)
        for start, end in tracked_point.absent_ranges:
            self._apply_absence_range(tracked_point, start, end)
        self._invalidate_point_cache(point_name)

    def clear_point_absences(self, point_name: str) -> None:
        tracked_point = self.points.get(point_name)
        if not tracked_point:
            return
        tracked_point.clear_absence_ranges()
        self.current_positions.pop(point_name, None)
        self._invalidate_point_cache(point_name)

    def point_absence_ranges(self, point_name: str) -> List[Tuple[int, int]]:
        tracked_point = self.points.get(point_name)
        if not tracked_point:
            return []
        return list(tracked_point.absent_ranges)

    def mark_stop_frame(self, point_name: str, frame_index: int, total_frames: int) -> bool:
        tracked_point = self.points.get(point_name)
        if not tracked_point:
            return False

        max_index = total_frames - 1 if total_frames > 0 else frame_index
        frame_index = max(0, min(frame_index, max_index))

        current_start = tracked_point.open_absence_start
        if current_start is None:
            tracked_point.open_absence_start = frame_index
            tracked_point.remove_absence_at(frame_index)
            self.current_positions.pop(point_name, None)
            self._invalidate_point_cache(point_name)
            return True

        if frame_index <= current_start:
            return False

        end_frame = frame_index - 1
        absence_start = current_start + 1
        if absence_start <= end_frame:
            tracked_point.add_absence(absence_start, end_frame)
            self._apply_absence_range(tracked_point, absence_start, end_frame)
        tracked_point.open_absence_start = None
        tracked_point.end_absence_at(frame_index)
        self.current_positions.pop(point_name, None)
        self._invalidate_point_cache(point_name)
        return True

    def timeline_markers(self) -> Dict[int, str]:
        marker_map: Dict[int, str] = {}
        for tracked_point in self.points.values():
            for frame in tracked_point.accepted_keyframe_frames():
                marker_map.setdefault(frame, "keyframe")
            for start, end in tracked_point.absent_ranges:
                marker_map[start] = "stop"
                resume_frame = end + 1
                marker_map[resume_frame] = "start"
            if tracked_point.open_absence_start is not None:
                marker_map[tracked_point.open_absence_start] = "stop"
        return marker_map

    def point_definitions(self) -> Dict[str, TrackedPoint]:
        return self.points

    def keyframe_frames(self) -> List[int]:
        frames: Set[int] = set()
        for tracked_point in self.points.values():
            frames.update(tracked_point.keyframe_frames())
        return sorted(frames)

    def accepted_keyframe_frames(self) -> List[int]:
        frames: Set[int] = set()
        for tracked_point in self.points.values():
            frames.update(tracked_point.accepted_keyframe_frames())
        return sorted(frames)

    def pending_keyframes(self) -> Dict[str, List[int]]:
        pending: Dict[str, List[int]] = {}
        for name, tracked_point in self.points.items():
            frames = tracked_point.pending_keyframe_frames()
            if frames:
                pending[name] = frames
        return pending

    def next_pending_keyframe(self, point_name: str) -> Optional[int]:
        tracked_point = self.points.get(point_name)
        if not tracked_point:
            return None
        frames = tracked_point.pending_keyframe_frames()
        return frames[0] if frames else None

    def accept_keyframe(self, point_name: str, frame_index: int) -> bool:
        tracked_point = self.points.get(point_name)
        if not tracked_point:
            return False
        changed = tracked_point.mark_keyframe_accepted(frame_index)
        if changed:
            tracked_point.interpolation_cache.pop(frame_index, None)
            self._invalidate_point_cache(point_name)
        return changed


    def reject_keyframe(self, point_name: str, frame_index: int) -> bool:
        tracked_point = self.points.get(point_name)
        if not tracked_point:
            return False
        removed = tracked_point.remove_keyframe(frame_index)
        if removed:
            tracked_point.interpolation_cache.pop(frame_index, None)
            self._invalidate_point_cache(point_name)
        return removed

    def delete_keyframe(self, point_name: str, frame_index: int) -> bool:
        tracked_point = self.points.get(point_name)
        if not tracked_point:
            return False
        related_spans = [span for span in self._iter_span_pairs(tracked_point) if span[0] <= frame_index <= span[1]]
        removed = tracked_point.remove_keyframe(frame_index)
        if not removed:
            return False
        tracked_point.positions.pop(frame_index, None)
        tracked_point.confidence.pop(frame_index, None)
        tracked_point.fb_errors.pop(frame_index, None)
        tracked_point.low_confidence_frames.discard(frame_index)
        tracked_point.interpolation_cache.pop(frame_index, None)
        for span in related_spans:
            self._mark_span_dirty(point_name, span)
        self._invalidate_point_cache(point_name)
        return True
        

    def update_from_settings(self, settings: TrackingSettings) -> None:
        self.max_history_frames = max(1, int(settings.history_frames))
        self.truncate_future_on_manual_set = bool(settings.truncate_future_on_manual_set)
        self.interpolation_mode = settings.baseline_mode.lower()
        if self.interpolation_mode not in {"linear", "cubic", "const-velocity"}:
            self.interpolation_mode = "linear"
        self.threshold_confidence = float(max(0.0, min(1.0, settings.issue_confidence_threshold)))
        self.min_threshold_pixels = float(max(0.1, settings.deviation_threshold))
        window = max(5, int(settings.optical_flow_window_size))
        if window % 2 == 0:
            window += 1
        self.lk_window_size = window
        self.lk_max_level = max(0, int(settings.optical_flow_pyramid))
        self.lk_term_count = max(1, int(settings.optical_flow_term_count))
        self.lk_term_epsilon = float(max(1e-7, settings.optical_flow_term_epsilon))
        self.lk_feature_quality = float(max(0.0, min(1.0, settings.optical_flow_feature_quality)))
        self.lk_min_distance = float(max(0.0, settings.optical_flow_min_distance))
        self.lk_min_eig_threshold = float(max(1e-8, settings.optical_flow_min_eig_threshold))
        self.lk_batch_size = max(1, int(settings.optical_flow_batch_size))
        self.performance_mode = settings.performance_mode or "Balanced"
        self.thread_priority = settings.thread_priority or "Normal"
        self.cache_frames_enabled = bool(settings.cache_frames_enabled)
        if not self.cache_frames_enabled:
            with self._frame_lock:
                self._frame_samples.clear()
                self._frame_order.clear()
        self._invalidate_point_cache()
        self._trim_frame_store()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _store_frame(self, frame_index: int, frame_bgr) -> Optional[FrameSample]:
        if frame_bgr is None:
            return None
        if not self.cache_frames_enabled:
            return FrameSample(frame_bgr.copy())
        with self._frame_lock:
            existing = self._frame_samples.get(frame_index)
            if existing is not None:
                return existing
            sample = FrameSample(frame_bgr.copy())
            self._frame_samples[frame_index] = sample
            self._frame_order.append(frame_index)
            self._trim_frame_store_locked()
            return sample

    def _trim_frame_store(self) -> None:
        with self._frame_lock:
            self._trim_frame_store_locked()

    def _trim_frame_store_locked(self) -> None:
        limit = self.max_history_frames
        if limit <= 0:
            self._frame_samples.clear()
            self._frame_order.clear()
            return
        limit = max(limit, self.span_length + 2)
        while len(self._frame_order) > limit:
            old_frame = self._frame_order.popleft()
            self._frame_samples.pop(old_frame, None)

    def _expected_position(self, tracked_point: TrackedPoint, frame_index: int) -> Optional[Point2D]:
        if tracked_point.is_absent(frame_index):
            tracked_point.interpolation_cache.pop(frame_index, None)
            return None
        if frame_index in tracked_point.interpolation_cache:
            return tracked_point.interpolation_cache[frame_index]

        if not tracked_point.keyframes:
            value = tracked_point.smoothed_position or tracked_point.positions.get(frame_index)
            if value is not None:
                tracked_point.interpolation_cache[frame_index] = value
            return value

        frames = tracked_point.keyframe_frames()
        if frame_index in tracked_point.keyframes:
            return tracked_point.keyframes[frame_index]

        prev_frame = max((f for f in frames if f < frame_index), default=None)
        next_frame = min((f for f in frames if f > frame_index), default=None)

        if prev_frame is None and next_frame is None:
            value = tracked_point.smoothed_position
            if value is not None:
                tracked_point.interpolation_cache[frame_index] = value
            return value
        if prev_frame is None:
            value = tracked_point.smoothed_position or tracked_point.keyframes[next_frame]
            if value is not None:
                tracked_point.interpolation_cache[frame_index] = value
            return value
        if next_frame is None:
            value = tracked_point.smoothed_position or tracked_point.keyframes[prev_frame]
            if value is not None:
                tracked_point.interpolation_cache[frame_index] = value
            return value

        start_pos = tracked_point.keyframes[prev_frame]
        end_pos = tracked_point.keyframes[next_frame]
        span = next_frame - prev_frame
        if span <= 0:
            return start_pos
        ratio = (frame_index - prev_frame) / span
        x = start_pos[0] + ratio * (end_pos[0] - start_pos[0])
        y = start_pos[1] + ratio * (end_pos[1] - start_pos[1])
        interpolated = (x, y)
        tracked_point.interpolation_cache[frame_index] = interpolated
        return interpolated

    def _interpolate_neighbors(self, tracked_point: TrackedPoint, frame_index: int) -> None:
        frames = tracked_point.keyframe_frames()
        frames_set = set(frames)
        prev_frame = max((f for f in frames if f < frame_index), default=None)
        next_frame = min((f for f in frames if f > frame_index), default=None)

        # interpolate previous -> current
        if prev_frame is not None:
            self._clear_cache_range(tracked_point, prev_frame, frame_index)
            self._fill_between(tracked_point, prev_frame, frame_index)

        # interpolate current -> next
        if next_frame is not None:
            self._clear_cache_range(tracked_point, frame_index, next_frame)
            self._fill_between(tracked_point, frame_index, next_frame)

        # Remove cached positions beyond max history
        self._trim_history(tracked_point, frame_index)

    def _fill_between(self, tracked_point: TrackedPoint, start_frame: int, end_frame: int) -> None:
        if end_frame <= start_frame + 1:
            return
        start_pos = tracked_point.keyframes[start_frame]
        end_pos = tracked_point.keyframes[end_frame]
        span = end_frame - start_frame
        for f in range(start_frame + 1, end_frame):
            ratio = (f - start_frame) / span
            x = start_pos[0] + ratio * (end_pos[0] - start_pos[0])
            y = start_pos[1] + ratio * (end_pos[1] - start_pos[1])
            tracked_point.positions[f] = (x, y)
            tracked_point.confidence[f] = 1.0
            tracked_point.interpolation_cache[f] = (x, y)

    def _apply_absence_range(self, tracked_point: TrackedPoint, start_frame: int, end_frame: int) -> None:
        frames_to_remove = [f for f in tracked_point.positions if start_frame <= f <= end_frame]
        for frame in frames_to_remove:
            tracked_point.positions.pop(frame, None)
            tracked_point.confidence.pop(frame, None)
            tracked_point.fb_errors.pop(frame, None)
            tracked_point.low_confidence_frames.discard(frame)
            tracked_point.interpolation_cache.pop(frame, None)

        keyframes_to_remove = [f for f in tracked_point.keyframes if start_frame <= f <= end_frame]
        for frame in keyframes_to_remove:
            tracked_point.keyframes.pop(frame, None)
            tracked_point.accepted_keyframes.discard(frame)
            tracked_point.interpolation_cache.pop(frame, None)

        cache_frames = [f for f in tracked_point.interpolation_cache if start_frame <= f <= end_frame]
        for frame in cache_frames:
            tracked_point.interpolation_cache.pop(frame, None)

        if (
            tracked_point.last_frame_index is not None
            and start_frame <= tracked_point.last_frame_index <= end_frame
        ):
            tracked_point.last_frame_index = None
            tracked_point.smoothed_position = None

        self.current_positions.pop(tracked_point.name, None)

    def _trim_history(self, tracked_point: TrackedPoint, current_frame: int) -> None:
        if self.max_history_frames <= 0:
            return
        min_frame = current_frame - self.max_history_frames
        to_remove = [f for f in tracked_point.positions if f < min_frame and f not in tracked_point.keyframes]
        for f in to_remove:
            tracked_point.positions.pop(f, None)
            tracked_point.confidence.pop(f, None)
            tracked_point.fb_errors.pop(f, None)
            tracked_point.low_confidence_frames.discard(f)
            tracked_point.interpolation_cache.pop(f, None)

    def _clear_cache_range(self, tracked_point: TrackedPoint, start_frame: int, end_frame: int) -> None:
        for f in range(start_frame + 1, end_frame):
            tracked_point.interpolation_cache.pop(f, None)
