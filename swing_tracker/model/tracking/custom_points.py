from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Callable, Deque, Dict, Iterable, List, Optional, Set, Tuple

import cv2

from ..entities import Point2D, TrackIssue, TrackedPoint
from ..settings import TrackingSettings
from .provisional import AnchorMetrics, ProvisionalChain, build_provisional_chain
from .segments import FrameSample, SegmentBuildResult, SegmentBuilder, TrackingSegment


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb


@dataclass
class CustomPointFrameResult:
    positions: Dict[str, Point2D]
    issues: List[TrackIssue]
    resolved_points: List[str]


class CustomPointTracker:
    def __init__(self, max_history_frames: int = 200) -> None:
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
        self._provisional_cache: Dict[str, Dict[Tuple[int, int], ProvisionalChain]] = {}
        self._provisional_rejections: Dict[str, Dict[Tuple[int, int], Set[int]]] = {}
        self._provisional_dirty: Dict[str, Set[Tuple[int, int]]] = {}
        self._frame_samples: Dict[int, FrameSample] = {}
        self._frame_order: Deque[int] = deque()
        self._segment_results: Dict[str, Dict[Tuple[int, int], SegmentBuildResult]] = {}
        self._segment_dirty: Dict[str, Set[Tuple[int, int]]] = {}
        self._frame_loader: Optional[Callable[[int], Optional[object]]] = None

    def configure_points(self, point_definitions: Dict[str, Tuple[int, int, int]]) -> None:
        self.points = {
            name: TrackedPoint(name=name, color=color)
            for name, color in point_definitions.items()
        }
        self.current_positions.clear()
        self._reset_provisional_state()
        self._frame_samples.clear()
        self._frame_order.clear()

    def reset(self) -> None:
        for tracked_point in self.points.values():
            tracked_point.clear()
        self.current_positions.clear()
        self._reset_provisional_state()
        self._frame_samples.clear()
        self._frame_order.clear()

    def set_frame_loader(self, loader: Optional[Callable[[int], Optional[object]]]) -> None:
        """Register a callback that returns a BGR frame for the given index."""

        self._frame_loader = loader

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
            return
        self._provisional_cache.pop(point_name, None)
        self._provisional_dirty.pop(point_name, None)
        self._segment_results.pop(point_name, None)
        self._segment_dirty.pop(point_name, None)

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

    def _segment_result(
        self, point_name: str, span: Tuple[int, int], tracked_point: TrackedPoint
    ) -> Optional[SegmentBuildResult]:
        result_map = self._segment_results.setdefault(point_name, {})
        dirty_spans = self._segment_dirty.setdefault(point_name, set())
        if span in result_map and span not in dirty_spans:
            return result_map[span]

        result = self._build_segment_result(point_name, span, tracked_point)
        if result is not None:
            result_map[span] = result
            dirty_spans.discard(span)
        else:
            result_map.pop(span, None)
        return result

    def _build_segment_result(
        self, point_name: str, span: Tuple[int, int], tracked_point: TrackedPoint
    ) -> Optional[SegmentBuildResult]:
        start_frame, end_frame = span
        if end_frame <= start_frame:
            return None
        start_pos = tracked_point.keyframes.get(start_frame)
        end_pos = tracked_point.keyframes.get(end_frame)
        if start_pos is None or end_pos is None:
            return None

        frames_needed = range(start_frame, end_frame + 1)
        frame_samples: Dict[int, FrameSample] = {}
        for frame_index in frames_needed:
            sample = self._frame_samples.get(frame_index)
            if sample is None and self._frame_loader is not None:
                frame_bgr = self._frame_loader(frame_index)
                if frame_bgr is not None:
                    self._store_frame(frame_index, frame_bgr)
                    sample = self._frame_samples.get(frame_index)
            if sample is None:
                return None
            frame_samples[frame_index] = sample

        builder = SegmentBuilder(
            baseline_mode=self.interpolation_mode,
            threshold_confidence=self.threshold_confidence,
            min_threshold_pixels=self.min_threshold_pixels,
            threshold_scale=self.threshold_scale,
            nms_window=self.nms_window,
            min_segment_len=self.min_segment_len,
            max_provisionals=self.max_provisionals_per_span,
            min_improvement_px=self.min_improvement_px,
            min_improvement_ratio=self.min_improvement_ratio,
            smoothing_window=self.smoothing_window,
            entity_colour=_rgb_to_hex(tracked_point.color),
        )

        result = builder.build(
            span_start=start_frame,
            span_end=end_frame,
            start_pos=start_pos,
            end_pos=end_pos,
            frames=frame_samples,
            accepted_frames=set(tracked_point.accepted_keyframe_frames()),
            rejected_frames=self._rejections_for(point_name, span).copy(),
        )
        if result is None:
            return None

        for frame_index, position in result.optical_flow.positions.items():
            if frame_index in tracked_point.keyframes:
                continue
            tracked_point.positions[frame_index] = position
            tracked_point.confidence[frame_index] = result.optical_flow.confidences.get(frame_index, 0.0)
            tracked_point.fb_errors[frame_index] = result.optical_flow.fb_errors.get(frame_index, 0.0)

        return result

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
            if end - start > self.span_length:
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

    def _compute_chain(self, point_name: str, span: Tuple[int, int]) -> Optional[ProvisionalChain]:
        tracked_point = self.points.get(point_name)
        if not tracked_point:
            return None
        start_frame, end_frame = span
        if end_frame <= start_frame:
            return None
        start_pos = tracked_point.keyframes.get(start_frame)
        end_pos = tracked_point.keyframes.get(end_frame)
        if start_pos is None or end_pos is None:
            return None
        rejections = self._rejections_for(point_name, span)
        segment_result = self._segment_result(point_name, span, tracked_point)
        if segment_result is not None:
            chain = segment_result.chain
        else:
            chain = build_provisional_chain(
                start_frame,
                end_frame,
                start_pos,
                end_pos,
                tracked_point.positions,
                tracked_point.confidence,
                tracked_point.fb_errors,
                baseline_mode=self.interpolation_mode,
                rejected_frames=rejections,
                threshold_confidence=self.threshold_confidence,
                min_threshold_pixels=self.min_threshold_pixels,
                threshold_scale=self.threshold_scale,
                nms_window=self.nms_window,
                min_segment_len=self.min_segment_len,
                max_provisionals=self.max_provisionals_per_span,
                min_improvement_px=self.min_improvement_px,
                min_improvement_ratio=self.min_improvement_ratio,
                smoothing_window=self.smoothing_window,
            )
        if chain.total_provisionals() == 0:
            # Ensure cache still records empty chain for UI consistency.
            self._provisional_cache.setdefault(point_name, {})[span] = chain
            self._provisional_dirty.setdefault(point_name, set()).discard(span)
            return chain
        self._provisional_cache.setdefault(point_name, {})[span] = chain
        self._provisional_dirty.setdefault(point_name, set()).discard(span)
        return chain

    def _get_chain(self, point_name: str, span: Tuple[int, int]) -> Optional[ProvisionalChain]:
        cache = self._provisional_cache.setdefault(point_name, {})
        if span in cache and span not in self._provisional_dirty.get(point_name, set()):
            return cache[span]
        return self._compute_chain(point_name, span)

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
        segments.sort(key=lambda segment: (segment.start_key.frame, segment.end_key.frame))
        return segments

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
        tracked_point = self.points[point_name]
        was_absent = tracked_point.is_absent(frame_index)
        open_start = tracked_point.open_absence_start
        tracked_point.remove_absence_at(frame_index)
        if was_absent and (open_start is None or frame_index < open_start):
            tracked_point.end_absence_at(frame_index)
        if self.truncate_future_on_manual_set:
            tracked_point.truncate_after(frame_index)
        tracked_point.set_keyframe(frame_index, position, accepted=True)
        self._interpolate_neighbors(tracked_point, frame_index)
        tracked_point.smoothed_position = position
        tracked_point.last_frame_index = frame_index
        self.current_positions[point_name] = position
        self._invalidate_point_cache(point_name)

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
            tracked_point.truncate_after(frame_index - 1)
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
        

    def update_from_settings(self, settings: TrackingSettings) -> None:
        self.max_history_frames = settings.history_frames
        self.truncate_future_on_manual_set = settings.truncate_future_on_manual_set
        self.interpolation_mode = settings.baseline_mode.lower()
        if self.interpolation_mode not in {"linear", "cubic", "const-velocity"}:
            self.interpolation_mode = "linear"
        self.threshold_confidence = float(max(0.0, min(1.0, settings.issue_confidence_threshold)))
        self._invalidate_point_cache()
        self._trim_frame_store()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _store_frame(self, frame_index: int, frame_bgr) -> None:
        if frame_bgr is None:
            return
        if frame_index in self._frame_samples:
            return
        sample = FrameSample(frame_bgr.copy())
        self._frame_samples[frame_index] = sample
        self._frame_order.append(frame_index)
        self._trim_frame_store()

    def _trim_frame_store(self) -> None:
        limit = self.max_history_frames
        if limit <= 0:
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
