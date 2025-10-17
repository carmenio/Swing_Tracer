from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import math

import cv2
import numpy as np

from ..entities import Point2D
from .provisional import ProvisionalChain, build_provisional_chain


def _distance(a: Point2D, b: Point2D) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def map_deviation_to_colour(residual: float, threshold_pixels: float) -> str:
    if residual < 2.0:
        return "#00ff00"
    if residual < 6.0:
        return "#66ff00"
    if residual < threshold_pixels:
        return "#ffaa00"
    return "#ff0000"


@dataclass
class KeyFrame:
    frame: int
    pos: Point2D
    type: str
    conf: float


@dataclass
class TrackingSegment:
    start_key: KeyFrame
    end_key: KeyFrame
    tracked_positions: Dict[int, Point2D] = field(default_factory=dict)
    residuals: Dict[int, float] = field(default_factory=dict)
    gradient_colours: Dict[int, str] = field(default_factory=dict)
    provisional_points: List[KeyFrame] = field(default_factory=list)
    accepted: bool = False
    entity_colour: Optional[str] = None

    def color_stops(self) -> List[Tuple[int, str]]:
        stops = sorted(self.gradient_colours.items())
        if not any(frame == self.start_key.frame for frame, _ in stops):
            stops.insert(0, (self.start_key.frame, self.gradient_colours.get(self.start_key.frame, "#00ff00")))
        if not any(frame == self.end_key.frame for frame, _ in stops):
            stops.append((self.end_key.frame, self.gradient_colours.get(self.end_key.frame, "#00ff00")))
        return stops


@dataclass
class OpticalFlowResult:
    positions: Dict[int, Point2D]
    confidences: Dict[int, float]
    fb_errors: Dict[int, float]


@dataclass
class SegmentBuildResult:
    span_start: int
    span_end: int
    chain: ProvisionalChain
    segments: List[TrackingSegment]
    optical_flow: OpticalFlowResult


class SegmentBuilder:
    def __init__(
        self,
        *,
        baseline_mode: str,
        threshold_confidence: float,
        min_threshold_pixels: float,
        threshold_scale: float,
        nms_window: int,
        min_segment_len: int,
        max_provisionals: int,
        min_improvement_px: float,
        min_improvement_ratio: float,
        smoothing_window: int,
        entity_colour: Optional[str] = None,
    ) -> None:
        self.baseline_mode = baseline_mode
        self.threshold_confidence = threshold_confidence
        self.min_threshold_pixels = min_threshold_pixels
        self.threshold_scale = threshold_scale
        self.nms_window = nms_window
        self.min_segment_len = min_segment_len
        self.max_provisionals = max_provisionals
        self.min_improvement_px = min_improvement_px
        self.min_improvement_ratio = min_improvement_ratio
        self.smoothing_window = smoothing_window
        self.entity_colour = entity_colour

    def build(
        self,
        *,
        span_start: int,
        span_end: int,
        start_pos: Point2D,
        end_pos: Point2D,
        frames: Dict[int, "FrameSample"],
        accepted_frames: Set[int],
        rejected_frames: Optional[Set[int]] = None,
    ) -> Optional[SegmentBuildResult]:
        if span_end <= span_start:
            return None

        if any(frame not in frames for frame in range(span_start, span_end + 1)):
            return None

        optical_flow = self._track_with_lk(span_start, span_end, start_pos, frames)
        if span_end not in optical_flow.positions:
            csrt_flow = self._track_with_csrt(span_start, span_end, start_pos, frames)
            for frame, pos in csrt_flow.positions.items():
                optical_flow.positions.setdefault(frame, pos)
                optical_flow.confidences.setdefault(frame, csrt_flow.confidences.get(frame, 0.0))
                optical_flow.fb_errors.setdefault(frame, csrt_flow.fb_errors.get(frame, 0.0))

        if span_end not in optical_flow.positions:
            # Unable to build a complete track.
            return None

        chain = build_provisional_chain(
            span_start,
            span_end,
            start_pos,
            end_pos,
            optical_flow.positions,
            optical_flow.confidences,
            optical_flow.fb_errors,
            baseline_mode=self.baseline_mode,
            rejected_frames=rejected_frames,
            threshold_confidence=self.threshold_confidence,
            min_threshold_pixels=self.min_threshold_pixels,
            threshold_scale=self.threshold_scale,
            nms_window=self.nms_window,
            min_segment_len=self.min_segment_len,
            max_provisionals=self.max_provisionals,
            min_improvement_px=self.min_improvement_px,
            min_improvement_ratio=self.min_improvement_ratio,
            smoothing_window=self.smoothing_window,
        )

        segments = self._segments_from_chain(chain, optical_flow.positions, accepted_frames)

        return SegmentBuildResult(
            span_start=span_start,
            span_end=span_end,
            chain=chain,
            segments=segments,
            optical_flow=optical_flow,
        )

    def _track_with_lk(
        self,
        span_start: int,
        span_end: int,
        start_pos: Point2D,
        frames: Dict[int, "FrameSample"],
    ) -> OpticalFlowResult:
        positions: Dict[int, Point2D] = {span_start: start_pos}
        confidences: Dict[int, float] = {span_start: 1.0}
        fb_errors: Dict[int, float] = {span_start: 0.0}

        prev_frame = frames[span_start].gray
        prev_point = np.array([[start_pos]], dtype=np.float32)

        lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        for frame_index in range(span_start + 1, span_end + 1):
            current_sample = frames.get(frame_index)
            if current_sample is None:
                break
            current_frame = current_sample.gray
            next_pt, status, err = cv2.calcOpticalFlowPyrLK(prev_frame, current_frame, prev_point, None, **lk_params)
            if status is None or status.size == 0 or status[0][0] != 1:
                break

            fb_pt, fb_status, fb_err = cv2.calcOpticalFlowPyrLK(current_frame, prev_frame, next_pt, None, **lk_params)
            fb_distance = float(np.linalg.norm(prev_point - fb_pt)) if fb_status is not None and fb_status.size > 0 and fb_status[0][0] == 1 else float("inf")
            confidence = float(1.0 / (1.0 + fb_distance)) if math.isfinite(fb_distance) else 0.0

            pos = (float(next_pt[0][0]), float(next_pt[0][1]))
            positions[frame_index] = pos
            confidences[frame_index] = confidence
            fb_errors[frame_index] = fb_distance if math.isfinite(fb_distance) else 1e6

            prev_frame = current_frame
            prev_point = next_pt

        return OpticalFlowResult(positions=positions, confidences=confidences, fb_errors=fb_errors)

    def _track_with_csrt(
        self,
        span_start: int,
        span_end: int,
        start_pos: Point2D,
        frames: Dict[int, "FrameSample"],
    ) -> OpticalFlowResult:
        if not hasattr(cv2, "TrackerCSRT_create"):
            return OpticalFlowResult(positions={}, confidences={}, fb_errors={})

        start_sample = frames.get(span_start)
        if start_sample is None:
            return OpticalFlowResult(positions={}, confidences={}, fb_errors={})

        tracker = cv2.TrackerCSRT_create()
        bbox_size = 18.0
        bbox = (
            max(0.0, start_pos[0] - bbox_size / 2.0),
            max(0.0, start_pos[1] - bbox_size / 2.0),
            bbox_size,
            bbox_size,
        )

        initialized = tracker.init(start_sample.bgr, bbox)
        if not initialized:
            return OpticalFlowResult(positions={}, confidences={}, fb_errors={})

        positions: Dict[int, Point2D] = {}
        confidences: Dict[int, float] = {}
        fb_errors: Dict[int, float] = {}

        last_pos = start_pos
        for frame_index in range(span_start + 1, span_end + 1):
            sample = frames.get(frame_index)
            if sample is None:
                break
            ok, track_box = tracker.update(sample.bgr)
            if not ok:
                break
            cx = float(track_box[0] + track_box[2] / 2.0)
            cy = float(track_box[1] + track_box[3] / 2.0)
            last_pos = (cx, cy)
            positions[frame_index] = last_pos
            confidences[frame_index] = 0.5
            fb_errors[frame_index] = 5.0

        if span_end not in positions:
            positions[span_end] = last_pos
            confidences[span_end] = confidences.get(span_end, 0.5)
            fb_errors[span_end] = fb_errors.get(span_end, 5.0)

        return OpticalFlowResult(positions=positions, confidences=confidences, fb_errors=fb_errors)

    def _segments_from_chain(
        self,
        chain: ProvisionalChain,
        positions: Dict[int, Point2D],
        accepted_frames: Set[int],
    ) -> List[TrackingSegment]:
        segments: List[TrackingSegment] = []
        anchors = chain.anchors
        if len(anchors) < 2:
            return segments

        for idx in range(len(anchors) - 1):
            start_anchor = anchors[idx]
            end_anchor = anchors[idx + 1]

            start_key = KeyFrame(
                frame=start_anchor.frame,
                pos=start_anchor.position,
                type=start_anchor.anchor_type,
                conf=start_anchor.confidence,
            )
            end_key = KeyFrame(
                frame=end_anchor.frame,
                pos=end_anchor.position,
                type=end_anchor.anchor_type,
                conf=end_anchor.confidence,
            )

            tracked_positions = {
                frame: positions[frame]
                for frame in range(start_anchor.frame, end_anchor.frame + 1)
                if frame in positions
            }

            tail = chain.tails.get(start_anchor.frame)
            residuals = {}
            gradient = {}
            if tail:
                residuals = {frame: value for frame, value in zip(tail.frames, tail.residuals)}
                for frame, value in zip(tail.frames, tail.residuals):
                    gradient[frame] = map_deviation_to_colour(value, chain.threshold_pixels)

            gradient[start_key.frame] = map_deviation_to_colour(start_anchor.residual, chain.threshold_pixels)
            gradient[end_key.frame] = map_deviation_to_colour(end_anchor.residual, chain.threshold_pixels)

            provisional_points: List[KeyFrame] = []
            if end_anchor.anchor_type == "provisional":
                provisional_points.append(
                    KeyFrame(
                        frame=end_anchor.frame,
                        pos=end_anchor.position,
                        type="provisional",
                        conf=end_anchor.confidence,
                    )
                )

            accepted = (
                start_anchor.anchor_type == "confirmed"
                and end_anchor.anchor_type == "confirmed"
                and start_anchor.frame in accepted_frames
                and end_anchor.frame in accepted_frames
            )

            segment = TrackingSegment(
                start_key=start_key,
                end_key=end_key,
                tracked_positions=tracked_positions,
                residuals=residuals,
                gradient_colours=gradient,
                provisional_points=provisional_points,
                accepted=accepted,
                entity_colour=self.entity_colour,
            )
            segments.append(segment)

        return segments


class FrameSample:
    def __init__(self, bgr: np.ndarray) -> None:
        self.bgr = bgr
        self.gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


__all__ = [
    "KeyFrame",
    "TrackingSegment",
    "OpticalFlowResult",
    "SegmentBuildResult",
    "SegmentBuilder",
    "FrameSample",
    "map_deviation_to_colour",
]
