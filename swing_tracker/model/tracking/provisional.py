from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from collections import deque
from bisect import bisect_right
import math
import statistics

from ..entities import Point2D


AnchorType = str


@dataclass
class AnchorMetrics:
    frame: int
    position: Point2D
    residual: float
    confidence: float
    fb_error: float
    anchor_type: AnchorType = "confirmed"


@dataclass
class TailMetrics:
    start_frame: int
    end_frame: int
    frames: List[int]
    residuals: List[float]
    confidences: List[float]

    def max_residual(self) -> float:
        return max(self.residuals) if self.residuals else 0.0


@dataclass
class ProvisionalCandidate:
    anchor: AnchorMetrics
    tail: Optional[TailMetrics]


@dataclass
class ProvisionalChain:
    span_start: int
    span_end: int
    anchors: List[AnchorMetrics]
    tails: Dict[int, TailMetrics] = field(default_factory=dict)
    baseline_mode: str = "linear"
    threshold_pixels: float = 6.0
    threshold_confidence: float = 0.6
    rejected_frames: Set[int] = field(default_factory=set)

    def provisionals(self) -> List[ProvisionalCandidate]:
        provisionals: List[ProvisionalCandidate] = []
        for anchor in self.anchors:
            if anchor.anchor_type != "provisional":
                continue
            provisionals.append(
                ProvisionalCandidate(anchor=anchor, tail=self.tails.get(anchor.frame))
            )
        return provisionals

    def total_provisionals(self) -> int:
        return sum(1 for anchor in self.anchors if anchor.anchor_type == "provisional")


def _median_filter(values: Sequence[float], window: int = 5) -> List[float]:
    if not values:
        return []
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded: List[float] = [values[0]] * radius + list(values) + [values[-1]] * radius
    filtered: List[float] = []
    for idx in range(radius, len(padded) - radius):
        segment = padded[idx - radius : idx + radius + 1]
        filtered.append(float(statistics.median(segment)))
    return filtered


def _lerp(a: Point2D, b: Point2D, t: float) -> Point2D:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _distance(a: Point2D, b: Point2D) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _baseline_points(
    start_frame: int,
    end_frame: int,
    start_pos: Point2D,
    end_pos: Point2D,
    frames: Sequence[int],
    mode: str,
    observed: Dict[int, Point2D],
) -> List[Point2D]:
    span = max(1, end_frame - start_frame)
    if not frames:
        return []

    if mode not in {"linear", "cubic", "const-velocity"}:
        mode = "linear"

    if mode == "const-velocity":
        # Estimate a crude velocity from the first available observation.
        first_obs_frame = next((f for f in frames if f in observed), None)
        velocity = None
        if first_obs_frame is not None:
            obs = observed[first_obs_frame]
            dt = max(1, first_obs_frame - start_frame)
            velocity = ((obs[0] - start_pos[0]) / dt, (obs[1] - start_pos[1]) / dt)
        else:
            velocity = ((end_pos[0] - start_pos[0]) / span, (end_pos[1] - start_pos[1]) / span)
        points: List[Point2D] = []
        for frame in frames:
            t = frame - start_frame
            x = start_pos[0] + velocity[0] * t
            y = start_pos[1] + velocity[1] * t
            blend = min(1.0, max(0.0, frame - start_frame) / span)
            target = _lerp((x, y), end_pos, blend)
            points.append(target)
        return points

    if mode == "cubic":
        # Cubic Hermite interpolation with finite differences as tangents.
        def tangent(frame: int, default: Point2D) -> Point2D:
            prev_frame = frame - 1
            next_frame = frame + 1
            prev_pos = observed.get(prev_frame, default)
            next_pos = observed.get(next_frame, default)
            return ((next_pos[0] - prev_pos[0]) * 0.5, (next_pos[1] - prev_pos[1]) * 0.5)

        m0 = tangent(start_frame, start_pos)
        m1 = tangent(end_frame, end_pos)
        points = []
        for frame in frames:
            t = (frame - start_frame) / span
            t2 = t * t
            t3 = t2 * t
            h00 = 2 * t3 - 3 * t2 + 1
            h10 = t3 - 2 * t2 + t
            h01 = -2 * t3 + 3 * t2
            h11 = t3 - t2
            x = h00 * start_pos[0] + h10 * m0[0] + h01 * end_pos[0] + h11 * m1[0]
            y = h00 * start_pos[1] + h10 * m0[1] + h01 * end_pos[1] + h11 * m1[1]
            points.append((x, y))
        return points

    # Default linear interpolation.
    return [_lerp(start_pos, end_pos, (frame - start_frame) / span) for frame in frames]


def _first_deviation(
    frames: Sequence[int],
    residuals: Sequence[float],
    confidences: Sequence[float],
    threshold_pixels: float,
    threshold_confidence: float,
    nms_window: int,
    rejected: Iterable[int],
) -> Optional[Tuple[int, float]]:
    rejected_set = set(rejected)
    violating_indices: List[int] = []
    for idx, frame in enumerate(frames):
        if frame in rejected_set:
            continue
        if confidences[idx] < threshold_confidence:
            continue
        if residuals[idx] < threshold_pixels:
            continue
        violating_indices.append(idx)
    if not violating_indices:
        return None

    clusters: List[List[int]] = []
    current_cluster: List[int] = [violating_indices[0]]
    for idx in violating_indices[1:]:
        if frames[idx] - frames[current_cluster[-1]] <= nms_window:
            current_cluster.append(idx)
        else:
            clusters.append(current_cluster)
            current_cluster = [idx]
    clusters.append(current_cluster)

    first_cluster = clusters[0]
    peak_idx = max(first_cluster, key=lambda i: residuals[i])
    return frames[peak_idx], residuals[peak_idx]


def build_provisional_chain(
    span_start: int,
    span_end: int,
    start_pos: Point2D,
    end_pos: Point2D,
    positions: Dict[int, Point2D],
    confidences: Dict[int, float],
    fb_errors: Dict[int, float],
    *,
    baseline_mode: str = "linear",
    rejected_frames: Optional[Set[int]] = None,
    threshold_confidence: float = 0.6,
    min_threshold_pixels: float = 6.0,
    threshold_scale: float = 0.04,
    nms_window: int = 8,
    min_segment_len: int = 6,
    max_provisionals: int = 5,
    min_improvement_px: float = 2.0,
    min_improvement_ratio: float = 0.2,
    smoothing_window: int = 5,
) -> ProvisionalChain:
    rejected_frames = rejected_frames or set()
    interval_extent = _distance(start_pos, end_pos)
    threshold_pixels = max(min_threshold_pixels, threshold_scale * interval_extent)

    anchors: List[AnchorMetrics] = [
        AnchorMetrics(
            frame=span_start,
            position=start_pos,
            residual=0.0,
            confidence=confidences.get(span_start, 1.0),
            fb_error=fb_errors.get(span_start, 0.0),
            anchor_type="confirmed",
        ),
        AnchorMetrics(
            frame=span_end,
            position=end_pos,
            residual=0.0,
            confidence=confidences.get(span_end, 1.0),
            fb_error=fb_errors.get(span_end, 0.0),
            anchor_type="confirmed",
        ),
    ]

    tails: Dict[int, TailMetrics] = {}
    frame_order: List[int] = [span_start, span_end]

    queue = deque([(0, 1, None)])  # (start_idx, end_idx, prev_tail_residuals)

    while queue:
        start_idx, end_idx, prev_tail_residuals = queue.popleft()
        start_anchor = anchors[start_idx]
        end_anchor = anchors[end_idx]

        segment_frames = [
            frame
            for frame in range(start_anchor.frame + 1, end_anchor.frame)
            if frame in positions
        ]
        if not segment_frames:
            tails[start_anchor.frame] = TailMetrics(
                start_frame=start_anchor.frame,
                end_frame=end_anchor.frame,
                frames=[],
                residuals=[],
                confidences=[],
            )
            continue

        baseline_points = _baseline_points(
            start_anchor.frame,
            end_anchor.frame,
            start_anchor.position,
            end_anchor.position,
            segment_frames,
            baseline_mode,
            positions,
        )

        raw_residuals: List[float] = []
        conf_series: List[float] = []
        for frame, baseline in zip(segment_frames, baseline_points):
            track_pos = positions.get(frame)
            if track_pos is None:
                raw_residuals.append(0.0)
                conf_series.append(0.0)
                continue
            raw_residuals.append(_distance(track_pos, baseline))
            conf_series.append(confidences.get(frame, 0.0))

        smoothed_residuals = _median_filter(raw_residuals, smoothing_window)
        tails[start_anchor.frame] = TailMetrics(
            start_frame=start_anchor.frame,
            end_frame=end_anchor.frame,
            frames=list(segment_frames),
            residuals=smoothed_residuals,
            confidences=conf_series,
        )

        if prev_tail_residuals is not None and start_anchor.anchor_type == "provisional":
            prev_max = max(prev_tail_residuals) if prev_tail_residuals else 0.0
            new_max = max(smoothed_residuals) if smoothed_residuals else 0.0
            improvement = prev_max - new_max
            improvement_ratio = improvement / prev_max if prev_max > 1e-5 else (float("inf") if improvement > 0 else 0.0)
            if improvement < min_improvement_px and improvement_ratio < min_improvement_ratio:
                removed_anchor = anchors.pop(start_idx)
                frame_order.remove(removed_anchor.frame)
                upstream_idx = max(0, start_idx - 1)
                downstream_idx = max(0, end_idx - 1)
                queue.appendleft((upstream_idx, downstream_idx, None))
                tails.pop(removed_anchor.frame, None)
                continue

        if sum(1 for anchor in anchors if anchor.anchor_type == "provisional") >= max_provisionals:
            continue

        deviation = _first_deviation(
            segment_frames,
            smoothed_residuals,
            conf_series,
            threshold_pixels,
            threshold_confidence,
            nms_window,
            rejected_frames,
        )

        if deviation is None:
            continue

        deviation_frame, deviation_value = deviation
        if end_anchor.frame - deviation_frame < min_segment_len:
            continue

        candidate_position = positions.get(deviation_frame)
        if candidate_position is None:
            continue

        if deviation_frame not in segment_frames:
            continue

        candidate_index = segment_frames.index(deviation_frame)

        # Update existing tail to stop at the provisional frame.
        tails[start_anchor.frame] = TailMetrics(
            start_frame=start_anchor.frame,
            end_frame=deviation_frame,
            frames=segment_frames[: candidate_index + 1],
            residuals=smoothed_residuals[: candidate_index + 1],
            confidences=conf_series[: candidate_index + 1],
        )

        confidence_at_frame = conf_series[candidate_index]
        fb_error = fb_errors.get(deviation_frame, 0.0)
        new_anchor = AnchorMetrics(
            frame=deviation_frame,
            position=candidate_position,
            residual=deviation_value,
            confidence=confidence_at_frame,
            fb_error=fb_error,
            anchor_type="provisional",
        )

        insert_pos = bisect_right(frame_order, deviation_frame)
        anchors.insert(insert_pos, new_anchor)
        frame_order.insert(insert_pos, deviation_frame)

        # Adjust downstream index after insertion.
        end_idx = anchors.index(end_anchor)

        tail_residuals_after = smoothed_residuals[candidate_index + 1 :]

        if tail_residuals_after:
            queue.append((insert_pos, end_idx, tail_residuals_after))
        else:
            queue.append((insert_pos, end_idx, []))

    return ProvisionalChain(
        span_start=span_start,
        span_end=span_end,
        anchors=anchors,
        tails=tails,
        baseline_mode=baseline_mode,
        threshold_pixels=threshold_pixels,
        threshold_confidence=threshold_confidence,
        rejected_frames=set(rejected_frames),
    )
