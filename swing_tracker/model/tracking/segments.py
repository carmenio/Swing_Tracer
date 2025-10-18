from __future__ import annotations

# This module handles optical flow tracking and segment building.
# It provides functions and classes to compute tracking segments from video frames,
# where each segment is defined by key frames and contains interpolated gradient colors
# that reflect deviations in tracking accuracy.

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import math

import cv2
import numpy as np

from ..entities import Point2D
from .provisional import ProvisionalChain, build_provisional_chain


def _distance(a: Point2D, b: Point2D) -> float:
    """
    Calculate the Euclidean distance between two 2D points.
    
    Args:
        a (Point2D): First point.
        b (Point2D): Second point.
    
    Returns:
        float: The Euclidean distance.
    """
    return math.hypot(a[0] - b[0], a[1] - b[1])


def map_deviation_to_colour(residual: float, threshold_pixels: float) -> str:
    """
    Map a tracking deviation (residual) to a hex color code.
    
    The color reflects the quality of tracking:
      - Green for very low deviation.
      - Yellowish shades for moderate deviation.
      - Red for high deviation.
    
    Args:
        residual (float): The tracking error.
        threshold_pixels (float): The pixel threshold for deviation.
    
    Returns:
        str: A hex color string (e.g. "#00ff00").
    """
    if residual < 2.0:
        return "#00ff00"  # Excellent tracking
    if residual < 6.0:
        return "#66ff00"  # Good tracking
    if residual < threshold_pixels:
        return "#ffaa00"  # Fair tracking
    return "#ff0000"      # Poor tracking

def interpolate_color(hex1: str, hex2: str, t: float) -> str:
    """
    Linearly interpolate between two hex color codes.
    
    This is useful for generating a smooth gradient between two colors.
    
    Args:
        hex1 (str): Starting hex color (e.g., "#00ff00").
        hex2 (str): Ending hex color (e.g., "#ff0000").
        t (float): Interpolation factor where 0 returns hex1 and 1 returns hex2.
        
    Returns:
        str: The interpolated hex color.
    """
    # Remove '#' and convert color components from hex to integers
    r1, g1, b1 = int(hex1[1:3], 16), int(hex1[3:5], 16), int(hex1[5:7], 16)
    r2, g2, b2 = int(hex2[1:3], 16), int(hex2[3:5], 16), int(hex2[5:7], 16)
    # Interpolate each color channel independently
    r = round(r1 + (r2 - r1) * t)
    g = round(g1 + (g2 - g1) * t)
    b = round(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


@dataclass
class KeyFrame:
    """
    Represents a key frame in the tracking sequence.
    
    Attributes:
        frame (int): The frame number.
        pos (Point2D): The position (x, y) of the key point.
        type (str): The type of key frame (e.g., "confirmed", "provisional", "difference").
        conf (float): The confidence level of the key frame.
    """
    frame: int
    pos: Point2D
    type: str
    conf: float


@dataclass
class TrackingSegment:
    """
    Defines a tracking segment between two key frames.
    
    Contains the tracking data across frames, including positions,
    residual errors, gradient colors for visualization, and provisional key points.
    
    Attributes:
        start_key (KeyFrame): The starting key frame.
        end_key (KeyFrame): The ending key frame.
        tracked_positions (Dict[int, Point2D]): Positions of the tracked point for each frame.
        residuals (Dict[int, float]): Residual errors for each frame.
        gradient_colours (Dict[int, str]): Color codes representing tracking quality.
        provisional_points (List[KeyFrame]): Any extra points (e.g., intermediate differences).
        accepted (bool): Whether the segment has been accepted.
        entity_colour (Optional[str]): A specific color associated with this tracking entity.
    """
    start_key: KeyFrame
    end_key: KeyFrame
    tracked_positions: Dict[int, Point2D] = field(default_factory=dict)
    residuals: Dict[int, float] = field(default_factory=dict)
    gradient_colours: Dict[int, str] = field(default_factory=dict)
    provisional_points: List[KeyFrame] = field(default_factory=list)
    accepted: bool = False
    entity_colour: Optional[str] = None
    point_name: Optional[str] = None

    def color_stops(self) -> List[Tuple[int, str]]:
        """
        Generate a sorted list of gradient color stops for visualization.

        Ensures that the start and end key frames are included in the color stops.

        Returns:
            List[Tuple[int, str]]: A list of tuples where each tuple consists of a frame number and its associated color.
        """
        stops = sorted(self.gradient_colours.items())
        if not any(frame == self.start_key.frame for frame, _ in stops):
            stops.insert(0, (self.start_key.frame, self.gradient_colours.get(self.start_key.frame, "#00ff00")))
        if not any(frame == self.end_key.frame for frame, _ in stops):
            stops.append((self.end_key.frame, self.gradient_colours.get(self.end_key.frame, "#00ff00")))
        return stops


@dataclass
class OpticalFlowResult:
    """
    Stores the results of optical flow tracking.
    
    Attributes:
        positions (Dict[int, Point2D]): The position of the tracked point at each frame.
        confidences (Dict[int, float]): The confidence level for each tracking result.
        fb_errors (Dict[int, float]): The forward-backward error for each frame.
    """
    positions: Dict[int, Point2D]
    confidences: Dict[int, float]
    fb_errors: Dict[int, float]


@dataclass
class SegmentBuildResult:
    """
    Represents the complete result of building tracking segments.
    
    Attributes:
        span_start (int): The starting frame of the segment.
        span_end (int): The ending frame of the segment.
        chain (ProvisionalChain): The chain of provisional points used to build segments.
        segments (List[TrackingSegment]): The list of calculated tracking segments.
        optical_flow (OpticalFlowResult): The optical flow tracking results.
    """
    span_start: int
    span_end: int
    chain: ProvisionalChain
    segments: List[TrackingSegment]
    optical_flow: OpticalFlowResult


class SegmentBuilder:
    """
    Builds tracking segments from a sequence of video frames using optical flow.
    
    The builder uses key frame data and optical flow tracking results to generate segments.
    
    Attributes:
        baseline_mode (str): Mode used for baseline computation.
        threshold_confidence (float): Minimum confidence threshold for valid tracking.
        min_threshold_pixels (float): Pixel threshold for considering a deviation.
        threshold_scale (float): Scaling factor for threshold adjustments.
        nms_window (int): Non-maximum suppression window size.
        min_segment_len (int): Minimum length of a valid segment.
        max_provisionals (int): Maximum number of provisional points allowed.
        min_improvement_px (float): Minimum pixel improvement required between segments.
        min_improvement_ratio (float): Minimum improvement ratio.
        smoothing_window (int): Window size for smoothing trajectory data.
        entity_colour (Optional[str]): Specific color for the tracked entity.
    """
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
        lk_window_size: Tuple[int, int] = (21, 21),
        lk_max_level: int = 3,
        lk_term_count: int = 30,
        lk_term_epsilon: float = 0.01,
        lk_min_eig_threshold: float = 1e-4,
        feature_quality_threshold: float = 0.4,
        min_track_distance: float = 0.0,
        batch_size: int = 8,
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
        if isinstance(lk_window_size, tuple):
            width = max(5, int(lk_window_size[0]))
            height = max(5, int(lk_window_size[1]))
        else:
            width = max(5, int(lk_window_size))
            height = width
        if width % 2 == 0:
            width += 1
        if height % 2 == 0:
            height += 1
        self.lk_window_size = (width, height)
        self.lk_max_level = max(0, int(lk_max_level))
        self.lk_term_count = max(1, int(lk_term_count))
        self.lk_term_epsilon = float(max(1e-7, lk_term_epsilon))
        self.lk_min_eig_threshold = float(max(1e-8, lk_min_eig_threshold))
        self.feature_quality_threshold = float(max(0.0, min(1.0, feature_quality_threshold)))
        self.min_track_distance = float(max(0.0, min_track_distance))
        self.batch_size = max(1, int(batch_size))

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
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Optional[SegmentBuildResult]:
        """
        Build tracking segments between a start and end frame using optical flow data.
        
        Validates the input frames and computes tracking data. If successful, it builds 
        a provisional chain and converts it into a list of tracking segments.
        
        Args:
            span_start (int): Starting frame number.
            span_end (int): Ending frame number.
            start_pos (Point2D): Starting position of the tracked point.
            end_pos (Point2D): Ending position of the tracked point.
            frames (Dict[int, "FrameSample"]): Dictionary of frames.
            accepted_frames (Set[int]): Set of frame numbers that are accepted.
            rejected_frames (Optional[Set[int]]): Set of frame numbers that are rejected.
        
        Returns:
            Optional[SegmentBuildResult]: The resultant tracking segments and related data,
            or None if tracking fails.
        """
        if span_end <= span_start:
            return None
        
        if any(frame not in frames for frame in range(span_start, span_end + 1)):
            return None

        optical_flow = self._track_with_lk(
            span_start,
            span_end,
            start_pos,
            frames,
            progress_callback=progress_callback,
        )
        # if span_end not in optical_flow.positions:
        #     # Try using CSRT tracking if Lucas-Kanade fails.
        #     csrt_flow = self._track_with_csrt(span_start, span_end, start_pos, frames)
        #     for frame, pos in csrt_flow.positions.items():
        #         optical_flow.positions.setdefault(frame, pos)
        #         optical_flow.confidences.setdefault(frame, csrt_flow.confidences.get(frame, 0.0))
        #         optical_flow.fb_errors.setdefault(frame, csrt_flow.fb_errors.get(frame, 0.0))

        if span_end not in optical_flow.positions:
            # Unable to build a complete tracking.
            return None

        # Build a provisional chain using the optical flow data.
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

        # Convert the chain into tracking segments.
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
        *,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> OpticalFlowResult:
        """
        Perform Lucas-Kanade optical flow tracking between frames.
        
        Tracks a point starting at 'start_pos' from 'span_start' to 'span_end',
        updating its position based on optical flow.
        
        Args:
            span_start (int): The starting frame.
            span_end (int): The ending frame.
            start_pos (Point2D): Initial position of the point.
            frames (Dict[int, "FrameSample"]): Dictionary mapping frame numbers to FrameSample objects.
        
        Returns:
            OpticalFlowResult: Contains tracked positions, confidences, and error metrics.
        """
        positions: Dict[int, Point2D] = {span_start: start_pos}
        confidences: Dict[int, float] = {span_start: 1.0}
        fb_errors: Dict[int, float] = {span_start: 0.0}

        prev_sample = frames.get(span_start)
        if prev_sample is None:
            return OpticalFlowResult(positions=positions, confidences=confidences, fb_errors=fb_errors)
        prev_frame = prev_sample.gray
        prev_point = np.array([[start_pos]], dtype=np.float32)

        lk_params = dict(
            winSize=self.lk_window_size,
            maxLevel=self.lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.lk_term_count, self.lk_term_epsilon),
            minEigThreshold=self.lk_min_eig_threshold,
        )

        total_frames = max(1, span_end - span_start)
        if progress_callback is not None:
            progress_callback(0.0)

        batch_counter = 0
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

            # Extract the (x, y) position from the optical flow output
            pos = (float(next_pt[0, 0, 0]), float(next_pt[0, 0, 1]))
            if self.min_track_distance > 0.0 and frame_index > span_start:
                displacement = float(np.linalg.norm(next_pt - prev_point))
                if displacement < self.min_track_distance:
                    confidence *= max(0.25, displacement / max(self.min_track_distance, 1.0))
            if confidence < self.feature_quality_threshold:
                break
            positions[frame_index] = pos
            confidences[frame_index] = confidence
            fb_errors[frame_index] = fb_distance if math.isfinite(fb_distance) else 1e6

            prev_frame = current_frame
            prev_point = next_pt

            batch_counter += 1
            if progress_callback is not None and (batch_counter >= self.batch_size or frame_index == span_end):
                progress = (frame_index - span_start) / total_frames
                progress_callback(min(1.0, max(0.0, progress)))
                batch_counter = 0

        if progress_callback is not None:
            progress_callback(1.0)

        return OpticalFlowResult(positions=positions, confidences=confidences, fb_errors=fb_errors)

    def _track_with_csrt(
        self,
        span_start: int,
        span_end: int,
        start_pos: Point2D,
        frames: Dict[int, "FrameSample"],
    ) -> OpticalFlowResult:
        """
        Perform CSRT-based tracking as a fallback method.
        
        Uses OpenCV's CSRT tracker to update the position of the point if Lucas-Kanade fails.
        
        Args:
            span_start (int): The starting frame.
            span_end (int): The ending frame.
            start_pos (Point2D): Initial position of the point.
            frames (Dict[int, "FrameSample"]): Dictionary of frame samples.
        
        Returns:
            OpticalFlowResult: Tracking results via CSRT methodology.
        """
        if not hasattr(cv2, "TrackerCSRT_create"):
            return OpticalFlowResult(positions={}, confidences={}, fb_errors={})

        start_sample = frames.get(span_start)
        if start_sample is None:
            return OpticalFlowResult(positions={}, confidences={}, fb_errors={})

        tracker = cv2.TrackerCSRT_create()
        bbox_size = 18.0
        bbox = tuple(map(int, (
            max(0.0, start_pos[0] - bbox_size / 2.0),
            max(0.0, start_pos[1] - bbox_size / 2.0),
            bbox_size,
            bbox_size,
        )))

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
        """
        Convert a provisional chain into a list of tracking segments.
        
        This method creates segments by pairing consecutive key frames,
        interpolates gradient colors between them, and adds any intermediate
        "difference" key points if needed.
        
        Args:
            chain (ProvisionalChain): The chain built from tracking data.
            positions (Dict[int, Point2D]): Tracked positions per frame.
            accepted_frames (Set[int]): Frames accepted for a valid segment.
        
        Returns:
            List[TrackingSegment]: A list of tracking segments.
        """
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

            # Compute gradient colors at the start and end key frames based on their residual errors.
            gradient[start_key.frame] = map_deviation_to_colour(start_anchor.residual, chain.threshold_pixels)
            gradient[end_key.frame] = map_deviation_to_colour(end_anchor.residual, chain.threshold_pixels)
            
            # Interpolate gradient colors for every frame between the start and end key frames.
            # This ensures a visible gradient line rather than just fixed point colors.
            for frame in range(start_key.frame + 1, end_key.frame):
                t = (frame - start_key.frame) / (end_key.frame - start_key.frame)
                start_color = gradient[start_key.frame]
                end_color = gradient[end_key.frame]
                gradient[frame] = interpolate_color(start_color, end_color, t)
            
            # Optionally, add an intermediate key point ("difference") if the residual difference is significant.
            if abs(start_anchor.residual - end_anchor.residual) > 0.1:
                mid_frame = (start_anchor.frame + end_anchor.frame) // 2
                if mid_frame in positions:
                    mid_pos = positions[mid_frame]
                else:
                    mid_pos = ((start_key.pos[0] + end_key.pos[0]) / 2, (start_key.pos[1] + end_key.pos[1]) / 2)
                mid_conf = (start_anchor.confidence + end_anchor.confidence) / 2
                # Override the interpolated gradient at the mid frame using the average residual.
                gradient[mid_frame] = map_deviation_to_colour((start_anchor.residual + end_anchor.residual) / 2, chain.threshold_pixels)
                provisional_points_extra = KeyFrame(
                    frame=mid_frame,
                    pos=mid_pos,
                    type="difference",
                    conf=mid_conf,
                )
            else:
                provisional_points_extra = None

            # Create a list for provisional (extra) key points.
            provisional_points: List[KeyFrame] = []
            # If the ending key frame is provisional, add it to the list.
            if end_anchor.anchor_type == "provisional":
                provisional_points.append(
                    KeyFrame(
                        frame=end_anchor.frame,
                        pos=end_anchor.position,
                        type="provisional",
                        conf=end_anchor.confidence,
                    )
                )
            # Also add the intermediate "difference" key point if it was generated.
            if provisional_points_extra is not None:
                provisional_points.append(provisional_points_extra)

            # Determine if the segment is accepted by verifying that both key frames are confirmed
            # and that their corresponding frames are in the accepted set.
            accepted = (
                start_anchor.anchor_type == "confirmed"
                and end_anchor.anchor_type == "confirmed"
                and start_anchor.frame in accepted_frames
                and end_anchor.frame in accepted_frames
            )

            # Create a TrackingSegment instance with all the gathered data.
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
    """
    Represents a video frame sample.
    
    Contains the original BGR image and its grayscale conversion,
    which is used for optical flow calculations.
    """
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
