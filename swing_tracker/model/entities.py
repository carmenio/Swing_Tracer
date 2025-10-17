from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

Point2D = Tuple[float, float]
ColorRGB = Tuple[int, int, int]


@dataclass
class TrackIssue:
    frame_index: int
    point_name: str
    confidence: float
    note: str
    hidden: bool = False


@dataclass
class LandmarkData:
    position: Point2D
    confidence: float


@dataclass
class TrackedPoint:
    name: str
    color: ColorRGB
    positions: Dict[int, Point2D] = field(default_factory=dict)
    confidence: Dict[int, float] = field(default_factory=dict)
    fb_errors: Dict[int, float] = field(default_factory=dict)
    low_confidence_frames: Set[int] = field(default_factory=set)
    keyframes: Dict[int, Point2D] = field(default_factory=dict)
    accepted_keyframes: Set[int] = field(default_factory=set)
    interpolation_cache: Dict[int, Point2D] = field(default_factory=dict)
    smoothed_position: Optional[Point2D] = None
    last_frame_index: Optional[int] = None
    absent_ranges: List[Tuple[int, int]] = field(default_factory=list)
    open_absence_start: Optional[int] = None

    def record(
        self,
        frame_index: int,
        position: Optional[Point2D],
        confidence: float,
        fb_error: Optional[float] = None,
    ) -> None:
        if position is not None:
            self.positions[frame_index] = position
            self.interpolation_cache[frame_index] = position
        else:
            self.positions.pop(frame_index, None)
            self.interpolation_cache.pop(frame_index, None)
        self.confidence[frame_index] = confidence
        if fb_error is not None:
            self.fb_errors[frame_index] = fb_error
        else:
            self.fb_errors.pop(frame_index, None)
        if confidence < 0.5:
            self.low_confidence_frames.add(frame_index)
        else:
            self.low_confidence_frames.discard(frame_index)
        self.last_frame_index = frame_index

    def history(self) -> List[Tuple[int, Point2D]]:
        return sorted(self.positions.items(), key=lambda item: item[0])

    def truncate_after(self, frame_index: int) -> None:
        frames_to_remove = [idx for idx in self.positions if idx > frame_index]
        for idx in frames_to_remove:
            self.positions.pop(idx, None)
            self.confidence.pop(idx, None)
            self.fb_errors.pop(idx, None)
            self.low_confidence_frames.discard(idx)
            self.interpolation_cache.pop(idx, None)
        keyframes_to_remove = [idx for idx in self.keyframes if idx > frame_index]
        for idx in keyframes_to_remove:
            self.keyframes.pop(idx, None)
            self.interpolation_cache.pop(idx, None)
            self.accepted_keyframes.discard(idx)
        cache_to_remove = [idx for idx in self.interpolation_cache if idx > frame_index]
        for idx in cache_to_remove:
            self.interpolation_cache.pop(idx, None)
        if self.last_frame_index is not None and self.last_frame_index > frame_index:
            remaining_frames = [idx for idx in self.positions if idx <= frame_index]
            if remaining_frames:
                self.last_frame_index = max(remaining_frames)
                self.smoothed_position = self.positions.get(self.last_frame_index)
            else:
                self.last_frame_index = None
                self.smoothed_position = None

    def clear(self) -> None:
        self.positions.clear()
        self.confidence.clear()
        self.fb_errors.clear()
        self.low_confidence_frames.clear()
        self.keyframes.clear()
        self.accepted_keyframes.clear()
        self.interpolation_cache.clear()
        self.smoothed_position = None
        self.last_frame_index = None
        self.absent_ranges.clear()
        self.open_absence_start = None

    def set_keyframe(self, frame_index: int, position: Point2D, *, accepted: bool = False) -> None:
        self.keyframes[frame_index] = position
        self.record(frame_index, position, 1.0, fb_error=None)
        self.interpolation_cache.pop(frame_index, None)
        if accepted:
            self.accepted_keyframes.add(frame_index)
        elif frame_index not in self.accepted_keyframes:
            self.accepted_keyframes.discard(frame_index)

    def keyframe_frames(self) -> List[int]:
        return sorted(self.keyframes.keys())

    def accepted_keyframe_frames(self) -> List[int]:
        return sorted(self.accepted_keyframes)

    def mark_keyframe_accepted(self, frame_index: int) -> bool:
        if frame_index not in self.keyframes:
            return False
        self.accepted_keyframes.add(frame_index)
        return True

    def remove_keyframe(self, frame_index: int) -> bool:
        if frame_index not in self.keyframes:
            return False
        self.keyframes.pop(frame_index, None)
        self.accepted_keyframes.discard(frame_index)
        self.interpolation_cache.pop(frame_index, None)
        return True

    def pending_keyframe_frames(self) -> List[int]:
        return sorted(idx for idx in self.keyframes if idx not in self.accepted_keyframes)

    def add_absence(self, start: int, end: int) -> None:
        if start > end:
            start, end = end, start
        merged: List[Tuple[int, int]] = []
        inserted = False
        for existing_start, existing_end in self.absent_ranges:
            if end < existing_start - 1:
                if not inserted:
                    merged.append((start, end))
                    inserted = True
                merged.append((existing_start, existing_end))
            elif start > existing_end + 1:
                merged.append((existing_start, existing_end))
            else:
                start = min(start, existing_start)
                end = max(end, existing_end)
        if not inserted:
            merged.append((start, end))
        self.absent_ranges = sorted(merged)

    def remove_absence_at(self, frame_index: int) -> None:
        updated: List[Tuple[int, int]] = []
        for start, end in self.absent_ranges:
            if frame_index < start or frame_index > end:
                updated.append((start, end))
                continue
            if start <= frame_index - 1:
                updated.append((start, frame_index - 1))
            if frame_index + 1 <= end:
                updated.append((frame_index + 1, end))
        self.absent_ranges = sorted((s, e) for s, e in updated if s <= e)

    def clear_absence_ranges(self) -> None:
        self.absent_ranges.clear()
        self.open_absence_start = None

    def is_absent(self, frame_index: int) -> bool:
        if self.open_absence_start is not None and frame_index >= self.open_absence_start:
            return True
        for start, end in self.absent_ranges:
            if start <= frame_index <= end:
                return True
        return False

    def end_absence_at(self, frame_index: int) -> None:
        updated: List[Tuple[int, int]] = []
        if self.open_absence_start is not None and frame_index >= self.open_absence_start:
            self.open_absence_start = None
        for start, end in self.absent_ranges:
            if frame_index < start:
                updated.append((start, end))
            elif frame_index <= end:
                if start <= frame_index - 1:
                    updated.append((start, frame_index - 1))
            else:
                updated.append((start, end))
        self.absent_ranges = [(s, e) for s, e in updated if s <= e]
