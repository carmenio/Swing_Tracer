from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class VideoMetadata:
    frame_count: int
    fps: float
    frame_duration_ms: int
    frame_size: Tuple[int, int]


class VideoPlayer:
    def __init__(self) -> None:
        self._capture: Optional[cv2.VideoCapture] = None
        self._metadata = VideoMetadata(frame_count=0, fps=30.0, frame_duration_ms=33, frame_size=(0, 0))
        self.current_frame_index: int = 0
        self.current_frame: Optional[np.ndarray] = None

    def load(self, path: str) -> VideoMetadata:
        self.release()
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise ValueError("Failed to open video.")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_duration_ms = int(1000 / fps) if fps > 0 else 33
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._capture = capture
        self._metadata = VideoMetadata(
            frame_count=frame_count,
            fps=fps,
            frame_duration_ms=frame_duration_ms,
            frame_size=(width, height),
        )
        self.current_frame_index = 0
        self.current_frame = None
        return self._metadata

    @property
    def metadata(self) -> VideoMetadata:
        return self._metadata

    def is_loaded(self) -> bool:
        return self._capture is not None

    def read_first_frame(self) -> Optional[np.ndarray]:
        if not self._capture:
            return None
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, frame = self._capture.read()
        if not success:
            return None
        self.current_frame_index = 0
        self.current_frame = frame
        return frame

    def read_next(self) -> Optional[np.ndarray]:
        if not self._capture:
            return None
        success, frame = self._capture.read()
        if not success:
            return None
        self.current_frame_index += 1
        self.current_frame = frame
        return frame

    def seek(self, frame_index: int) -> Optional[np.ndarray]:
        if not self._capture:
            return None
        frame_index = max(0, min(frame_index, self._metadata.frame_count - 1))
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = self._capture.read()
        if not success:
            return None
        self.current_frame_index = frame_index
        self.current_frame = frame
        return frame

    def reset(self) -> None:
        if not self._capture:
            return
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame_index = 0
        self.current_frame = None

    def release(self) -> None:
        if self._capture:
            self._capture.release()
        self._capture = None
        self.current_frame = None
        self.current_frame_index = 0
        self._metadata = VideoMetadata(frame_count=0, fps=30.0, frame_duration_ms=33, frame_size=(0, 0))
