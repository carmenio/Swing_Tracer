from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import threading

from .settings import AppSettings, SettingsManager, get_settings_path
from .tracking import CustomPointTracker
from .video import VideoPlayer


class SwingTrackerModel:
    """Encapsulates the non-UI state for the Swing Tracker application."""

    POINT_DEFINITIONS: Dict[str, Tuple[int, int, int]] = {
        "Golf Ball": (255, 255, 0),
        "Club Handle": (255, 64, 64),
        "Club Midpoint": (64, 200, 64),
        "Club Toe": (64, 128, 255),
        "Club Heel": (255, 200, 64),
    }

    def __init__(self, root_path: Path) -> None:
        self.settings_manager = SettingsManager(get_settings_path(root_path))
        self.settings: AppSettings = self.settings_manager.settings

        self.video_player = VideoPlayer()
        self.custom_tracker = CustomPointTracker()
        self.custom_tracker.update_from_settings(self.settings.tracking)
        self.custom_tracker.set_frame_loader(self._fetch_tracking_frame)

        self.current_frame_bgr: Optional[cv2.typing.MatLike] = None

        self.active_point: Optional[str] = None
        self.viewport_range: Tuple[float, float] = (
            self.settings.general.viewport_start,
            self.settings.general.viewport_end,
        )
        self._tracking_capture: Optional[cv2.VideoCapture] = None
        self._tracking_capture_lock = threading.RLock()

    def set_active_point(self, point_name: Optional[str]) -> None:
        self.active_point = point_name

    def set_viewport_range(self, start: float, end: float) -> None:
        self.viewport_range = (start, end)

    def set_tracking_video_path(self, path: Optional[Path]) -> None:
        with self._tracking_capture_lock:
            if self._tracking_capture is not None:
                self._tracking_capture.release()
                self._tracking_capture = None

            if path is None:
                return

            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                return

            try:
                if hasattr(cv2, "CAP_PROP_HW_ACCELERATION") and hasattr(cv2, "VIDEO_ACCELERATION_ANY"):
                    capture.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
            except Exception:
                pass

            self._tracking_capture = capture

    def _fetch_tracking_frame(self, frame_index: int):
        with self._tracking_capture_lock:
            capture = self._tracking_capture
            if capture is None:
                return None
            if frame_index < 0:
                return None
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success or frame is None:
                return None
            return frame.copy()
