from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2

from .settings import AppSettings, SettingsManager, get_settings_path
from .tracking import CustomPointTracker
from .tracking.custom_points import CustomPointFrameResult
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
        self.auto_tracking_enabled: bool = self.settings.general.auto_track

        self.video_player = VideoPlayer()
        self.custom_tracker = CustomPointTracker()
        self.custom_tracker.update_from_settings(self.settings.tracking)

        self.current_frame_bgr: Optional[cv2.typing.MatLike] = None
        self.preprocessed_custom_results: Dict[int, CustomPointFrameResult] = {}
        self.use_preprocessed_results: bool = False

        self.active_point: Optional[str] = None
        self.viewport_range: Tuple[float, float] = (
            self.settings.general.viewport_start,
            self.settings.general.viewport_end,
        )

    def reset_preprocessing(self) -> None:
        self.preprocessed_custom_results.clear()
        self.use_preprocessed_results = False

    def set_active_point(self, point_name: Optional[str]) -> None:
        self.active_point = point_name

    def set_viewport_range(self, start: float, end: float) -> None:
        self.viewport_range = (start, end)
