from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

from ..model.app_model import SwingTrackerModel
from ..model.settings import AppSettings, SettingsManager
from ..model.video import VideoPlayer

if False:  # pragma: no cover - for type checking without circular imports
    from ..view.main_window import SwingTrackerWindow


class SwingTrackerController:
    """Coordinates interactions between the view and the underlying model."""

    def __init__(self, root_path: Path) -> None:
        self._model = SwingTrackerModel(root_path)
        self.view: Optional["SwingTrackerWindow"] = None

    def set_view(self, view: "SwingTrackerWindow") -> None:
        self.view = view

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def settings_manager(self) -> SettingsManager:
        return self._model.settings_manager

    @property
    def settings(self) -> AppSettings:
        return self._model.settings

    @settings.setter
    def settings(self, value: AppSettings) -> None:
        self._model.settings = value

    @property
    def video_player(self) -> VideoPlayer:
        return self._model.video_player

    @property
    def custom_tracker(self):  # type: ignore[override]
        return self._model.custom_tracker

    @property
    def tracking_manager(self):
        return self._model.tracking_manager

    def set_tracking_video_path(self, path: Optional[Path]) -> None:
        self._model.set_tracking_video_path(path)

    @property
    def current_frame_bgr(self) -> Optional[Any]:
        return self._model.current_frame_bgr

    @current_frame_bgr.setter
    def current_frame_bgr(self, frame: Optional[Any]) -> None:
        self._model.current_frame_bgr = frame

    @property
    def active_point(self) -> Optional[str]:
        return self._model.active_point

    @active_point.setter
    def active_point(self, point_name: Optional[str]) -> None:
        self._model.active_point = point_name

    @property
    def viewport_range(self) -> Tuple[float, float]:
        return self._model.viewport_range

    @viewport_range.setter
    def viewport_range(self, value: Tuple[float, float]) -> None:
        self._model.viewport_range = value

    @property
    def point_definitions(self) -> Dict[str, Tuple[int, int, int]]:
        return self._model.POINT_DEFINITIONS

    def refresh_settings(self) -> None:
        """Reload settings from disk and update the model."""
        self.settings_manager.load()
        self.settings = self.settings_manager.settings
