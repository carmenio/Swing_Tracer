import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List


SETTINGS_FILENAME = "settings.json"


@dataclass
class GeneralSettings:
    auto_track: bool = True
    trail_length: int = 10
    viewport_start: float = 0.0
    viewport_end: float = 100.0
    theme: str = "Dark"


@dataclass
class TrackingSettings:
    tracking_enabled: bool = True
    smoothing_enabled: bool = True
    smoothing_alpha: float = 0.2
    max_auto_track_frames: int = 30
    direction_change_threshold: float = 35.0
    deviation_threshold: float = 18.0
    cache_size: int = 200
    history_frames: int = 300
    optical_flow_window_size: int = 21
    optical_flow_pyramid: int = 3
    optical_flow_term_count: int = 30
    optical_flow_term_epsilon: float = 0.01
    optical_flow_feature_quality: float = 0.4
    optical_flow_min_distance: float = 1.5
    optical_flow_min_eig_threshold: float = 1e-4
    optical_flow_batch_size: int = 8
    issue_confidence_threshold: float = 0.6
    resolved_confidence_threshold: float = 0.8
    truncate_future_on_manual_set: bool = False
    baseline_mode: str = "linear"
    performance_mode: str = "Balanced"
    thread_priority: str = "Normal"
    cache_frames_enabled: bool = True


@dataclass
class TimelineSettings:
    marker_colors: Dict[str, str] = field(
        default_factory=lambda: {
            "Golf Ball": "#ffff66",
            "Club Handle": "#ff4040",
            "Club Midpoint": "#44ff44",
            "Club Toe": "#4488ff",
            "Club Heel": "#ffdd44",
        }
    )
    min_zoom_width: float = 5.0
    max_zoom_width: float = 100.0
    show_keyframes: bool = True
    show_auto_keyframes: bool = True
    show_issues: bool = True
    detailed_height: int = 40
    overview_height: int = 40


@dataclass
class PlaybackSettings:
    default_speed: str = "1x"
    loop_playback: bool = False
    auto_pause_on_issue: bool = False
    frame_step_amount: int = 1


@dataclass
class InputSettings:
    auto_track_on_click: bool = False
    pointer_snap_radius: int = 12
    gesture_sensitivity: float = 1.0
    pan_speed: float = 1.0
    scroll_inertia: float = 0.9
    auto_jump_enabled: bool = False
    auto_jump_frames: int = 2


@dataclass
class NotifySettings:
    issue_notifications: bool = True
    issue_sound: bool = False
    issue_severity_colors: Dict[str, str] = field(
        default_factory=lambda: {
            "Low": "#fbbf24",
            "Medium": "#f97316",
            "High": "#ef4444",
        }
    )
    log_level: str = "Info"
    auto_save_keyframes: bool = False


@dataclass
class DataSettings:
    default_session_folder: str = "./sessions"
    export_format: str = "JSON"
    auto_save_interval: int = 300


@dataclass
class AppSettings:
    general: GeneralSettings = field(default_factory=GeneralSettings)
    tracking: TrackingSettings = field(default_factory=TrackingSettings)
    timeline: TimelineSettings = field(default_factory=TimelineSettings)
    playback: PlaybackSettings = field(default_factory=PlaybackSettings)
    input: InputSettings = field(default_factory=InputSettings)
    notify: NotifySettings = field(default_factory=NotifySettings)
    data: DataSettings = field(default_factory=DataSettings)


class SettingsManager:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.settings = AppSettings()
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        self.settings = self._from_dict(data)

    def save(self) -> None:
        self.path.write_text(json.dumps(self._to_dict(), indent=2))

    def reset(self) -> None:
        self.settings = AppSettings()
        self.save()

    def _to_dict(self) -> Dict:
        return asdict(self.settings)

    def _from_dict(self, data: Dict) -> AppSettings:
        def merge(default_cls, section):
            instance = default_cls()
            if isinstance(section, dict):
                for key, value in section.items():
                    if hasattr(instance, key):
                        setattr(instance, key, value)
            return instance

        settings = AppSettings()
        if not isinstance(data, dict):
            return settings
        if "general" in data:
            settings.general = merge(GeneralSettings, data["general"])
        if "tracking" in data:
            settings.tracking = merge(TrackingSettings, data["tracking"])
        if "timeline" in data:
            settings.timeline = merge(TimelineSettings, data["timeline"])
        if "playback" in data:
            settings.playback = merge(PlaybackSettings, data["playback"])
        if "input" in data:
            settings.input = merge(InputSettings, data["input"])
        if "notify" in data:
            settings.notify = merge(NotifySettings, data["notify"])
        if "data" in data:
            settings.data = merge(DataSettings, data["data"])
        return settings


def get_settings_path(root: Path) -> Path:
    return root / SETTINGS_FILENAME
