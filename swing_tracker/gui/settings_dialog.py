from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from ..settings import (
    AppSettings,
    DataSettings,
    GeneralSettings,
    InputSettings,
    NotifySettings,
    PlaybackSettings,
    SettingsManager,
    TimelineSettings,
    TrackingSettings,
)


class SettingsDialog(QtWidgets.QDialog):
    settingsApplied = QtCore.pyqtSignal(AppSettings)

    def __init__(self, manager: SettingsManager, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setMinimumSize(780, 560)
        self.manager = manager
        self.settings = AppSettings(
            general=GeneralSettings(**asdict(manager.settings.general)),
            tracking=TrackingSettings(**asdict(manager.settings.tracking)),
            timeline=TimelineSettings(**asdict(manager.settings.timeline)),
            playback=PlaybackSettings(**asdict(manager.settings.playback)),
            input=InputSettings(**asdict(manager.settings.input)),
            notify=NotifySettings(**asdict(manager.settings.notify)),
            data=DataSettings(**asdict(manager.settings.data)),
        )

        self._build_ui()
        self._populate_fields()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.setStyleSheet(
            """
            QDialog {
                background-color: #101010;
                color: #f0f0f0;
                border-radius: 10px;
            }
            QLabel {
                color: #f0f0f0;
                font-size: 12px;
            }
            QGroupBox {
                border: none;
                margin-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 0px;
                padding: 0 0 6px 0;
                color: #bfbfbf;
                font-size: 11px;
            }
            QTabBar::tab {
                background: rgba(255,255,255,0.05);
                padding: 6px 14px;
                margin-right: 6px;
                border-radius: 16px;
                color: #bdbdbd;
            }
            QTabBar::tab:selected {
                background: rgba(255,255,255,0.16);
                color: #ffffff;
            }
            QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox {
                background: #181818;
                border: 1px solid #2a2a2a;
                border-radius: 8px;
                padding: 6px;
                color: #f0f0f0;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #2b2b2b;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 18px;
                background: #f0f0f0;
                border-radius: 9px;
                margin: -6px 0;
            }
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 1px solid #444;
                border-radius: 4px;
                background: rgba(255,255,255,0.05);
            }
            QCheckBox::indicator:checked {
                background: #4ade80;
                border: 1px solid rgba(74,222,128,0.5);
            }
            QPushButton {
                background-color: rgba(255,255,255,0.08);
                border: 1px solid #2a2a2a;
                border-radius: 14px;
                padding: 8px 18px;
                color: #f0f0f0;
            }
            QPushButton:hover {
                background-color: rgba(255,255,255,0.16);
            }
            QPushButton.primary {
                background-color: #f8fafc;
                color: #0f172a;
            }
            QPushButton.primary:hover {
                background-color: #ffffff;
            }
            """
        )

        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setContentsMargins(24, 24, 24, 24)
        outer_layout.setSpacing(16)

        header_layout = QtWidgets.QVBoxLayout()
        title_label = QtWidgets.QLabel("Settings")
        title_label.setStyleSheet("font-size: 20px; font-weight: 600; color: #ffffff;")
        subtitle_label = QtWidgets.QLabel("Configure tracking, playback, and UI preferences")
        subtitle_label.setStyleSheet("color: #b0b0b0;")
        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        outer_layout.addLayout(header_layout)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setTabPosition(QtWidgets.QTabWidget.North)
        self.tabs.setDocumentMode(True)

        self._general_tab = self._wrap_with_scroll(self._build_general_tab())
        self._tracking_tab = self._wrap_with_scroll(self._build_tracking_tab())
        self._timeline_tab = self._wrap_with_scroll(self._build_timeline_tab())
        self._playback_tab = self._wrap_with_scroll(self._build_playback_tab())
        self._input_tab = self._wrap_with_scroll(self._build_input_tab())
        self._notify_tab = self._wrap_with_scroll(self._build_notify_tab())
        self._data_tab = self._wrap_with_scroll(self._build_data_tab())

        self.tabs.addTab(self._general_tab, "General")
        self.tabs.addTab(self._tracking_tab, "Tracking")
        self.tabs.addTab(self._timeline_tab, "Timeline")
        self.tabs.addTab(self._playback_tab, "Playback")
        self.tabs.addTab(self._input_tab, "Input")
        self.tabs.addTab(self._notify_tab, "Notify")
        self.tabs.addTab(self._data_tab, "Data")

        outer_layout.addWidget(self.tabs)

        footer_layout = QtWidgets.QHBoxLayout()
        footer_layout.setContentsMargins(0, 12, 0, 0)

        self.reset_button = QtWidgets.QPushButton("Reset to Defaults")
        self.reset_button.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload))
        footer_layout.addWidget(self.reset_button)
        footer_layout.addStretch(1)

        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.save_button = QtWidgets.QPushButton("Save Settings")
        self.save_button.setProperty("class", "primary")
        footer_layout.addWidget(self.cancel_button)
        footer_layout.addWidget(self.save_button)
        outer_layout.addLayout(footer_layout)

        self.reset_button.clicked.connect(self._reset_to_defaults)
        self.cancel_button.clicked.connect(self.reject)
        self.save_button.clicked.connect(self._apply_and_close)

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------
    def _build_general_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setSpacing(20)

        self.general_auto_track = QtWidgets.QCheckBox("Auto Track")
        self.general_auto_track.setToolTip("Enable forward tracking after manual keyframes.")
        layout.addWidget(self.general_auto_track)

        trail_layout = QtWidgets.QVBoxLayout()
        trail_label = QtWidgets.QLabel("Trail Length (frames)")
        trail_desc = QtWidgets.QLabel("Number of frames for on-screen point trails")
        trail_desc.setStyleSheet("color: #909090; font-size: 11px;")
        self.general_trail_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.general_trail_slider.setRange(1, 200)
        self.general_trail_value = QtWidgets.QLabel()
        self.general_trail_value.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        slider_row = QtWidgets.QHBoxLayout()
        slider_row.addWidget(self.general_trail_slider)
        slider_row.addWidget(self.general_trail_value)

        trail_layout.addWidget(trail_label)
        trail_layout.addWidget(trail_desc)
        trail_layout.addLayout(slider_row)
        layout.addLayout(trail_layout)

        viewport_group = QtWidgets.QGroupBox("Default Viewport Range (%)")
        viewport_layout = QtWidgets.QGridLayout(viewport_group)
        viewport_desc = QtWidgets.QLabel("Initial zoom span for the timelines")
        viewport_desc.setStyleSheet("color: #909090; font-size: 11px;")
        viewport_layout.addWidget(viewport_desc, 0, 0, 1, 2)

        viewport_layout.addWidget(QtWidgets.QLabel("Start"), 1, 0)
        self.general_viewport_start = QtWidgets.QDoubleSpinBox()
        self.general_viewport_start.setRange(0.0, 100.0)
        viewport_layout.addWidget(self.general_viewport_start, 1, 1)

        viewport_layout.addWidget(QtWidgets.QLabel("End"), 2, 0)
        self.general_viewport_end = QtWidgets.QDoubleSpinBox()
        self.general_viewport_end.setRange(0.0, 100.0)
        viewport_layout.addWidget(self.general_viewport_end, 2, 1)
        layout.addWidget(viewport_group)

        theme_layout = QtWidgets.QGridLayout()
        theme_layout.addWidget(QtWidgets.QLabel("Theme"), 0, 0)
        self.general_theme = QtWidgets.QComboBox()
        self.general_theme.addItems(["Dark", "Light"])
        theme_layout.addWidget(self.general_theme, 0, 1)
        layout.addLayout(theme_layout)

        controls_group = QtWidgets.QGroupBox("Session Controls")
        controls_layout = QtWidgets.QVBoxLayout(controls_group)
        controls_layout.setSpacing(10)

        self.control_reset_button = QtWidgets.QPushButton("Reset Session")
        self.control_reset_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.control_reset_button.clicked.connect(self._handle_reset_session)
        controls_layout.addWidget(self.control_reset_button)

        self.control_auto_track_button = QtWidgets.QPushButton()
        self.control_auto_track_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.control_auto_track_button.clicked.connect(self._handle_toggle_auto_track)
        controls_layout.addWidget(self.control_auto_track_button)

        self.control_clear_history_button = QtWidgets.QPushButton("Clear Selected Point History")
        self.control_clear_history_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.control_clear_history_button.clicked.connect(self._handle_clear_point_history)
        controls_layout.addWidget(self.control_clear_history_button)

        layout.addWidget(controls_group)

        layout.addStretch(1)
        self.general_trail_slider.valueChanged.connect(
            lambda value: self.general_trail_value.setText(str(value))
        )
        return widget

    def _build_tracking_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setSpacing(18)

        self.tracking_enable_smoothing = QtWidgets.QCheckBox("Enable Smoothing")
        layout.addWidget(self.tracking_enable_smoothing)

        self.tracking_truncate_future = QtWidgets.QCheckBox("Delete future points after manual keyframe")
        self.tracking_truncate_future.setToolTip(
            "If enabled, setting a manual keyframe clears tracked data on later frames."
        )
        layout.addWidget(self.tracking_truncate_future)

        # Smoothing alpha
        layout.addLayout(self._slider_row("Smoothing Alpha", "Weight for exponential smoothing (0-1)",
                                          0, 100,
                                          attr_name="tracking_smoothing_slider",
                                          value_label_name="tracking_smoothing_value",
                                          formatter=lambda v: f"{v / 100:.2f}"))

        # Max auto track frames
        layout.addLayout(self._spin_row("Max Auto Track Frames", "How many frames to evaluate after a keyframe",
                                        "tracking_max_frames", 1, 600))

        # Direction change threshold
        layout.addLayout(self._slider_row("Direction Change Threshold (degrees)",
                                          "Angle change that triggers auto keyframes/issues",
                                          0, 180,
                                          attr_name="tracking_direction_slider",
                                          value_label_name="tracking_direction_value",
                                          formatter=lambda v: f"{v}°"))

        # Deviation threshold
        layout.addLayout(self._spin_row("Deviation Threshold (pixels)",
                                        "Drift allowed before logging an issue",
                                        "tracking_deviation", 1, 200))

        # Cache/history
        layout.addLayout(self._spin_row("Cache Size", "", "tracking_cache_size", 0, 1000))
        layout.addLayout(self._spin_row("History Frames", "", "tracking_history_frames", 0, 2000))

        # Optical flow parameters grid
        flow_group = QtWidgets.QGroupBox("Optical Flow Parameters")
        flow_layout = QtWidgets.QGridLayout(flow_group)
        flow_layout.addWidget(QtWidgets.QLabel("Window Size"), 0, 0)
        self.tracking_flow_window = QtWidgets.QSpinBox()
        self.tracking_flow_window.setRange(5, 61)
        flow_layout.addWidget(self.tracking_flow_window, 0, 1)
        flow_layout.addWidget(QtWidgets.QLabel("Pyramid Level"), 0, 2)
        self.tracking_flow_pyramid = QtWidgets.QSpinBox()
        self.tracking_flow_pyramid.setRange(1, 6)
        flow_layout.addWidget(self.tracking_flow_pyramid, 0, 3)
        flow_layout.addWidget(QtWidgets.QLabel("Termination"), 0, 4)
        self.tracking_flow_termination = QtWidgets.QDoubleSpinBox()
        self.tracking_flow_termination.setDecimals(3)
        self.tracking_flow_termination.setRange(0.001, 1.0)
        flow_layout.addWidget(self.tracking_flow_termination, 0, 5)
        layout.addWidget(flow_group)

        # Confidence thresholds
        layout.addLayout(self._slider_row("Issue Threshold", "", 0, 100,
                                          attr_name="tracking_issue_slider",
                                          value_label_name="tracking_issue_value",
                                          formatter=lambda v: f"{v / 100:.2f}"))
        layout.addLayout(self._slider_row("Resolved Threshold", "", 0, 100,
                                          attr_name="tracking_resolved_slider",
                                          value_label_name="tracking_resolved_value",
                                          formatter=lambda v: f"{v / 100:.2f}"))

        layout.addStretch(1)
        return widget

    def _build_timeline_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setSpacing(16)

        color_group = QtWidgets.QGroupBox("Marker Colors")
        color_layout = QtWidgets.QFormLayout(color_group)
        self.timeline_color_buttons: Dict[str, QtWidgets.QPushButton] = {}
        for key in ["Golf Ball", "Club Handle", "Club Midpoint", "Club Toe", "Club Heel"]:
            btn = QtWidgets.QPushButton()
            btn.setFixedWidth(90)
            btn.clicked.connect(lambda _, k=key: self._pick_color(k))
            self.timeline_color_buttons[key] = btn
            color_layout.addRow(QtWidgets.QLabel(key), btn)
        layout.addWidget(color_group)

        layout.addLayout(self._spin_row("Timeline Min Width (%)", "", "timeline_min_width", 1, 100))
        layout.addLayout(self._spin_row("Timeline Max Width (%)", "", "timeline_max_width", 5, 100))

        self.timeline_show_keyframes = QtWidgets.QCheckBox("Show Keyframes")
        self.timeline_show_auto_keyframes = QtWidgets.QCheckBox("Show Auto Keyframes")
        self.timeline_show_issues = QtWidgets.QCheckBox("Show Issues")

        for checkbox in (
            self.timeline_show_keyframes,
            self.timeline_show_auto_keyframes,
            self.timeline_show_issues,
        ):
            layout.addWidget(checkbox)

        layout.addLayout(self._spin_row("Detailed Timeline Height (px)", "",
                                        "timeline_detailed_height", 20, 200))
        layout.addLayout(self._spin_row("Overview Timeline Height (px)", "",
                                        "timeline_overview_height", 20, 200))

        layout.addStretch(1)
        return widget

    def _build_playback_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setSpacing(16)

        layout.addLayout(self._combo_row("Default Playback Speed", "",
                                         "playback_speed", ["0.25x", "0.5x", "1x", "1.5x", "2x"]))
        self.playback_loop = QtWidgets.QCheckBox("Loop Playback")
        self.playback_auto_pause = QtWidgets.QCheckBox("Auto Pause On Issue")
        layout.addWidget(self.playback_loop)
        layout.addWidget(self.playback_auto_pause)
        layout.addLayout(self._spin_row("Frame Step Amount", "",
                                        "playback_frame_step", 1, 120))
        layout.addStretch(1)
        return widget

    def _build_input_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setSpacing(16)

        self.input_auto_track_click = QtWidgets.QCheckBox("Auto Track On Click")
        layout.addWidget(self.input_auto_track_click)

        layout.addLayout(self._slider_row("Pointer Snap Radius (pixels)", "", 0, 200,
                                          attr_name="input_snap_slider",
                                          value_label_name="input_snap_value"))
        layout.addLayout(self._slider_row("Gesture Sensitivity", "", 1, 200,
                                          attr_name="input_gesture_slider",
                                          value_label_name="input_gesture_value",
                                          formatter=lambda v: f"{v / 100:.2f}"))
        layout.addLayout(self._slider_row("Pan Speed", "", 1, 200,
                                          attr_name="input_pan_slider",
                                          value_label_name="input_pan_value",
                                          formatter=lambda v: f"{v / 100:.2f}"))
        layout.addLayout(self._slider_row("Scroll Inertia", "", 0, 100,
                                          attr_name="input_inertia_slider",
                                          value_label_name="input_inertia_value",
                                          formatter=lambda v: f"{v / 100:.2f}"))

        layout.addStretch(1)
        return widget

    def _build_notify_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setSpacing(16)

        self.notify_issue_notifications = QtWidgets.QCheckBox("Issue Notifications")
        self.notify_issue_sound = QtWidgets.QCheckBox("Issue Sound")
        layout.addWidget(self.notify_issue_notifications)
        layout.addWidget(self.notify_issue_sound)

        color_group = QtWidgets.QGroupBox("Issue Severity Colors")
        color_layout = QtWidgets.QFormLayout(color_group)
        self.notify_color_buttons: Dict[str, QtWidgets.QPushButton] = {}
        for level in ["Low", "Medium", "High"]:
            btn = QtWidgets.QPushButton()
            btn.setFixedWidth(90)
            btn.clicked.connect(lambda _, k=level: self._pick_notify_color(k))
            self.notify_color_buttons[level] = btn
            color_layout.addRow(QtWidgets.QLabel(level), btn)
        layout.addWidget(color_group)

        layout.addLayout(self._combo_row("Log Level", "", "notify_log_level",
                                         ["Debug", "Info", "Warn", "Error"]))
        self.notify_auto_save_keyframes = QtWidgets.QCheckBox("Auto Save Keyframes")
        layout.addWidget(self.notify_auto_save_keyframes)
        layout.addStretch(1)
        return widget

    def _build_data_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setSpacing(16)

        layout.addLayout(self._line_edit_row("Default Session Folder", "", "data_session_folder"))
        layout.addLayout(self._combo_row("Export Format", "", "data_export_format",
                                         ["JSON", "CSV", "YAML"]))
        layout.addLayout(self._spin_row("Auto Save Interval (seconds)", "",
                                        "data_auto_save_interval", 0, 3600))

        layout.addStretch(1)
        return widget

    def _wrap_with_scroll(self, content: QtWidgets.QWidget) -> QtWidgets.QWidget:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet("QScrollArea {background-color: transparent;}")
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(content)
        scroll.setWidget(container)
        return scroll

    def _main_window(self) -> Optional[QtWidgets.QWidget]:
        parent = self.parent()
        if isinstance(parent, QtWidgets.QWidget):
            return parent
        return None

    def _refresh_control_buttons(self) -> None:
        main_window = self._main_window()
        video_loaded = False
        auto_tracking_enabled = self.settings.general.auto_track
        active_point_available = False

        if main_window is not None:
            video_player = getattr(main_window, "video_player", None)
            video_loaded = bool(video_player and getattr(video_player, "is_loaded", lambda: False)())
            auto_tracking_enabled = bool(getattr(main_window, "auto_tracking_enabled", auto_tracking_enabled))
            active_point = getattr(main_window, "active_point", None)
            tracker = getattr(main_window, "custom_tracker", None)
            if tracker and active_point and active_point in getattr(tracker, "point_definitions", lambda: {})():
                active_point_available = True

        self.control_reset_button.setEnabled(video_loaded)

        if auto_tracking_enabled:
            self.control_auto_track_button.setText("Disable Auto Track")
        else:
            self.control_auto_track_button.setText("Enable Auto Track")

        self.general_auto_track.setChecked(auto_tracking_enabled)
        self.settings.general.auto_track = auto_tracking_enabled

        self.control_clear_history_button.setEnabled(active_point_available)
        if not active_point_available:
            self.control_clear_history_button.setToolTip("Select a point in the main window to clear its history.")
        else:
            active_point = getattr(main_window, "active_point", "")
            self.control_clear_history_button.setToolTip(f"Clear history for {active_point}.")

    def _handle_reset_session(self) -> None:
        main_window = self._main_window()
        if main_window and hasattr(main_window, "stop_playback"):
            main_window.stop_playback()
        self._refresh_control_buttons()

    def _handle_toggle_auto_track(self) -> None:
        main_window = self._main_window()
        current_state = self.general_auto_track.isChecked()
        if main_window and hasattr(main_window, "_set_auto_tracking_enabled"):
            new_state = not current_state
            main_window._set_auto_tracking_enabled(new_state)
            self.general_auto_track.setChecked(new_state)
            self.settings.general.auto_track = new_state
        else:
            # Fallback to toggling settings only
            new_state = not current_state
            self.general_auto_track.setChecked(new_state)
            self.settings.general.auto_track = new_state
        self._refresh_control_buttons()

    def _handle_clear_point_history(self) -> None:
        main_window = self._main_window()
        if main_window and hasattr(main_window, "_clear_selected_point_history"):
            main_window._clear_selected_point_history()
        self._refresh_control_buttons()

    # ------------------------------------------------------------------
    # Helpers for building rows
    # ------------------------------------------------------------------
    def _slider_row(
        self,
        label: str,
        description: str,
        minimum: int,
        maximum: int,
        attr_name: str,
        value_label_name: str,
        formatter=None,
    ) -> QtWidgets.QVBoxLayout:
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(minimum, maximum)
        value_label = QtWidgets.QLabel("0")
        value_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        slider_layout = QtWidgets.QHBoxLayout()
        slider_layout.addWidget(slider)
        slider_layout.addWidget(value_label)

        container = QtWidgets.QVBoxLayout()
        label_widget = QtWidgets.QLabel(label)
        container.addWidget(label_widget)
        if description:
            desc_widget = QtWidgets.QLabel(description)
            desc_widget.setStyleSheet("color: #909090; font-size: 11px;")
            container.addWidget(desc_widget)
        container.addLayout(slider_layout)

        setattr(self, attr_name, slider)
        setattr(self, value_label_name, value_label)
        def update_label(value: int) -> None:
            if formatter:
                value_label.setText(formatter(value))
            elif maximum == 100 and minimum == 0:
                value_label.setText(f"{value / 100:.2f}")
            else:
                value_label.setText(str(value))

        slider.valueChanged.connect(update_label)
        return container

    def _spin_row(
        self,
        label: str,
        description: str,
        attr_name: str,
        minimum: int,
        maximum: int,
    ) -> QtWidgets.QVBoxLayout:
        spin = QtWidgets.QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(1)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(label))
        row.addStretch(1)
        row.addWidget(spin)

        container = QtWidgets.QVBoxLayout()
        container.addLayout(row)
        if description:
            desc = QtWidgets.QLabel(description)
            desc.setStyleSheet("color: #909090; font-size: 11px;")
            container.addWidget(desc)
        setattr(self, attr_name, spin)
        return container

    def _combo_row(self, label: str, description: str, attr_name: str, options: list[str]) -> QtWidgets.QVBoxLayout:
        combo = QtWidgets.QComboBox()
        combo.addItems(options)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(label))
        row.addStretch(1)
        row.addWidget(combo)

        container = QtWidgets.QVBoxLayout()
        container.addLayout(row)
        if description:
            desc = QtWidgets.QLabel(description)
            desc.setStyleSheet("color: #909090; font-size: 11px;")
            container.addWidget(desc)
        setattr(self, attr_name, combo)
        return container

    def _line_edit_row(self, label: str, description: str, attr_name: str) -> QtWidgets.QVBoxLayout:
        line_edit = QtWidgets.QLineEdit()
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(label))
        row.addStretch(1)
        row.addWidget(line_edit)

        container = QtWidgets.QVBoxLayout()
        container.addLayout(row)
        if description:
            desc = QtWidgets.QLabel(description)
            desc.setStyleSheet("color: #909090; font-size: 11px;")
            container.addWidget(desc)
        setattr(self, attr_name, line_edit)
        return container

    # ------------------------------------------------------------------
    # Populate fields
    # ------------------------------------------------------------------
    def _populate_fields(self) -> None:
        g = self.settings.general
        self.general_auto_track.setChecked(g.auto_track)
        self.general_trail_slider.setValue(g.trail_length)
        self.general_trail_value.setText(str(g.trail_length))
        self.general_viewport_start.setValue(g.viewport_start)
        self.general_viewport_end.setValue(g.viewport_end)
        index = self.general_theme.findText(g.theme)
        self.general_theme.setCurrentIndex(max(0, index))

        t = self.settings.tracking
        self.tracking_enable_smoothing.setChecked(t.smoothing_enabled)
        self.tracking_truncate_future.setChecked(t.truncate_future_on_manual_set)
        self.tracking_smoothing_slider.setValue(int(t.smoothing_alpha * 100))
        self.tracking_smoothing_value.setText(f"{t.smoothing_alpha:.2f}")
        self.tracking_max_frames.setValue(t.max_auto_track_frames)
        self.tracking_direction_slider.setValue(int(t.direction_change_threshold))
        self.tracking_direction_value.setText(f"{int(t.direction_change_threshold)}°")
        self.tracking_deviation.setValue(int(t.deviation_threshold))
        self.tracking_cache_size.setValue(t.cache_size)
        self.tracking_history_frames.setValue(t.history_frames)
        self.tracking_flow_window.setValue(t.optical_flow_window_size)
        self.tracking_flow_pyramid.setValue(t.optical_flow_pyramid)
        self.tracking_flow_termination.setValue(t.optical_flow_termination)
        self.tracking_issue_slider.setValue(int(t.issue_confidence_threshold * 100))
        self.tracking_issue_value.setText(f"{t.issue_confidence_threshold:.2f}")
        self.tracking_resolved_slider.setValue(int(t.resolved_confidence_threshold * 100))
        self.tracking_resolved_value.setText(f"{t.resolved_confidence_threshold:.2f}")

        tl = self.settings.timeline
        for key, btn in self.timeline_color_buttons.items():
            color = tl.marker_colors.get(key, "#ffffff")
            btn.setStyleSheet(f"background-color: {color}; border-radius: 8px;")
            btn.setText(color)
        self.timeline_min_width.setValue(int(tl.min_zoom_width))
        self.timeline_max_width.setValue(int(tl.max_zoom_width))
        self.timeline_show_keyframes.setChecked(tl.show_keyframes)
        self.timeline_show_auto_keyframes.setChecked(tl.show_auto_keyframes)
        self.timeline_show_issues.setChecked(tl.show_issues)
        self.timeline_detailed_height.setValue(tl.detailed_height)
        self.timeline_overview_height.setValue(tl.overview_height)

        pb = self.settings.playback
        index = self.playback_speed.findText(pb.default_speed)
        self.playback_speed.setCurrentIndex(max(0, index))
        self.playback_loop.setChecked(pb.loop_playback)
        self.playback_auto_pause.setChecked(pb.auto_pause_on_issue)
        self.playback_frame_step.setValue(pb.frame_step_amount)

        ip = self.settings.input
        self.input_auto_track_click.setChecked(ip.auto_track_on_click)
        self.input_snap_slider.setValue(ip.pointer_snap_radius)
        self.input_snap_value.setText(str(ip.pointer_snap_radius))
        self.input_gesture_slider.setValue(int(ip.gesture_sensitivity * 100))
        self.input_gesture_value.setText(f"{ip.gesture_sensitivity:.2f}")
        self.input_pan_slider.setValue(int(ip.pan_speed * 100))
        self.input_pan_value.setText(f"{ip.pan_speed:.2f}")
        self.input_inertia_slider.setValue(int(ip.scroll_inertia * 100))
        self.input_inertia_value.setText(f"{ip.scroll_inertia:.2f}")

        nt = self.settings.notify
        self.notify_issue_notifications.setChecked(nt.issue_notifications)
        self.notify_issue_sound.setChecked(nt.issue_sound)
        for key, btn in self.notify_color_buttons.items():
            color = nt.issue_severity_colors.get(key, "#ffffff")
            btn.setStyleSheet(f"background-color: {color}; border-radius: 8px;")
            btn.setText(color)
        idx = self.notify_log_level.findText(nt.log_level)
        self.notify_log_level.setCurrentIndex(max(0, idx))
        self.notify_auto_save_keyframes.setChecked(nt.auto_save_keyframes)

        ds = self.settings.data
        self.data_session_folder.setText(ds.default_session_folder)
        idx = self.data_export_format.findText(ds.export_format)
        self.data_export_format.setCurrentIndex(max(0, idx))
        self.data_auto_save_interval.setValue(ds.auto_save_interval)

        self._refresh_control_buttons()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _reset_to_defaults(self) -> None:
        self.settings = AppSettings()
        self._populate_fields()

    def _apply_and_close(self) -> None:
        self._collect_values()
        self.settingsApplied.emit(self.settings)
        self.accept()

    def _collect_values(self) -> None:
        g = self.settings.general
        g.auto_track = self.general_auto_track.isChecked()
        g.trail_length = self.general_trail_slider.value()
        start = self.general_viewport_start.value()
        end = self.general_viewport_end.value()
        if end <= start:
            end = min(100.0, start + 5.0)
        g.viewport_start = start
        g.viewport_end = end
        g.theme = self.general_theme.currentText()

        t = self.settings.tracking
        t.smoothing_alpha = self.tracking_smoothing_slider.value() / 100.0
        t.max_auto_track_frames = self.tracking_max_frames.value()
        t.direction_change_threshold = self.tracking_direction_slider.value()
        t.deviation_threshold = self.tracking_deviation.value()
        t.cache_size = self.tracking_cache_size.value()
        t.history_frames = self.tracking_history_frames.value()
        t.optical_flow_window_size = self.tracking_flow_window.value()
        t.optical_flow_pyramid = self.tracking_flow_pyramid.value()
        t.optical_flow_termination = self.tracking_flow_termination.value()
        t.issue_confidence_threshold = self.tracking_issue_slider.value() / 100.0
        t.resolved_confidence_threshold = self.tracking_resolved_slider.value() / 100.0
        t.smoothing_enabled = self.tracking_enable_smoothing.isChecked()
        t.truncate_future_on_manual_set = self.tracking_truncate_future.isChecked()

        tl = self.settings.timeline
        for key, btn in self.timeline_color_buttons.items():
            tl.marker_colors[key] = btn.text()
        tl.min_zoom_width = self.timeline_min_width.value()
        tl.max_zoom_width = self.timeline_max_width.value()
        tl.show_keyframes = self.timeline_show_keyframes.isChecked()
        tl.show_auto_keyframes = self.timeline_show_auto_keyframes.isChecked()
        tl.show_issues = self.timeline_show_issues.isChecked()
        tl.detailed_height = self.timeline_detailed_height.value()
        tl.overview_height = self.timeline_overview_height.value()

        pb = self.settings.playback
        pb.default_speed = self.playback_speed.currentText()
        pb.loop_playback = self.playback_loop.isChecked()
        pb.auto_pause_on_issue = self.playback_auto_pause.isChecked()
        pb.frame_step_amount = self.playback_frame_step.value()

        ip = self.settings.input
        ip.auto_track_on_click = self.input_auto_track_click.isChecked()
        ip.pointer_snap_radius = self.input_snap_slider.value()
        ip.gesture_sensitivity = self.input_gesture_slider.value() / 100.0
        ip.pan_speed = self.input_pan_slider.value() / 100.0
        ip.scroll_inertia = self.input_inertia_slider.value() / 100.0

        nt = self.settings.notify
        nt.issue_notifications = self.notify_issue_notifications.isChecked()
        nt.issue_sound = self.notify_issue_sound.isChecked()
        for key, btn in self.notify_color_buttons.items():
            nt.issue_severity_colors[key] = btn.text()
        nt.log_level = self.notify_log_level.currentText()
        nt.auto_save_keyframes = self.notify_auto_save_keyframes.isChecked()

        ds = self.settings.data
        ds.default_session_folder = self.data_session_folder.text()
        ds.export_format = self.data_export_format.currentText()
        ds.auto_save_interval = self.data_auto_save_interval.value()

    def _pick_color(self, point_name: str) -> None:
        current = self.timeline_color_buttons[point_name].text()
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(current), self, f"Select color for {point_name}")
        if color.isValid():
            hex_color = color.name()
            btn = self.timeline_color_buttons[point_name]
            btn.setStyleSheet(f"background-color: {hex_color}; border-radius: 8px;")
            btn.setText(hex_color)

    def _pick_notify_color(self, level: str) -> None:
        current = self.notify_color_buttons[level].text()
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(current), self, f"Select color for {level}")
        if color.isValid():
            hex_color = color.name()
            btn = self.notify_color_buttons[level]
            btn.setStyleSheet(f"background-color: {hex_color}; border-radius: 8px;")
            btn.setText(hex_color)
