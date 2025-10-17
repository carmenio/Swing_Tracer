import math
from bisect import bisect_left
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets

from ..model import AppSettings, Point2D, SettingsManager
from ..model.video import VideoPlayer
from ..model.video import VideoPlayer
from ..model.tracking.custom_points import CustomPointFrameResult
from .settings_dialog import SettingsDialog
from .video_widget import VideoContainer
from .timeline import DetailedTimeline, OverviewTimeline, TimelineMarker, clamp

if TYPE_CHECKING:  # pragma: no cover - for static analysis only
    from ..controller import SwingTrackerController


class FrameDecodeWorker(QtCore.QObject):
    frameReady = QtCore.pyqtSignal(int, object)
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)

    def __init__(self, player: VideoPlayer) -> None:
        super().__init__()
        self._player = player

    @QtCore.pyqtSlot(int)
    def advance(self, frames_to_advance: int) -> None:
        try:
            frame = self._player.advance(frames_to_advance)
            frame_index = self._player.current_frame_index
            self.frameReady.emit(frame_index, frame)
        except Exception as exc:  # pragma: no cover - defensive
            self.error.emit(str(exc))
        finally:
            self.finished.emit()


class FrameDecodeWorker(QtCore.QObject):
    frameReady = QtCore.pyqtSignal(int, object)
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)

    def __init__(self, player: VideoPlayer) -> None:
        super().__init__()
        self._player = player

    @QtCore.pyqtSlot(int)
    def advance(self, frames_to_advance: int) -> None:
        try:
            frame = self._player.advance(frames_to_advance)
            frame_index = self._player.current_frame_index
            self.frameReady.emit(frame_index, frame)
        except Exception as exc:  # pragma: no cover - defensive
            self.error.emit(str(exc))
        finally:
            self.finished.emit()


class OffFrameDialog(QtWidgets.QDialog):
    def __init__(
        self,
        point_name: str,
        total_frames: int,
        current_frame: int,
        existing_ranges: List[Tuple[int, int]],
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{point_name} Off-Frame Ranges")
        self.setModal(True)
        self.setMinimumWidth(360)
        self._total_frames = max(0, total_frames)
        self._ranges: List[Tuple[int, int]] = list(existing_ranges)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        description = QtWidgets.QLabel(
            "Define frame ranges where this point leaves the video. "
            "These ranges will be highlighted on the timeline and excluded from tracking."
        )
        description.setWordWrap(True)
        description.setStyleSheet("color: #b0b0b0; font-size: 11px;")
        layout.addWidget(description)

        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setStyleSheet(
            """
            QListWidget {
                background-color: #151515;
                border: 1px solid #242424;
                border-radius: 10px;
                padding: 6px;
            }
            QListWidget::item {
                color: #e0e0e0;
                padding: 6px;
            }
            QListWidget::item:selected {
                background-color: rgba(255, 255, 255, 0.12);
            }
            """
        )
        layout.addWidget(self.list_widget)

        controls = QtWidgets.QGroupBox("Add Range")
        controls_layout = QtWidgets.QGridLayout(controls)
        controls_layout.setVerticalSpacing(8)
        controls_layout.addWidget(QtWidgets.QLabel("Start Frame"), 0, 0)
        controls_layout.addWidget(QtWidgets.QLabel("End Frame"), 1, 0)

        max_frame = max(0, self._total_frames - 1)
        default_frame = min(max_frame, max(0, current_frame))
        self.start_spin = QtWidgets.QSpinBox()
        self.start_spin.setRange(0, max_frame if self._total_frames else 0)
        self.start_spin.setValue(default_frame)
        controls_layout.addWidget(self.start_spin, 0, 1)

        self.end_spin = QtWidgets.QSpinBox()
        self.end_spin.setRange(0, max_frame if self._total_frames else 0)
        self.end_spin.setValue(default_frame)
        controls_layout.addWidget(self.end_spin, 1, 1)

        self.add_button = QtWidgets.QPushButton("Add Range")
        self.add_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.add_button.setEnabled(self._total_frames > 0)
        self.add_button.clicked.connect(self._add_range)
        controls_layout.addWidget(self.add_button, 0, 2, 2, 1)
        layout.addWidget(controls)

        action_row = QtWidgets.QHBoxLayout()
        self.remove_button = QtWidgets.QPushButton("Remove Selected")
        self.remove_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.remove_button.clicked.connect(self._remove_selected)
        action_row.addWidget(self.remove_button)

        self.clear_button = QtWidgets.QPushButton("Clear All")
        self.clear_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.clear_button.clicked.connect(self._clear_all)
        action_row.addWidget(self.clear_button)
        action_row.addStretch(1)

        layout.addLayout(action_row)

        footer = QtWidgets.QHBoxLayout()
        footer.addStretch(1)
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        apply_btn = QtWidgets.QPushButton("Apply")
        apply_btn.setProperty("class", "primary")
        apply_btn.clicked.connect(self.accept)
        footer.addWidget(cancel_btn)
        footer.addWidget(apply_btn)
        layout.addLayout(footer)

        self._refresh_list()

    def _refresh_list(self) -> None:
        self.list_widget.clear()
        for start, end in self._merged_ranges():
            self.list_widget.addItem(f"Frames {start} – {end}")

    def _merged_ranges(self) -> List[Tuple[int, int]]:
        merged: List[Tuple[int, int]] = []
        for start, end in sorted(self._ranges):
            if start > end:
                start, end = end, start
            start = max(0, start)
            end = max(start, end)
            if not merged or start > merged[-1][1] + 1:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return merged

    def _add_range(self) -> None:
        if self._total_frames <= 0:
            return
        start = self.start_spin.value()
        end = self.end_spin.value()
        if start > end:
            start, end = end, start
        self._ranges.append((start, end))
        self._refresh_list()

    def _remove_selected(self) -> None:
        row = self.list_widget.currentRow()
        if row < 0:
            return
        merged = self._merged_ranges()
        if row >= len(merged):
            return
        target_start, target_end = merged[row]
        filtered: List[Tuple[int, int]] = []
        for start, end in self._ranges:
            rng_start = min(start, end)
            rng_end = max(start, end)
            if rng_end < target_start or rng_start > target_end:
                filtered.append((start, end))
        self._ranges = filtered
        self._refresh_list()

    def _clear_all(self) -> None:
        self._ranges.clear()
        self._refresh_list()

    def ranges(self) -> List[Tuple[int, int]]:
        return self._merged_ranges()


class SparklineWidget(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(36)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self._values: List[float] = []
        self._threshold: Optional[float] = None
        self._max_value: float = 0.0

    def set_samples(self, values: Sequence[float]) -> None:
        self._values = list(values)
        self._max_value = max(self._values) if self._values else 0.0
        self.update()

    def set_threshold(self, threshold: Optional[float]) -> None:
        self._threshold = threshold
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        rect = self.rect().adjusted(4, 4, -4, -4)
        painter.setPen(QtGui.QPen(QtGui.QColor(40, 40, 40), 1))
        painter.setBrush(QtGui.QColor(22, 22, 22))
        painter.drawRoundedRect(rect, 8, 8)

        if not self._values:
            painter.setPen(QtGui.QPen(QtGui.QColor(90, 90, 90, 120), 1, QtCore.Qt.DashLine))
            painter.drawLine(rect.bottomLeft(), rect.bottomRight())
            return

        max_value = self._max_value if self._max_value > 1e-6 else 1.0
        count = len(self._values)
        path = QtGui.QPainterPath()
        for idx, value in enumerate(self._values):
            x_ratio = idx / max(1, count - 1)
            x = rect.left() + x_ratio * rect.width()
            normalized = min(1.0, max(0.0, value / max_value))
            y = rect.bottom() - normalized * rect.height()
            if idx == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        painter.setPen(QtGui.QPen(QtGui.QColor(90, 200, 255, 220), 1.8))
        painter.drawPath(path)

        if self._threshold is not None and max_value > 0:
            threshold_ratio = min(1.0, max(0.0, self._threshold / max_value))
            threshold_y = rect.bottom() - threshold_ratio * rect.height()
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 120, 90, 180), 1, QtCore.Qt.DashLine))
            painter.drawLine(rect.left(), threshold_y, rect.right(), threshold_y)


class IssueRow(QtWidgets.QFrame):
    clicked = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setCursor(QtCore.Qt.PointingHandCursor)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class SwingTrackerWindow(QtWidgets.QMainWindow):
    def __init__(self, controller: "SwingTrackerController") -> None:
        super().__init__()
        self.controller = controller
        self.controller.set_view(self)

        self.setWindowTitle("Swing Tracker")
        self.resize(1440, 840)

        self.issues_collapsed: bool = False

        self._speed_actions: List[QtWidgets.QAction] = []
        self._build_ui()
        self._setup_connections()
        self.auto_track_task: Optional[dict] = None
        self.auto_track_timer = QtCore.QTimer(self)
        self.auto_track_timer.setInterval(0)
        self.auto_track_timer.timeout.connect(self._auto_track_step)

        self.playback_timer = QtCore.QTimer(self)
        self.playback_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.playback_timer.timeout.connect(self._next_frame)
        self.playback_speed: float = 1.0
        self._playback_clock = QtCore.QElapsedTimer()
        self._playback_accumulator: float = 0.0
        self._decode_in_flight: bool = False

        self._decoder_thread = QtCore.QThread(self)
        self._decoder_thread.setObjectName("FrameDecodeThread")
        self._decoder_worker = FrameDecodeWorker(self.video_player)
        self._decoder_worker.moveToThread(self._decoder_thread)
        self._decoder_worker.frameReady.connect(self._on_decode_frame_ready)
        self._decoder_worker.finished.connect(self._on_decode_finished)
        self._decoder_worker.error.connect(self._on_decode_error)
        self._decoder_thread.start()

        self._issue_items: List[Dict[str, Any]] = []
        self._active_issue_frame: Optional[int] = None

        self._apply_settings_to_ui()
        self._refresh_issue_panel()
        self._refresh_point_statuses()

    # ------------------------------------------------------------------
    # Model-backed properties
    # ------------------------------------------------------------------
    @property
    def settings_manager(self) -> SettingsManager:
        return self.controller.settings_manager

    @property
    def settings(self) -> AppSettings:
        return self.controller.settings

    @settings.setter
    def settings(self, value: AppSettings) -> None:
        self.controller.settings = value

    @property
    def auto_tracking_enabled(self) -> bool:
        return self.controller.auto_tracking_enabled

    @auto_tracking_enabled.setter
    def auto_tracking_enabled(self, value: bool) -> None:
        self.controller.auto_tracking_enabled = value

    @property
    def video_player(self):  # type: ignore[override]
        return self.controller.video_player

    @property
    def custom_tracker(self):  # type: ignore[override]
        return self.controller.custom_tracker

    @property
    def current_frame_bgr(self):  # type: ignore[override]
        return self.controller.current_frame_bgr

    @current_frame_bgr.setter
    def current_frame_bgr(self, value) -> None:  # type: ignore[override]
        self.controller.current_frame_bgr = value

    @property
    def preprocessed_custom_results(self):  # type: ignore[override]
        return self.controller.preprocessed_custom_results

    @preprocessed_custom_results.setter
    def preprocessed_custom_results(self, value):  # type: ignore[override]
        self.controller.preprocessed_custom_results = value

    @property
    def use_preprocessed_results(self) -> bool:
        return self.controller.use_preprocessed_results

    @use_preprocessed_results.setter
    def use_preprocessed_results(self, value: bool) -> None:
        self.controller.use_preprocessed_results = value

    @property
    def active_point(self) -> Optional[str]:
        return self.controller.active_point

    @active_point.setter
    def active_point(self, value: Optional[str]) -> None:
        self.controller.active_point = value

    @property
    def viewport_range(self) -> Tuple[float, float]:
        return self.controller.viewport_range

    @viewport_range.setter
    def viewport_range(self, value: Tuple[float, float]) -> None:
        self.controller.viewport_range = value

    @property
    def point_definitions(self) -> Dict[str, Tuple[int, int, int]]:
        return self.controller.point_definitions

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        central.setStyleSheet(
            """
            QWidget {
                background-color: #0b0b0b;
                color: #f0f0f0;
                font-size: 13px;
            }
            QLabel {
                color: #f0f0f0;
            }
            """
        )
        self.setCentralWidget(central)

        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(16)

        # Header row
        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setSpacing(12)
        self.headline_label = QtWidgets.QLabel("Track body movement and golf equipment through video analysis")
        self.headline_label.setStyleSheet("font-size: 16px; font-weight: 500; color: #dcdcdc;")
        header_layout.addWidget(self.headline_label)
        header_layout.addStretch(1)

        self.load_button = self._build_primary_button("Load Video", "background-color: #1f1f1f;")
        self.preprocess_button = self._build_primary_button("Preprocess", "background-color: #1f1f1f;")
        self.preprocess_button.setEnabled(False)
        self.settings_button = self._build_primary_button("Settings", "background-color: #1f1f1f;")
        header_layout.addWidget(self.load_button)
        header_layout.addWidget(self.preprocess_button)
        header_layout.addWidget(self.settings_button)
        main_layout.addLayout(header_layout)

        # Middle content area with resizable splitter
        self.video_container = VideoContainer()
        self.sidebar = self._build_sidebar()
        self.sidebar.setMinimumWidth(260)
        self.sidebar.setMaximumWidth(480)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(self.video_container)
        self.splitter.addWidget(self.sidebar)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)
        main_layout.addWidget(self.splitter, stretch=1)

        # Playback bar
        main_layout.addWidget(self._build_playback_bar())
        self.detailed_timeline.set_frame_map([])
        self.overview_timeline.set_frame_map([])
        self.detailed_timeline.set_viewport_range(*self.viewport_range)
        self.overview_timeline.set_viewport_range(*self.viewport_range)

    def _build_primary_button(self, text: str, base_style: str) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text)
        button.setCursor(QtCore.Qt.PointingHandCursor)
        button.setStyleSheet(
            f"""
            QPushButton {{
                {base_style}
                color: #f8f8f8;
                border-radius: 16px;
                padding: 8px 18px;
                font-weight: 500;
                border: 1px solid #2f2f2f;
            }}
            QPushButton:hover {{
                background-color: #2b2b2b;
            }}
            QPushButton:disabled {{
                color: #777777;
                background-color: #161616;
                border-color: #202020;
            }}
            """
        )
        return button

    def _build_sidebar(self) -> QtWidgets.QFrame:
        sidebar = QtWidgets.QFrame()
        sidebar.setStyleSheet(
            """
            QFrame {
                background-color: transparent;
            }
            """
        )
        sidebar.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        layout = QtWidgets.QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.points_card = self._build_points_card()
        self.points_card.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        self.issues_card = self._build_issues_card()
        self.issues_card.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)

        layout.addWidget(self.points_card)
        layout.addWidget(self.issues_card)
        layout.setStretch(0, 0)
        layout.setStretch(1, 1)
        return sidebar

    def _build_points_card(self) -> QtWidgets.QFrame:
        card = QtWidgets.QFrame()
        card.setStyleSheet(
            """
            QFrame {
                background-color: #121212;
                border: 1px solid #1f1f1f;
                border-radius: 16px;
            }
            """
        )
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header = QtWidgets.QHBoxLayout()
        header_label = QtWidgets.QLabel("Points")
        header_label.setStyleSheet("font-size: 14px; font-weight: 600; color: #f4f4f4;")
        header.addWidget(header_label)
        header.addStretch(1)
        layout.addLayout(header)

        self.point_rows: Dict[str, Dict[str, QtWidgets.QWidget]] = {}
        for name, color in self.point_definitions.items():
            row_widget = QtWidgets.QFrame()
            row_widget.setObjectName("pointRow")
            row_widget.setStyleSheet(
                """
                QFrame#pointRow {
                    background-color: transparent;
                    border-radius: 12px;
                }
                QFrame#pointRow[active="true"] {
                    background-color: rgba(255, 255, 255, 0.06);
                }
                """
            )
            row_widget.setProperty("active", False)

            row_layout = QtWidgets.QHBoxLayout(row_widget)
            row_layout.setContentsMargins(10, 6, 10, 6)
            row_layout.setSpacing(8)

            indicator = QtWidgets.QLabel()
            indicator.setFixedSize(10, 10)
            indicator.setStyleSheet(
                f"background-color: rgb({color[0]}, {color[1]}, {color[2]}); border-radius: 5px;"
            )
            row_layout.addWidget(indicator)

            name_label = QtWidgets.QLabel(name.replace("Club ", "Club "))
            name_label.setStyleSheet("font-size: 12px; color: #f0f0f0;")
            row_layout.addWidget(name_label, stretch=1)

            status_label = QtWidgets.QLabel("Not Set")
            status_label.setFixedWidth(56)
            status_label.setAlignment(QtCore.Qt.AlignCenter)
            status_label.setStyleSheet(
                """
                QLabel {
                    font-size: 11px;
                    border-radius: 10px;
                    padding: 2px 6px;
                    background-color: rgba(255, 255, 255, 0.08);
                    color: #b9b9b9;
                }
                """
            )
            row_layout.addWidget(status_label)

            set_button = QtWidgets.QPushButton("Set")
            set_button.setCursor(QtCore.Qt.PointingHandCursor)
            set_button.setFixedWidth(52)
            set_button.setStyleSheet(
                """
                QPushButton {
                    background-color: rgba(255, 255, 255, 0.08);
                    border: 1px solid #2f2f2f;
                    border-radius: 12px;
                    font-size: 11px;
                    padding: 3px 0;
                    color: #f0f0f0;
                }
                QPushButton:hover {
                    background-color: rgba(255, 255, 255, 0.18);
                }
                """
            )
            set_button.clicked.connect(lambda _, point=name: self._set_active_point(point))
            row_layout.addWidget(set_button)

            layout.addWidget(row_widget)
            self.point_rows[name] = {
                "row": row_widget,
                "status": status_label,
                "set_button": set_button,
            }

        self.mark_stop_button = QtWidgets.QPushButton("Mark Off-Frame")
        self.mark_stop_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.mark_stop_button.setStyleSheet("")
        self.mark_stop_button.setEnabled(False)
        layout.addWidget(self.mark_stop_button)

        return card

    def _build_issues_card(self) -> QtWidgets.QFrame:
        card = QtWidgets.QFrame()
        card.setStyleSheet(
            """
            QFrame {
                background-color: #121212;
                border: 1px solid #1f1f1f;
                border-radius: 16px;
            }
            """
        )
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header_layout = QtWidgets.QHBoxLayout()
        title_label = QtWidgets.QLabel("Key Point Tracker")
        title_label.setStyleSheet("font-size: 14px; font-weight: 600; color: #f4f4f4;")
        header_layout.addWidget(title_label)

        self.issue_count_badge = QtWidgets.QLabel("0")
        self.issue_count_badge.setFixedSize(22, 22)
        self.issue_count_badge.setAlignment(QtCore.Qt.AlignCenter)
        self.issue_count_badge.setStyleSheet(
            """
            QLabel {
                background-color: #2a2a2a;
                border-radius: 11px;
                font-size: 11px;
                color: #f0f0f0;
            }
            """
        )
        header_layout.addWidget(self.issue_count_badge)
        header_layout.addStretch(1)

        self.issue_toggle_button = QtWidgets.QToolButton()
        self.issue_toggle_button.setText("▴")
        self.issue_toggle_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.issue_toggle_button.setStyleSheet(
            """
            QToolButton {
                border: none;
                color: #f0f0f0;
                font-size: 14px;
            }
            QToolButton:hover {
                color: #ffffff;
            }
            """
        )
        header_layout.addWidget(self.issue_toggle_button)
        layout.addLayout(header_layout)

        self.issue_scroll = QtWidgets.QScrollArea()
        self.issue_scroll.setWidgetResizable(True)
        self.issue_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.issue_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.issue_scroll.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.issue_scroll.setStyleSheet(
            """
            QScrollArea { border: none; }
            QScrollBar:vertical { background: transparent; width: 8px; }
            QScrollBar::handle:vertical { background: rgba(255,255,255,0.2); border-radius: 4px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            """
        )

        self.issue_list_container = QtWidgets.QWidget()
        self.issue_list_layout = QtWidgets.QVBoxLayout(self.issue_list_container)
        self.issue_list_layout.setContentsMargins(0, 0, 0, 0)
        self.issue_list_layout.setSpacing(8)
        self.issue_scroll.setWidget(self.issue_list_container)
        layout.addWidget(self.issue_scroll)

        return card

    def _build_playback_bar(self) -> QtWidgets.QFrame:
        bar = QtWidgets.QFrame()
        bar.setStyleSheet(
            """
            QFrame {
                background-color: #121212;
                border: 1px solid #1f1f1f;
                border-radius: 18px;
            }
            """
        )

        outer_layout = QtWidgets.QVBoxLayout(bar)
        outer_layout.setContentsMargins(16, 12, 16, 12)
        outer_layout.setSpacing(10)

        # Top row (detailed timeline)
        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(12)

        self.play_toggle = QtWidgets.QPushButton("▶")
        self.play_toggle.setFixedSize(36, 36)
        self.play_toggle.setCursor(QtCore.Qt.PointingHandCursor)
        self.play_toggle.setStyleSheet(
            """
            QPushButton {
                border-radius: 10px;
                background-color: #1f1f1f;
                color: #f0f0f0;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2a2a2a;
            }
            QPushButton:disabled {
                color: #777777;
                background-color: #161616;
            }
            """
        )
        top_row.addWidget(self.play_toggle)

        self.detailed_timeline = DetailedTimeline()
        top_row.addWidget(self.detailed_timeline, stretch=1)

        self.current_time_label = QtWidgets.QLabel("0:00")
        self.current_time_label.setFixedWidth(48)
        self.current_time_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.current_time_label.setStyleSheet("color: #bbbbbb; font-size: 12px;")
        top_row.addWidget(self.current_time_label)

        outer_layout.addLayout(top_row)

        # Bottom row (overview timeline)
        bottom_row = QtWidgets.QHBoxLayout()
        bottom_row.setSpacing(12)

        self.playback_speed_button = QtWidgets.QToolButton()
        self.playback_speed_button.setText("1x")
        self.playback_speed_button.setFixedSize(60, 30)
        self.playback_speed_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.playback_speed_button = QtWidgets.QToolButton()
        self.playback_speed_button.setText("1x")
        self.playback_speed_button.setFixedSize(60, 30)
        self.playback_speed_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.playback_speed_button.setStyleSheet(
            """
            QToolButton {
            QToolButton {
                background-color: #1f1f1f;
                border: 1px solid #2f2f2f;
                border-radius: 10px;
                color: #f0f0f0;
                font-size: 12px;
                padding: 0 12px;
                padding: 0 12px;
            }
            QToolButton:hover {
            QToolButton:hover {
                background-color: #242424;
            }
            QToolButton::menu-indicator {
                image: none;
            }
            QToolButton::menu-indicator {
                image: none;
            }
            """
        )
        self.playback_speed_menu = QtWidgets.QMenu(self.playback_speed_button)
        self.playback_speed_button.setMenu(self.playback_speed_menu)
        self._populate_speed_menu()
        self.playback_speed_menu = QtWidgets.QMenu(self.playback_speed_button)
        self.playback_speed_button.setMenu(self.playback_speed_menu)
        self._populate_speed_menu()
        bottom_row.addWidget(self.playback_speed_button)

        self.overview_timeline = OverviewTimeline()
        bottom_row.addWidget(self.overview_timeline, stretch=1)

        self.total_time_label = QtWidgets.QLabel("0:00")
        self.total_time_label.setFixedWidth(48)
        self.total_time_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.total_time_label.setStyleSheet("color: #bbbbbb; font-size: 12px;")
        bottom_row.addWidget(self.total_time_label)

        outer_layout.addLayout(bottom_row)

        return bar

    # ------------------------------------------------------------------
    # Connections & helpers
    # ------------------------------------------------------------------
    def _populate_speed_menu(self) -> None:
        self.playback_speed_menu.clear()
        self._speed_actions.clear()
        for preset in [0.25, 0.5, 1.0, 2.0, 4.0]:
            label = self._format_speed_text(preset)
            action = self.playback_speed_menu.addAction(label)
            action.setData(preset)
            action.setCheckable(True)
            action.triggered.connect(lambda checked, value=preset: self._set_playback_speed(value))
            self._speed_actions.append(action)
        self.playback_speed_menu.addSeparator()
        custom_action = self.playback_speed_menu.addAction("Custom…")
        custom_action.triggered.connect(self._prompt_custom_speed)

    def _populate_speed_menu(self) -> None:
        self.playback_speed_menu.clear()
        self._speed_actions.clear()
        for preset in [0.25, 0.5, 1.0, 2.0, 4.0]:
            label = self._format_speed_text(preset)
            action = self.playback_speed_menu.addAction(label)
            action.setData(preset)
            action.setCheckable(True)
            action.triggered.connect(lambda checked, value=preset: self._set_playback_speed(value))
            self._speed_actions.append(action)
        self.playback_speed_menu.addSeparator()
        custom_action = self.playback_speed_menu.addAction("Custom…")
        custom_action.triggered.connect(self._prompt_custom_speed)

    def _setup_connections(self) -> None:
        self.load_button.clicked.connect(self.load_video)
        self.preprocess_button.clicked.connect(self._preprocess_video)

        self.play_toggle.clicked.connect(self.toggle_playback)

        self.video_container.request_load.connect(self.load_video)
        self.video_container.canvas.clicked.connect(self._handle_video_click)

        self.detailed_timeline.seekRequested.connect(self.seek_to_frame)
        self.overview_timeline.seekRequested.connect(self.seek_to_frame)
        self.overview_timeline.viewportChanged.connect(self._on_viewport_changed)

        self.mark_stop_button.clicked.connect(self._mark_stop_frame)

        self.issue_toggle_button.clicked.connect(self._toggle_issue_collapse)
        self.settings_button.clicked.connect(self._open_settings_dialog)

    def _format_speed_text(self, speed: float) -> str:
        return f"{speed:g}x"

    def _parse_speed_string(self, value: str) -> float:
        try:
            cleaned = value.strip().lower().replace("x", "")
            return float(cleaned) if cleaned else 1.0
        except ValueError:
            return 1.0

    def _set_playback_speed(self, speed: float, persist: bool = True) -> None:
        clamped = max(0.05, min(16.0, float(speed)))
        self.playback_speed = clamped
        label = self._format_speed_text(clamped)
        self.playback_speed_button.setText(label)
        self._update_speed_menu_checks()
        if persist:
            self.settings.playback.default_speed = label
            self.settings_manager.save()
        if self.playback_timer.isActive():
            self._reset_playback_clock()
            self._playback_clock.start()

    def _update_speed_menu_checks(self) -> None:
        for action in self._speed_actions:
            try:
                action_value = float(action.data())
            except (TypeError, ValueError):
                action.setChecked(False)
                continue
            action.setChecked(abs(action_value - self.playback_speed) < 1e-3)

    def _prompt_custom_speed(self) -> None:
        speed, ok = QtWidgets.QInputDialog.getDouble(
            self,
            "Playback Speed",
            "Speed multiplier:",
            self.playback_speed,
            0.05,
            32.0,
            2,
        )
        if ok:
            self._set_playback_speed(speed)

    def _reset_playback_clock(self) -> None:
        self._playback_clock = QtCore.QElapsedTimer()
        self._playback_accumulator = 0.0

    def _queue_frame_advance(self, frames_to_advance: int) -> None:
        if self._decode_in_flight:
            return
        frames = max(1, min(int(frames_to_advance), 16))
        self._decode_in_flight = True
        QtCore.QMetaObject.invokeMethod(
            self._decoder_worker,
            "advance",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(int, frames),
        )

    # ------------------------------------------------------------------
    # Video workflow
    # ------------------------------------------------------------------
    def load_video(self) -> None:
        dialog = QtWidgets.QFileDialog(self, "Select Swing Video")
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        dialog.setNameFilters(["Videos (*.mp4 *.mov *.avi *.mkv)", "All Files (*)"])
        if not dialog.exec_():
            return

        file_path = dialog.selectedFiles()[0]

        if self.playback_timer.isActive():
            self.playback_timer.stop()
        self._update_play_button(False)

        try:
            metadata = self.video_player.load(file_path)
        except ValueError:
            QtWidgets.QMessageBox.critical(self, "Error", "Failed to open video.")
            return

        self._invalidate_preprocessing()

        self.viewport_range = (
            self.settings.general.viewport_start,
            self.settings.general.viewport_end,
        )
        duration_seconds = metadata.frame_count / metadata.fps if metadata.fps else 0
        self.total_time_label.setText(self._format_timestamp(duration_seconds))
        self.current_time_label.setText("0:00")

        self.detailed_timeline.set_frame_map([])
        self.overview_timeline.set_frame_map([])
        self.overview_timeline.set_viewport_range(*self.viewport_range)
        self.detailed_timeline.set_viewport_range(*self.viewport_range)
        self.detailed_timeline.set_current_frame(0)
        self.overview_timeline.set_current_frame(0)
        self.detailed_timeline.set_markers([])
        self.overview_timeline.set_markers([])

        self.preprocess_button.setEnabled(True)
        self.play_toggle.setEnabled(True)
        self.mark_stop_button.setEnabled(True)

        self._initialize_points()
        self._refresh_issue_panel()

        first_frame = self.video_player.read_first_frame()
        if first_frame is not None:
            self._process_frame(first_frame, record=True)
            self.video_container.reset_view()
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Unable to read the first frame.")
            self.video_container.show_placeholder()
            self._update_timeline_position()

    def toggle_playback(self) -> None:
        if not self.video_player.is_loaded():
            return
        if self.playback_timer.isActive():
            self.playback_timer.stop()
            self._update_play_button(False)
            self._decode_in_flight = False
            self._reset_playback_clock()
            return

        if self.video_player.current_frame is None:
            first_frame = self.video_player.read_first_frame()
            if first_frame is None:
                return
            self._process_frame(first_frame, record=False)

        self._reset_playback_clock()
        self._playback_clock.start()
        self._decode_in_flight = False
        self.playback_timer.start(0)
        self._queue_frame_advance(1)
        self._playback_clock.restart()
        self._update_play_button(True)

    def stop_playback(self) -> None:
        if not self.video_player.is_loaded():
            return
        self._cancel_auto_track_task()
        self.playback_timer.stop()
        self._update_play_button(False)
        self._decode_in_flight = False
        self._reset_playback_clock()
        self._decode_in_flight = False
        self._reset_playback_clock()
        self.video_player.reset()
        self.custom_tracker.reset()
        self.current_frame_bgr = None
        self._refresh_issue_panel()
        self.current_time_label.setText("0:00")
        first_frame = self.video_player.read_first_frame()
        if first_frame is not None:
            self._process_frame(first_frame, record=True)
            self.video_container.reset_view()
        else:
            self.video_container.show_placeholder()
            self._update_timeline_position()

    def seek_to_frame(self, frame_index: int, resume_playback: bool = False) -> None:
        if not self.video_player.is_loaded():
            return

        was_playing = self.playback_timer.isActive()
        self._cancel_auto_track_task()

        frame = self.video_player.seek(frame_index)
        if frame is None:
            return

        self.playback_timer.stop()
        self._update_play_button(False)
        self._decode_in_flight = False
        self._decode_in_flight = False
        self._process_frame(frame, record=False)
        self._reset_playback_clock()

        if resume_playback and was_playing:
            self.toggle_playback()

    def _next_frame(self) -> None:
        if not self.video_player.is_loaded():
            self.playback_timer.stop()
            self._update_play_button(False)
            self._decode_in_flight = False
            return

        if self._decode_in_flight:
            return

        fps = self.video_player.metadata.fps or 30.0
        if fps <= 0:
            fps = 30.0
        frame_duration = 1000.0 / fps

        elapsed = self._playback_clock.restart()
        self._playback_accumulator += elapsed * self.playback_speed
        frames_ready = int(self._playback_accumulator / frame_duration)
        if frames_ready <= 0:
            return

        frames_to_request = max(1, min(frames_ready, 16))
        self._playback_accumulator -= frames_to_request * frame_duration
        self._queue_frame_advance(frames_to_request)

    @QtCore.pyqtSlot(int, object)
    def _on_decode_frame_ready(self, frame_index: int, frame_data: object) -> None:
        if frame_data is None:
            self.playback_timer.stop()
            self._update_play_button(False)
            self._decode_in_flight = False
            self._reset_playback_clock()
            return

        if not self.playback_timer.isActive():
            self._decode_in_flight = False
            return

        frame = frame_data if frame_data is not None else self.video_player.current_frame
        if frame is None:
            self.playback_timer.stop()
            self._update_play_button(False)
            self._decode_in_flight = False
            self._reset_playback_clock()
            return

        # VideoPlayer.advance already updates current_frame/current_frame_index.
        self._process_frame(frame, record=True)

    @QtCore.pyqtSlot()
    def _on_decode_finished(self) -> None:
        self._decode_in_flight = False

    @QtCore.pyqtSlot(str)
    def _on_decode_error(self, message: str) -> None:
        # Log decode errors without interrupting the UI flow.
        QtCore.qWarning(f"Frame decode error: {message}")
        self._decode_in_flight = False

    # ------------------------------------------------------------------
    # Preprocessing & tracking
    # ------------------------------------------------------------------
    def _preprocess_video(self) -> None:
        if not self.video_player.is_loaded():
            return

        self.playback_timer.stop()
        self._update_play_button(False)

        metadata = self.video_player.metadata
        if metadata.frame_count <= 0:
            QtWidgets.QMessageBox.information(self, "Info", "Video has no frames to preprocess.")
            return

        progress = QtWidgets.QProgressDialog("Preprocessing video...", "Cancel", 0, metadata.frame_count, self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        self._invalidate_preprocessing()
        self.custom_tracker.reset()
        self._refresh_issue_panel()

        custom_results: Dict[int, CustomPointFrameResult] = {}

        frame = self.video_player.read_first_frame()
        if frame is None:
            progress.close()
            QtWidgets.QMessageBox.warning(self, "Warning", "Unable to read the first frame.")
            return

        cancelled = False
        for _ in range(metadata.frame_count):
            QtWidgets.QApplication.processEvents()
            if progress.wasCanceled():
                cancelled = True
                break

            frame_index = self.video_player.current_frame_index
            custom_result = self.custom_tracker.process_frame(frame, frame_index, record=True)

            custom_results[frame_index] = custom_result
            self._apply_frame_results(custom_result)

            progress.setValue(frame_index + 1)
            frame = self.video_player.read_next()
            if frame is None:
                break

        progress.close()

        if cancelled:
            QtWidgets.QMessageBox.information(
                self, "Preprocess Cancelled", "Preprocessing cancelled. Live tracking will be used."
            )
            self._invalidate_preprocessing()
            first_frame = self.video_player.seek(0)
            if first_frame is not None:
                self._process_frame(first_frame, record=True)
            return

        self.preprocessed_custom_results = custom_results
        self.use_preprocessed_results = True

        first_frame = self.video_player.seek(0)
        if first_frame is not None:
            self._process_frame(first_frame, record=False)
            self.video_container.reset_view()

        QtWidgets.QMessageBox.information(self, "Preprocess Complete", "Tracking data cached for faster playback.")

    def _process_frame(self, frame_bgr, record: bool) -> None:
        frame_index = self.video_player.current_frame_index
        self.current_frame_bgr = frame_bgr.copy()

        if self.use_preprocessed_results and frame_index in self.preprocessed_custom_results:
            custom_result = self.preprocessed_custom_results.get(frame_index, CustomPointFrameResult({}, [], []))
            self.custom_tracker.current_positions = dict(custom_result.positions)
            self._apply_frame_results(custom_result, update_issues=False)
        else:
            custom_result = self.custom_tracker.process_frame(frame_bgr, frame_index, record=record)
            self._apply_frame_results(custom_result)

        self._render_current_frame()
        self._refresh_point_statuses()
        self._update_time_labels()
        self._update_timeline_position()

    def _apply_frame_results(
        self,
        custom_result: Optional[CustomPointFrameResult],
        update_issues: bool = True,
    ) -> None:
        if not update_issues or not custom_result:
            return

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def _render_current_frame(self) -> None:
        if self.current_frame_bgr is None:
            return

        bgr_frame = self.current_frame_bgr
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb_frame.shape
        image = QtGui.QImage(rgb_frame.data, width, height, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)

        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        self._draw_custom_traces(painter)
        self._draw_custom_points(painter)
        painter.end()

        self.video_container.set_frame(pixmap)
        self._update_timeline_position(adjust_viewport=False)

    def _draw_custom_traces(self, painter: QtGui.QPainter) -> None:
        current_index = self.video_player.current_frame_index if self.video_player.is_loaded() else None
        for tracked_point in self.custom_tracker.point_definitions().values():
            history = tracked_point.history()
            trail_length = self.trail_length
            if current_index is not None:
                min_frame = max(0, current_index - trail_length)
                history = [entry for entry in history if entry[0] >= min_frame]
            if len(history) < 2:
                continue
            stop_frames = {start for start, _ in tracked_point.absent_ranges}
            if tracked_point.open_absence_start is not None:
                stop_frames.add(tracked_point.open_absence_start)
            start_frames = {end + 1 for _, end in tracked_point.absent_ranges}
            for idx in range(1, len(history)):
                frame_idx = history[idx][0]
                prev_pos = history[idx - 1][1]
                curr_pos = history[idx][1]
                age = (current_index - frame_idx) if current_index is not None else 0
                ratio = max(0.0, min(1.0, (trail_length - age) / max(1, trail_length)))
                if ratio <= 0.0:
                    continue
                alpha = max(30, int(255 * ratio))
                highlight = (
                    frame_idx in stop_frames
                    or frame_idx in start_frames
                    or history[idx - 1][0] in stop_frames
                    or history[idx - 1][0] in start_frames
                )
                if highlight:
                    color = QtGui.QColor("#ff5c5c")
                else:
                    color = QtGui.QColor(*tracked_point.color)
                color.setAlpha(alpha)
                pen = QtGui.QPen(color, 2)
                painter.setPen(pen)
                painter.drawLine(QtCore.QPointF(*prev_pos), QtCore.QPointF(*curr_pos))

    def _draw_custom_points(self, painter: QtGui.QPainter) -> None:
        for name, position in self.custom_tracker.current_positions.items():
            tracked_point = self.custom_tracker.point_definitions().get(name)
            if not tracked_point:
                continue
            color = QtGui.QColor(*tracked_point.color)
            painter.setPen(QtGui.QPen(color, 2))
            painter.setBrush(QtGui.QBrush(color))
            painter.drawEllipse(QtCore.QPointF(*position), 5, 5)

    # ------------------------------------------------------------------
    # UI updates
    # ------------------------------------------------------------------
    def _refresh_point_statuses(self) -> None:
        frame_index = self.video_player.current_frame_index if self.video_player.is_loaded() else None
        for name, widgets in self.point_rows.items():
            tracked_point = self.custom_tracker.point_definitions().get(name)
            current_frame = frame_index
            absent = bool(
                tracked_point and current_frame is not None and tracked_point.is_absent(current_frame)
            )
            is_set = bool(
                tracked_point
                and current_frame is not None
                and current_frame in tracked_point.positions
                and not absent
            )
            status_label: QtWidgets.QLabel = widgets["status"]  # type: ignore[assignment]
            if absent:
                status_label.setText("Off Frame")
                status_label.setStyleSheet(
                    """
                    QLabel {
                        font-size: 11px;
                        border-radius: 10px;
                        padding: 2px 6px;
                        background-color: rgba(255, 140, 105, 0.3);
                        color: #ffd7c9;
                    }
                    """
                )
            elif is_set:
                status_label.setText("Set")
                status_label.setStyleSheet(
                    """
                    QLabel {
                        font-size: 11px;
                        border-radius: 10px;
                        padding: 2px 6px;
                        background-color: rgba(80, 200, 120, 0.35);
                        color: #c8f7da;
                    }
                    """
                )
            else:
                status_label.setText("Not Set")
                status_label.setStyleSheet(
                    """
                    QLabel {
                        font-size: 11px;
                        border-radius: 10px;
                        padding: 2px 6px;
                        background-color: rgba(255, 255, 255, 0.08);
                        color: #b9b9b9;
                    }
                    """
                )

            row: QtWidgets.QFrame = widgets["row"]  # type: ignore[assignment]
            is_active = name == self.active_point
            row.setProperty("active", is_active)
            row.style().unpolish(row)
            row.style().polish(row)

            button: QtWidgets.QPushButton = widgets["set_button"]  # type: ignore[assignment]
            button.setText("Active" if is_active else "Set")
            button.setStyleSheet(
                """
                QPushButton {
                    background-color: """
                + ("rgba(255, 255, 255, 0.2)" if is_active else "rgba(255, 255, 255, 0.08)")
                + """;
                    border: 1px solid #2f2f2f;
                    border-radius: 12px;
                    font-size: 11px;
                    padding: 3px 0;
                    color: #f0f0f0;
                }
                QPushButton:hover {
                    background-color: rgba(255, 255, 255, 0.28);
                }
                """
            )

        self._update_mark_stop_button_state()

    def _update_mark_stop_button_state(self) -> None:
        off_style = (
            """
            QPushButton {
                background-color: rgba(255, 90, 90, 0.16);
                border: 1px solid rgba(255, 90, 90, 0.55);
                border-radius: 12px;
                font-size: 11px;
                padding: 6px 10px;
                color: #ffb3b3;
            }
            QPushButton:hover {
                background-color: rgba(255, 90, 90, 0.26);
            }
            """
        )
        resume_style = (
            """
            QPushButton {
                background-color: rgba(80, 200, 120, 0.18);
                border: 1px solid rgba(80, 200, 120, 0.55);
                border-radius: 12px;
                font-size: 11px;
                padding: 6px 10px;
                color: #baf4d0;
            }
            QPushButton:hover {
                background-color: rgba(80, 200, 120, 0.28);
            }
            """
        )

        if not self.video_player.is_loaded() or self.active_point is None:
            self.mark_stop_button.setEnabled(False)
            self.mark_stop_button.setText("Mark Off-Frame")
            self.mark_stop_button.setStyleSheet(off_style)
            return

        tracked_point = self.custom_tracker.point_definitions().get(self.active_point)
        if not tracked_point:
            self.mark_stop_button.setEnabled(False)
            self.mark_stop_button.setText("Mark Off-Frame")
            self.mark_stop_button.setStyleSheet(off_style)
            return

        self.mark_stop_button.setEnabled(True)
        if tracked_point.open_absence_start is not None:
            self.mark_stop_button.setText("Mark Resume Frame")
            self.mark_stop_button.setStyleSheet(resume_style)
        else:
            self.mark_stop_button.setText("Mark Off-Frame")
            self.mark_stop_button.setStyleSheet(off_style)

    def _refresh_issue_panel(self) -> None:
        while self.issue_list_layout.count():
            item = self.issue_list_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        total_pending = 0
        if self.video_player.is_loaded():
            for point in self.point_definitions.keys():
                total_pending += self.custom_tracker.count_provisionals(point)
        self.issue_count_badge.setText(str(total_pending))

        self._issue_items = []

        if not self.video_player.is_loaded() or not self.active_point:
            placeholder = QtWidgets.QLabel("Load a video and select a point to review tracking suggestions.")
            placeholder.setWordWrap(True)
            placeholder.setAlignment(QtCore.Qt.AlignCenter)
            placeholder.setStyleSheet("color: #6f6f6f; font-size: 12px; padding: 16px 8px;")
            self.issue_list_layout.addWidget(placeholder)
            self.issue_list_layout.addStretch(1)
            self._active_issue_frame = None
            self.detailed_timeline.clear_highlight()
            self._update_issue_selection_styles()
            self._update_timeline_markers()
            return

        current_frame = self.video_player.current_frame_index
        chain = self.custom_tracker.provisional_chain_for_frame(self.active_point, current_frame)
        provisionals = chain.provisionals() if chain else []

        if not provisionals:
            placeholder = QtWidgets.QLabel("No provisional key points detected in this 50-frame span.")
            placeholder.setAlignment(QtCore.Qt.AlignCenter)
            placeholder.setStyleSheet("color: #6f6f6f; font-size: 12px; padding: 16px 8px;")
            self.issue_list_layout.addWidget(placeholder)
            self.issue_list_layout.addStretch(1)
            self._active_issue_frame = None
            self.detailed_timeline.clear_highlight()
            self._update_issue_selection_styles()
            self._update_timeline_markers()
            return

        header = QtWidgets.QFrame()
        header.setStyleSheet(
            """
            QFrame {
                background-color: rgba(255, 255, 255, 0.04);
                border-radius: 10px;
            }
            """
        )
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(10, 6, 10, 6)
        header_layout.setSpacing(8)

        span_label = QtWidgets.QLabel(f"Span F{chain.span_start} – F{chain.span_end}")
        span_label.setStyleSheet("color: #f0f0f0; font-size: 12px; font-weight: 600;")
        header_layout.addWidget(span_label)

        header_layout.addStretch(1)

        accept_all = QtWidgets.QPushButton("Accept All Shown")
        accept_all.setCursor(QtCore.Qt.PointingHandCursor)
        accept_all.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(80, 200, 120, 0.2);
                border: 1px solid rgba(80, 200, 120, 0.4);
                border-radius: 12px;
                font-size: 11px;
                padding: 4px 10px;
                color: #c8f7da;
            }
            QPushButton:hover {
                background-color: rgba(80, 200, 120, 0.3);
            }
            """
        )
        accept_all.clicked.connect(
            lambda _, p=self.active_point, span=(chain.span_start, chain.span_end): self._accept_all_provisionals(p, span)
        )
        header_layout.addWidget(accept_all)

        reject_all = QtWidgets.QPushButton("Reject All Shown")
        reject_all.setCursor(QtCore.Qt.PointingHandCursor)
        reject_all.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(220, 80, 80, 0.18);
                border: 1px solid rgba(220, 80, 80, 0.35);
                border-radius: 12px;
                font-size: 11px;
                padding: 4px 10px;
                color: #ffc7c7;
            }
            QPushButton:hover {
                background-color: rgba(220, 80, 80, 0.28);
            }
            """
        )
        reject_all.clicked.connect(
            lambda _, p=self.active_point, span=(chain.span_start, chain.span_end): self._reject_all_provisionals(p, span)
        )
        header_layout.addWidget(reject_all)

        self.issue_list_layout.addWidget(header)

        sorted_candidates = sorted(provisionals, key=lambda c: c.anchor.frame)
        for candidate in sorted_candidates:
            row, metadata = self._build_provisional_row(
                self.active_point,
                (chain.span_start, chain.span_end),
                chain,
                candidate,
            )
            self._issue_items.append(metadata)
            self.issue_list_layout.addWidget(row)

        frames_present = {item["frame"] for item in self._issue_items}
        if self._active_issue_frame is not None and self._active_issue_frame not in frames_present:
            self._active_issue_frame = None

        self.issue_list_layout.addStretch(1)
        self._update_issue_selection_styles()
        self._update_timeline_markers()

    def _build_provisional_row(
        self,
        point_name: str,
        span: Tuple[int, int],
        chain,
        candidate,
    ) -> Tuple[IssueRow, Dict[str, Any]]:
        row = IssueRow()
        row.setProperty("selected", False)
        row.setStyleSheet(
            """
            QFrame {
                background-color: rgba(255, 255, 255, 0.05);
                border-radius: 14px;
                border: 1px solid rgba(255, 255, 255, 0.06);
            }
            QFrame[selected="true"] {
                border: 1px solid rgba(80, 200, 120, 0.6);
                background-color: rgba(80, 200, 120, 0.18);
            }
            """
        )
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)

        base_color = self.point_definitions.get(point_name, (255, 255, 255))
        indicator = QtWidgets.QLabel()
        indicator.setFixedSize(12, 12)
        indicator.setStyleSheet(
            f"background-color: rgb({base_color[0]}, {base_color[1]}, {base_color[2]}); border-radius: 6px;"
        )
        layout.addWidget(indicator)

        info_layout = QtWidgets.QVBoxLayout()
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(2)

        frame_label = QtWidgets.QLabel(f"Frame {candidate.anchor.frame}")
        frame_label.setStyleSheet("color: #f0f0f0; font-size: 12px; font-weight: 600;")
        info_layout.addWidget(frame_label)

        residual = candidate.anchor.residual
        confidence_pct = candidate.anchor.confidence * 100.0
        fb_error = candidate.anchor.fb_error
        metrics_label = QtWidgets.QLabel(
            f"Residual {residual:.1f}px • Confidence {confidence_pct:.0f}% • FB {fb_error:.2f}"
        )
        metrics_label.setStyleSheet("color: #bcbcbc; font-size: 11px;")
        info_layout.addWidget(metrics_label)

        layout.addLayout(info_layout, stretch=1)

        sparkline = SparklineWidget()
        sparkline.setFixedWidth(160)
        tail_residuals = candidate.tail.residuals if candidate.tail else []
        sparkline.set_samples(tail_residuals)
        sparkline.set_threshold(chain.threshold_pixels)
        layout.addWidget(sparkline)

        button_column = QtWidgets.QVBoxLayout()
        button_column.setContentsMargins(0, 0, 0, 0)
        button_column.setSpacing(4)

        jump_button = QtWidgets.QPushButton("Jump")
        jump_button.setCursor(QtCore.Qt.PointingHandCursor)
        jump_button.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.18);
                border-radius: 10px;
                font-size: 11px;
                padding: 3px 10px;
                color: #f0f0f0;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.16);
            }
            """
        )
        row.clicked.connect(lambda f=candidate.anchor.frame: self._focus_issue(f))
        jump_button.clicked.connect(lambda _, f=candidate.anchor.frame: self._focus_issue(f))
        button_column.addWidget(jump_button)

        accept_button = QtWidgets.QPushButton("Accept")
        accept_button.setCursor(QtCore.Qt.PointingHandCursor)
        accept_button.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(80, 200, 120, 0.25);
                border: 1px solid rgba(80, 200, 120, 0.45);
                border-radius: 10px;
                font-size: 11px;
                padding: 3px 10px;
                color: #c8f7da;
            }
            QPushButton:hover {
                background-color: rgba(80, 200, 120, 0.35);
            }
            """
        )
        accept_button.clicked.connect(
            lambda _, p=point_name, sp=span, f=candidate.anchor.frame: self._accept_provisional_candidate(p, sp, f)
        )
        button_column.addWidget(accept_button)

        reject_button = QtWidgets.QPushButton("Reject")
        reject_button.setCursor(QtCore.Qt.PointingHandCursor)
        reject_button.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(220, 80, 80, 0.2);
                border: 1px solid rgba(220, 80, 80, 0.35);
                border-radius: 10px;
                font-size: 11px;
                padding: 3px 10px;
                color: #ffc7c7;
            }
            QPushButton:hover {
                background-color: rgba(220, 80, 80, 0.3);
            }
            """
        )
        reject_button.clicked.connect(
            lambda _, p=point_name, sp=span, f=candidate.anchor.frame: self._reject_provisional_candidate(p, sp, f)
        )
        button_column.addWidget(reject_button)

        layout.addLayout(button_column)

        metadata = {
            "row": row,
            "frame": candidate.anchor.frame,
            "span": span,
            "point": point_name,
        }
        return row, metadata

    def _accept_provisional_candidate(self, point_name: str, span: Tuple[int, int], frame_index: int) -> None:
        if self.custom_tracker.accept_provisional(point_name, span, frame_index):
            self._update_timeline_markers()
            self._refresh_issue_panel()
            self._advance_to_next_issue(frame_index)

    def _reject_provisional_candidate(self, point_name: str, span: Tuple[int, int], frame_index: int) -> None:
        if self.custom_tracker.reject_provisional(point_name, span, frame_index):
            self._update_timeline_markers()
            self._refresh_issue_panel()
            self._advance_to_next_issue(frame_index, pause_when_empty=True)

    def _accept_all_provisionals(self, point_name: str, span: Tuple[int, int]) -> None:
        if self.custom_tracker.accept_all_provisionals(point_name, span):
            self._update_timeline_markers()
            self._refresh_issue_panel()
            self._advance_to_next_issue(span[1])

    def _reject_all_provisionals(self, point_name: str, span: Tuple[int, int]) -> None:
        if self.custom_tracker.reject_all_provisionals(point_name, span):
            self._update_timeline_markers()
            self._refresh_issue_panel()
            self._advance_to_next_issue(span[1], pause_when_empty=True)

    def _focus_issue(self, frame_index: int) -> None:
        if not self.video_player.is_loaded():
            return
        was_playing = self.playback_timer.isActive()
        self._active_issue_frame = frame_index
        self.seek_to_frame(frame_index, resume_playback=was_playing)
        self._ensure_viewport_contains_frame(frame_index)
        self.detailed_timeline.pulse_highlight(frame_index)
        self._update_issue_selection_styles()

    def _advance_to_next_issue(self, previous_frame: int, pause_when_empty: bool = False) -> None:
        if not self._issue_items:
            self._handle_no_remaining_issues(pause_when_empty)
            return
        sorted_items = sorted(self._issue_items, key=lambda item: item["frame"])
        next_item = next((item for item in sorted_items if item["frame"] > previous_frame), None)
        if not next_item:
            self._handle_no_remaining_issues(pause_when_empty)
            return
        self._focus_issue(next_item["frame"])

    def _handle_no_remaining_issues(self, pause_playback: bool) -> None:
        self._active_issue_frame = None
        self._update_issue_selection_styles()
        self.detailed_timeline.clear_highlight()
        if pause_playback and self.playback_timer.isActive():
            self.playback_timer.stop()
            self._update_play_button(False)
            self._decode_in_flight = False
            self._reset_playback_clock()
        self._notify_no_remaining_issues()

    def _notify_no_remaining_issues(self) -> None:
        message = "No remaining key points to review"
        global_pos = self.mapToGlobal(self.rect().center())
        QtWidgets.QToolTip.showText(global_pos, message, self, self.rect(), 2000)

    def _update_issue_selection_styles(self) -> None:
        for item in self._issue_items:
            row = item.get("row")
            if not isinstance(row, QtWidgets.QFrame):
                continue
            selected = item.get("frame") == self._active_issue_frame
            row.setProperty("selected", bool(selected))
            row.style().unpolish(row)
            row.style().polish(row)

    def _toggle_issue_collapse(self) -> None:
        self.issues_collapsed = not self.issues_collapsed
        self.issue_list_container.setVisible(not self.issues_collapsed)
        self.issue_scroll.setVisible(not self.issues_collapsed)
        self.issue_toggle_button.setText("▾" if self.issues_collapsed else "▴")

    # ------------------------------------------------------------------
    # Interactions
    # ------------------------------------------------------------------
    def _set_active_point(self, point_name: str) -> None:
        if point_name not in self.point_definitions:
            return
        self.active_point = point_name
        self._refresh_point_statuses()
        self._update_timeline_markers()
        self._update_timeline_absences()
        self._update_timeline_position(adjust_viewport=False)

    def _auto_track_forward(self, point_name: str) -> None:
        if not self.video_player.is_loaded() or not self.auto_tracking_enabled:
            return
        tracked_point = self.custom_tracker.point_definitions().get(point_name)
        if not tracked_point:
            return

        current_frame = self.video_player.current_frame_index
        start_pos = tracked_point.positions.get(current_frame)
        if start_pos is None:
            return

        future_keyframes = [frame for frame in tracked_point.keyframe_frames() if frame > current_frame]
        if future_keyframes:
            return

        metadata = self.video_player.metadata
        if metadata.frame_count <= current_frame + 1:
            return

        frames_remaining = metadata.frame_count - 1 - current_frame
        max_frames = min(self.max_auto_track_frames, max(0, frames_remaining))
        if max_frames <= 0:
            return

        self._cancel_auto_track_task()

        if self.current_frame_bgr is not None:
            self.custom_tracker.prev_gray = cv2.cvtColor(self.current_frame_bgr, cv2.COLOR_BGR2GRAY)

        self.auto_track_task = {
            "point_name": point_name,
            "original_frame": current_frame,
            "frames_remaining": max_frames,
            "frames_processed": 0,
            "last_good_index": current_frame,
            "last_good_pos": start_pos,
            "samples": [(current_frame, start_pos)],
        }
        self.auto_track_timer.start(0)

    def _auto_track_step(self) -> None:
        task = self.auto_track_task
        if not task or not self.video_player.is_loaded() or not self.auto_tracking_enabled:
            self._complete_auto_track_task(finalize=False)
            return

        metadata = self.video_player.metadata
        if task["frames_processed"] >= task["frames_remaining"] or self.video_player.current_frame_index >= metadata.frame_count - 1:
            self._complete_auto_track_task()
            return

        frame = self.video_player.read_next()
        if frame is None:
            self._complete_auto_track_task()
            return

        frame_index = self.video_player.current_frame_index
        result = self.custom_tracker.process_frame(frame, frame_index, record=True)
        self._apply_frame_results(result)

        point_name = task["point_name"]
        point_issue = next((issue for issue in result.issues if issue.point_name == point_name), None)
        pos = result.positions.get(point_name)
        if point_issue or pos is None:
            self._complete_auto_track_task()
            return

        task["last_good_index"] = frame_index
        task["last_good_pos"] = pos
        task["samples"].append((frame_index, pos))
        task["frames_processed"] += 1

        if task["frames_processed"] >= task["frames_remaining"]:
            self._complete_auto_track_task()

    def _cancel_auto_track_task(self, finalize: bool = False) -> None:
        if not self.auto_track_task:
            return
        task = self.auto_track_task
        self.auto_track_timer.stop()
        self.auto_track_task = None
        if finalize:
            self._complete_auto_track_task(task, finalize=True)
        else:
            frame = self.video_player.seek(task["original_frame"])
            if frame is not None:
                self._process_frame(frame, record=False)
            self._update_timeline_position(adjust_viewport=False)

    def _complete_auto_track_task(self, task: Optional[dict] = None, finalize: bool = True) -> None:
        if task is None:
            task = self.auto_track_task
        if not task:
            self.auto_track_timer.stop()
            self.auto_track_task = None
            return

        self.auto_track_timer.stop()
        self.auto_track_task = None

        point_name = task["point_name"]
        original_frame = task["original_frame"]
        added_keyframe = False

        if finalize and task["last_good_pos"] is not None and task["last_good_index"] > original_frame:
            self.custom_tracker.add_auto_keyframe(point_name, task["last_good_index"])
            added_keyframe = True

        if finalize and len(task["samples"]) > 2:
            if self._mark_direction_changes(point_name, task["samples"]):
                added_keyframe = True

        frame = self.video_player.seek(original_frame)
        if frame is not None:
            self._process_frame(frame, record=False)

        if finalize and added_keyframe:
            self._refresh_issue_panel()
        self._update_timeline_position(adjust_viewport=False)

    def _mark_direction_changes(self, point_name: str, samples: List[Tuple[int, Point2D]]) -> bool:
        tracked_point = self.custom_tracker.point_definitions().get(point_name)
        if not tracked_point or len(samples) < 3:
            return False

        key_added = False
        prev_vec = None
        threshold_cos = math.cos(math.radians(self.direction_threshold))

        for idx in range(1, len(samples)):
            prev_frame, prev_pos = samples[idx - 1]
            frame, pos = samples[idx]
            vec_x = pos[0] - prev_pos[0]
            vec_y = pos[1] - prev_pos[1]
            mag = math.hypot(vec_x, vec_y)
            if mag < 1e-5:
                continue
            norm_vec = (vec_x / mag, vec_y / mag)
            if prev_vec is not None:
                dot = prev_vec[0] * norm_vec[0] + prev_vec[1] * norm_vec[1]
                dot = max(-1.0, min(1.0, dot))
                if dot < threshold_cos:
                    if frame not in tracked_point.keyframes:
                        self.custom_tracker.add_auto_keyframe(point_name, frame)
                        key_added = True
            prev_vec = norm_vec

        return key_added

    def _handle_video_click(self, x: float, y: float) -> None:
        if not self.video_player.is_loaded():
            return
        if self.active_point is None:
            self._set_active_point(next(iter(self.point_definitions)))
        if self.playback_timer.isActive():
            self.playback_timer.stop()
            self._update_play_button(False)

        point = self.active_point
        if point not in self.custom_tracker.point_definitions():
            return

        self._cancel_auto_track_task()
        self._invalidate_preprocessing()
        self.custom_tracker.set_manual_point(self.video_player.current_frame_index, point, (x, y))
        self._refresh_issue_panel()
        self._refresh_point_statuses()
        self._update_timeline_markers()
        self._update_timeline_absences()
        self._render_current_frame()
        self._auto_track_forward(point)
        self._update_timeline_position()

    def _clear_selected_point_history(self) -> None:
        if self.active_point is None:
            return
        if self.active_point not in self.custom_tracker.point_definitions():
            return
        self._cancel_auto_track_task()
        self._invalidate_preprocessing()
        self.custom_tracker.clear_point_history(self.active_point)
        self._refresh_issue_panel()
        self._refresh_point_statuses()
        self._update_timeline_markers()
        self._render_current_frame()
        self._update_timeline_position()

    def _mark_stop_frame(self) -> None:
        if not self.video_player.is_loaded():
            return
        if self.active_point is None:
            QtWidgets.QMessageBox.information(
                self,
                "No Point Selected",
                "Select a point before updating its off-frame status.",
            )
            return
        tracked_point = self.custom_tracker.point_definitions().get(self.active_point)
        if not tracked_point:
            return

        frame_index = self.video_player.current_frame_index
        pending_start = tracked_point.open_absence_start
        if pending_start is not None and frame_index <= pending_start:
            QtWidgets.QMessageBox.information(
                self,
                "Move Forward",
                "Move to a frame after the stop frame before marking the resume frame.",
            )
            return

        total_frames = self.video_player.metadata.frame_count if self.video_player.is_loaded() else frame_index + 1
        self._invalidate_preprocessing()
        if not self.custom_tracker.mark_stop_frame(self.active_point, frame_index, total_frames):
            QtWidgets.QMessageBox.information(
                self,
                "Unable to Update",
                "Select a valid frame before toggling the off-frame state.",
            )
            return

        self._refresh_point_statuses()
        self._update_timeline_markers()
        self._update_timeline_absences()
        if self.current_frame_bgr is not None:
            self._render_current_frame()

    def _jump_to_issue(self, frame_index: int) -> None:
        self._focus_issue(frame_index)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _initialize_points(self) -> None:
        self.custom_tracker.configure_points(self.point_definitions)
        if self.active_point not in self.point_definitions:
            self.active_point = next(iter(self.point_definitions))
        self._refresh_point_statuses()
        self._update_timeline_absences()
        self._update_timeline_markers()

    def _on_viewport_changed(self, start: float, end: float) -> None:
        min_width = 5.0
        width = max(min_width, end - start)
        start = clamp(start, 0.0, 100.0 - width)
        end = clamp(end, start + min_width, 100.0)
        self.viewport_range = (start, end)
        self.detailed_timeline.set_viewport_range(start, end)
        self.overview_timeline.set_viewport_range(start, end)
        self._update_timeline_position(adjust_viewport=False)

    def _ensure_viewport_contains_frame(self, frame_index: int) -> None:
        frames = self.detailed_timeline.frame_map
        if not frames or len(frames) <= 1:
            return

        total_span = float(len(frames) - 1)
        insert_index = bisect_left(frames, frame_index)
        if insert_index >= len(frames):
            insert_index = len(frames) - 1
        elif frames[insert_index] != frame_index and insert_index > 0:
            insert_index -= 1

        frame_percent = (insert_index / total_span) * 100.0
        start, end = self.viewport_range
        width = max(5.0, end - start)
        changed = False

        if frame_percent < start:
            start = clamp(frame_percent - width * 0.1, 0.0, 100.0 - width)
            end = start + width
            changed = True
        elif frame_percent > end:
            end = clamp(frame_percent + width * 0.1, width, 100.0)
            start = end - width
            start = clamp(start, 0.0, 100.0 - width)
            end = start + width
            changed = True

        if changed:
            self.viewport_range = (start, end)
            self.detailed_timeline.set_viewport_range(start, end)
            self.overview_timeline.set_viewport_range(start, end)

    def _apply_settings_to_ui(self) -> None:
        general = self.settings.general
        start = clamp(general.viewport_start, 0.0, 100.0)
        end = clamp(general.viewport_end, 0.0, 100.0)
        if end <= start:
            end = min(100.0, start + 5.0)
        self.viewport_range = (start, end)
        self.auto_tracking_enabled = general.auto_track
        self.detailed_timeline.set_viewport_range(start, end)
        self.overview_timeline.set_viewport_range(start, end)
        self.overview_timeline.set_zoom_limits(
            self.settings.timeline.min_zoom_width,
            self.settings.timeline.max_zoom_width,
        )
        detailed_height = max(24, self.settings.timeline.detailed_height)
        overview_height = max(24, self.settings.timeline.overview_height)
        self.detailed_timeline.setMinimumHeight(detailed_height)
        self.detailed_timeline.setMaximumHeight(detailed_height)
        self.overview_timeline.setMinimumHeight(overview_height)
        self.overview_timeline.setMaximumHeight(overview_height)
        self.custom_tracker.update_from_settings(self.settings.tracking)
        default_speed = self._parse_speed_string(self.settings.playback.default_speed)
        self._set_playback_speed(default_speed, persist=False)
        self._update_timeline_position(adjust_viewport=False)
        self._update_timeline_absences()

    def _open_settings_dialog(self) -> None:
        dialog = SettingsDialog(self.settings_manager, self)
        dialog.settingsApplied.connect(self._apply_new_settings)
        dialog.exec_()

    def _apply_new_settings(self, new_settings: AppSettings) -> None:
        self.settings_manager.settings = new_settings
        self.settings_manager.save()
        self.settings = self.settings_manager.settings
        self._apply_settings_to_ui()
        self._update_timeline_markers()
        self._refresh_issue_panel()

    def _timeline_frame_map(self) -> List[int]:
        if not self.video_player.is_loaded():
            return []
        frame_count = self.video_player.metadata.frame_count
        if frame_count <= 0:
            return []
        return list(range(frame_count))

    def _update_timeline_position(self, adjust_viewport: bool = True) -> None:
        frames = self._timeline_frame_map()
        self.detailed_timeline.set_frame_map(frames)
        self.overview_timeline.set_frame_map(frames)
        current_frame = self.video_player.current_frame_index if self.video_player.is_loaded() else 0

        if adjust_viewport and self.video_player.is_loaded():
            self._ensure_viewport_contains_frame(current_frame)

        self.detailed_timeline.set_viewport_range(*self.viewport_range)
        self.overview_timeline.set_viewport_range(*self.viewport_range)
        self.detailed_timeline.set_current_frame(current_frame)
        self.overview_timeline.set_current_frame(current_frame)

    def _update_timeline_markers(self) -> None:
        markers: List[TimelineMarker] = []
        if not self.video_player.is_loaded() or not self.active_point:
            self.detailed_timeline.set_markers(markers)
            self.overview_timeline.set_markers(markers)
            return

        tracked_point = self.custom_tracker.point_definitions().get(self.active_point)
        if not tracked_point:
            self.detailed_timeline.set_markers(markers)
            self.overview_timeline.set_markers(markers)
            return

        frame_count = self.video_player.metadata.frame_count
        max_frame = frame_count - 1 if frame_count > 0 else None

        def in_bounds(frame: int) -> bool:
            if frame < 0:
                return False
            if max_frame is None:
                return True
            return frame <= max_frame

        marker_records: Dict[int, Tuple[int, TimelineMarker]] = {}

        def register_marker(frame: int, color: QtGui.QColor, category: str, priority: int) -> None:
            if not in_bounds(frame):
                return
            normalized = TimelineMarker(int(frame), QtGui.QColor(color), category)
            existing = marker_records.get(normalized.frame)
            if existing and existing[0] > priority:
                return
            marker_records[normalized.frame] = (priority, normalized)

        base_color = QtGui.QColor(*tracked_point.color)
        base_color.setAlpha(235)
        auto_color = QtGui.QColor(255, 170, 60)
        auto_color.setAlpha(150)
        interpolated_color = QtGui.QColor(90, 210, 140)
        interpolated_color.setAlpha(130)
        absence_color = QtGui.QColor(255, 80, 80)
        absence_color.setAlpha(170)
        provisional_color = QtGui.QColor(base_color)
        provisional_color.setAlpha(190)

        manual_frames = set(tracked_point.accepted_keyframe_frames())
        for frame in manual_frames:
            register_marker(frame, base_color, "manual", 80)

        auto_frames = set(tracked_point.keyframe_frames()) - manual_frames
        for frame in auto_frames:
            register_marker(frame, auto_color, "auto", 60)

        for anchor in self.custom_tracker.provisional_markers(self.active_point):
            if anchor.frame in manual_frames:
                continue
            register_marker(anchor.frame, provisional_color, "provisional", 70)

        interpolated_frames = set(tracked_point.interpolation_cache.keys())
        interpolated_frames.difference_update(manual_frames)
        interpolated_frames.difference_update(auto_frames)
        for frame in interpolated_frames:
            register_marker(frame, interpolated_color, "interpolated", 40)

        for start, end in tracked_point.absent_ranges:
            register_marker(start, absence_color, "stop", 100)
            resume_frame = end + 1
            register_marker(resume_frame, absence_color, "start", 100)

        if tracked_point.open_absence_start is not None:
            register_marker(tracked_point.open_absence_start, absence_color, "stop", 100)

        markers = [entry[1] for entry in sorted(marker_records.values(), key=lambda item: item[1].frame)]
        self.detailed_timeline.set_markers(markers)
        self.overview_timeline.set_markers(markers)

    def _update_timeline_absences(self) -> None:
        ranges: List[Tuple[int, int]] = []
        if self.video_player.is_loaded() and self.active_point:
            tracked_point = self.custom_tracker.point_definitions().get(self.active_point)
            if tracked_point:
                ranges = list(tracked_point.absent_ranges)
                if tracked_point.open_absence_start is not None:
                    metadata = self.video_player.metadata
                    last_frame = metadata.frame_count - 1 if metadata.frame_count > 0 else self.video_player.current_frame_index
                    last_frame = max(tracked_point.open_absence_start, last_frame)
                    ranges.append((tracked_point.open_absence_start, last_frame))
                ranges.sort(key=lambda item: item[0])
        self.detailed_timeline.set_absence_ranges(ranges)
        self.overview_timeline.set_absence_ranges(ranges)

    @property
    def trail_length(self) -> int:
        return max(1, int(self.settings.general.trail_length))

    @property
    def max_auto_track_frames(self) -> int:
        return max(1, int(self.settings.tracking.max_auto_track_frames))

    @property
    def direction_threshold(self) -> float:
        return float(self.settings.tracking.direction_change_threshold)

    def _set_auto_tracking_enabled(self, enabled: bool) -> None:
        self.auto_tracking_enabled = enabled
        self.settings.general.auto_track = enabled
        if not enabled:
            self._cancel_auto_track_task()
        self.settings_manager.save()

    def _invalidate_preprocessing(self) -> None:
        self._cancel_auto_track_task()
        self.use_preprocessed_results = False
        self.preprocessed_custom_results.clear()

    def _update_play_button(self, playing: bool) -> None:
        self.play_toggle.setText("⏸" if playing else "▶")

    def _format_timestamp(self, seconds: float) -> str:
        total_seconds = max(0, int(round(seconds)))
        minutes = total_seconds // 60
        secs = total_seconds % 60
        return f"{minutes}:{secs:02d}"

    def _update_time_labels(self) -> None:
        if not self.video_player.is_loaded() or self.video_player.metadata.fps == 0:
            self.current_time_label.setText("0:00")
            return
        seconds = self.video_player.current_frame_index / self.video_player.metadata.fps
        self.current_time_label.setText(self._format_timestamp(seconds))


    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.playback_timer.stop()
        self._cancel_auto_track_task()
        self._decoder_thread.quit()
        self._decoder_thread.wait(1500)
        self.video_player.release()
        super().closeEvent(event)
