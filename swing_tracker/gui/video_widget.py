from typing import Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets


class VideoCanvas(QtWidgets.QWidget):
    clicked = QtCore.pyqtSignal(float, float)
    zoom_changed = QtCore.pyqtSignal(float)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent, True)
        self.setMouseTracking(True)
        self._pixmap: Optional[QtGui.QPixmap] = None
        self._zoom: float = 1.0
        self._min_zoom: float = 0.5
        self._max_zoom: float = 4.0
        self._fit_scale: float = 1.0
        self._offset = QtCore.QPointF(0.0, 0.0)
        self._dragging = False
        self._press_pos = QtCore.QPoint()
        self._last_pos = QtCore.QPoint()
        self.setCursor(QtCore.Qt.CrossCursor)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def set_frame(self, pixmap: QtGui.QPixmap) -> None:
        self._pixmap = pixmap
        self._update_fit_scale()
        self.update()
        self.zoom_changed.emit(self._zoom)

    def clear(self) -> None:
        self._pixmap = None
        self._fit_scale = 1.0
        self._offset = QtCore.QPointF(0.0, 0.0)
        self._zoom = 1.0
        self.update()
        self.zoom_changed.emit(self._zoom)

    def has_frame(self) -> bool:
        return self._pixmap is not None

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#000000"))
        if not self._pixmap:
            return

        scaled_size = self._scaled_size()
        scaled_pixmap = self._pixmap.scaled(
            scaled_size.toSize(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        left, top = self._content_origin(scaled_pixmap.size())
        painter.drawPixmap(QtCore.QPointF(left, top), scaled_pixmap)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_fit_scale()
        self._clamp_offset()

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self._press_pos = event.pos()
            self._last_pos = event.pos()
            self._dragging = False
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.buttons() & QtCore.Qt.LeftButton and self._pixmap:
            delta = event.pos() - self._last_pos
            if not self._dragging:
                distance = event.pos() - self._press_pos
                if self._zoom > 1.0 and (abs(distance.x()) > 3 or abs(distance.y()) > 3):
                    self._dragging = True
            if self._dragging:
                self._offset += QtCore.QPointF(delta)
                self._clamp_offset()
                self._last_pos = event.pos()
                self.update()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            if self._dragging:
                self.setCursor(QtCore.Qt.CrossCursor)
            else:
                coords = self._map_to_frame(event.pos())
                if coords:
                    self.clicked.emit(*coords)
            self._dragging = False
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        if not self._dragging:
            self.setCursor(QtCore.Qt.CrossCursor)
        super().leaveEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if not self._pixmap:
            return
        angle_delta = event.angleDelta().y()
        if angle_delta == 0:
            return
        factor = 1.02 if angle_delta > 0 else 0.98
        new_zoom = max(self._min_zoom, min(self._max_zoom, self._zoom * factor))
        self._set_zoom(new_zoom, anchor=event.pos())
        event.accept()

    # ------------------------------------------------------------------
    # Zoom / pan helpers
    # ------------------------------------------------------------------
    def zoom_in(self) -> None:
        self._set_zoom(min(self._max_zoom, self._zoom * 1.05))

    def zoom_out(self) -> None:
        self._set_zoom(max(self._min_zoom, self._zoom / 1.05))

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._offset = QtCore.QPointF(0.0, 0.0)
        self.update()
        self.zoom_changed.emit(self._zoom)
        self.setCursor(QtCore.Qt.CrossCursor)

    def _set_zoom(self, zoom: float, anchor: Optional[QtCore.QPoint] = None) -> None:
        zoom = max(self._min_zoom, min(self._max_zoom, zoom))
        if zoom == self._zoom or not self._pixmap:
            return

        old_zoom = self._zoom
        old_scaled = self._scaled_size()
        old_left, old_top = self._content_origin(old_scaled.toSize())

        self._zoom = zoom

        if anchor is not None:
            if old_scaled.width() > 0 and old_scaled.height() > 0:
                rel_x = (anchor.x() - old_left) / old_scaled.width()
                rel_y = (anchor.y() - old_top) / old_scaled.height()
                rel_x = max(0.0, min(1.0, rel_x))
                rel_y = max(0.0, min(1.0, rel_y))
                new_scaled = self._scaled_size()
                new_left_center = (self.width() - new_scaled.width()) / 2
                new_top_center = (self.height() - new_scaled.height()) / 2
                new_left = anchor.x() - rel_x * new_scaled.width()
                new_top = anchor.y() - rel_y * new_scaled.height()
                self._offset = QtCore.QPointF(new_left - new_left_center, new_top - new_top_center)
        self._clamp_offset()
        self.update()
        self.zoom_changed.emit(self._zoom)
        self.setCursor(QtCore.Qt.CrossCursor)

    def _scaled_size(self) -> QtCore.QSizeF:
        if not self._pixmap:
            return QtCore.QSizeF(0.0, 0.0)
        scale = self._fit_scale * self._zoom
        return QtCore.QSizeF(self._pixmap.width() * scale, self._pixmap.height() * scale)

    def _content_origin(self, scaled_size: QtCore.QSize) -> Tuple[float, float]:
        left = (self.width() - scaled_size.width()) / 2 + self._offset.x()
        top = (self.height() - scaled_size.height()) / 2 + self._offset.y()
        return left, top

    def _clamp_offset(self) -> None:
        if not self._pixmap:
            self._offset = QtCore.QPointF(0.0, 0.0)
            return
        scaled = self._scaled_size()
        max_x = max(0.0, (scaled.width() - self.width()) / 2)
        max_y = max(0.0, (scaled.height() - self.height()) / 2)
        clamped_x = max(-max_x, min(max_x, self._offset.x()))
        clamped_y = max(-max_y, min(max_y, self._offset.y()))
        self._offset = QtCore.QPointF(clamped_x, clamped_y)

    def _map_to_frame(self, pos: QtCore.QPoint) -> Optional[Tuple[float, float]]:
        if not self._pixmap:
            return None
        scaled_size = self._scaled_size()
        if scaled_size.width() == 0 or scaled_size.height() == 0:
            return None

        left, top = self._content_origin(scaled_size.toSize())
        x = pos.x() - left
        y = pos.y() - top
        if x < 0 or y < 0 or x > scaled_size.width() or y > scaled_size.height():
            return None
        scale = self._fit_scale * self._zoom
        return x / scale, y / scale

    def _update_fit_scale(self) -> None:
        if not self._pixmap or self.width() == 0 or self.height() == 0:
            return
        scale_w = self.width() / self._pixmap.width()
        scale_h = self.height() / self._pixmap.height()
        new_fit = min(scale_w, scale_h)
        if new_fit <= 0:
            new_fit = 1.0
        if abs(new_fit - self._fit_scale) > 1e-6:
            self._fit_scale = new_fit
            self._clamp_offset()
            self.update()
            self.zoom_changed.emit(self._zoom)


class ZoomControlPanel(QtWidgets.QFrame):
    zoom_in_requested = QtCore.pyqtSignal()
    zoom_out_requested = QtCore.pyqtSignal()
    reset_requested = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            """
            QFrame {
                background-color: rgba(20, 20, 20, 180);
                border: 1px solid #2f2f2f;
                border-radius: 10px;
            }
            QPushButton {
                color: #f0f0f0;
                background-color: transparent;
                border: none;
                font-size: 16px;
                padding: 4px 6px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.1);
            }
            QLabel {
                color: #f0f0f0;
                font-size: 12px;
                padding: 2px 0;
            }
            """
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        self.zoom_in_button = QtWidgets.QPushButton("+")
        self.zoom_out_button = QtWidgets.QPushButton("−")
        self.reset_button = QtWidgets.QPushButton("⟲")
        self.percent_label = QtWidgets.QLabel("100%")
        self.percent_label.setAlignment(QtCore.Qt.AlignCenter)

        layout.addWidget(self.zoom_in_button, alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(self.percent_label, alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(self.zoom_out_button, alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(self.reset_button, alignment=QtCore.Qt.AlignCenter)

        self.zoom_in_button.clicked.connect(self.zoom_in_requested.emit)
        self.zoom_out_button.clicked.connect(self.zoom_out_requested.emit)
        self.reset_button.clicked.connect(self.reset_requested.emit)

    def set_percentage(self, zoom: float) -> None:
        percentage = int(round(zoom * 100))
        self.percent_label.setText(f"{percentage}%")


class VideoPlaceholder(QtWidgets.QWidget):
    request_load = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            """
            background-color: #111111;
            border: 1px dashed #2f2f2f;
            border-radius: 16px;
            """
        )

        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setContentsMargins(24, 24, 24, 32)
        outer_layout.setSpacing(16)

        headline = QtWidgets.QLabel("Track body movement and golf equipment through video analysis")
        headline.setStyleSheet(
            """
            color: #d6d6d6;
            font-size: 16px;
            font-weight: 500;
            """
        )
        headline.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        outer_layout.addWidget(headline)

        divider = QtWidgets.QWidget()
        divider.setFixedHeight(1)
        divider.setStyleSheet("border-bottom: 1px dashed #2f2f2f;")
        outer_layout.addWidget(divider)

        center_container = QtWidgets.QWidget()
        center_layout = QtWidgets.QVBoxLayout(center_container)
        center_layout.setAlignment(QtCore.Qt.AlignCenter)
        center_layout.setSpacing(12)

        icon_label = QtWidgets.QLabel()
        icon_label.setAlignment(QtCore.Qt.AlignCenter)
        icon_label.setPixmap(self._build_icon_pixmap(72))
        center_layout.addWidget(icon_label)

        title = QtWidgets.QLabel("No video loaded")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("color: #f0f0f0; font-size: 18px; font-weight: 500;")
        center_layout.addWidget(title)

        subtitle = QtWidgets.QLabel("Upload a golf swing video to start tracking")
        subtitle.setAlignment(QtCore.Qt.AlignCenter)
        subtitle.setStyleSheet("color: #b0b0b0; font-size: 13px;")
        center_layout.addWidget(subtitle)

        load_button = QtWidgets.QPushButton("Choose Video File")
        load_button.setFixedWidth(180)
        load_button.setCursor(QtCore.Qt.PointingHandCursor)
        load_button.setStyleSheet(
            """
            QPushButton {
                color: #0f0f0f;
                background-color: #f4f4f4;
                border-radius: 18px;
                padding: 10px 24px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #ffffff;
            }
            """
        )
        load_button.clicked.connect(self.request_load)
        center_layout.addWidget(load_button, alignment=QtCore.Qt.AlignCenter)

        outer_layout.addStretch(1)
        outer_layout.addWidget(center_container, alignment=QtCore.Qt.AlignCenter)
        outer_layout.addStretch(3)

    def _build_icon_pixmap(self, size: int) -> QtGui.QPixmap:
        pixmap = QtGui.QPixmap(size, size)
        pixmap.fill(QtCore.Qt.transparent)

        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        pen = QtGui.QPen(QtGui.QColor("#9a9a9a"))
        pen.setWidth(4)
        painter.setPen(pen)

        margin = size * 0.22
        center_x = size / 2
        arrow_height = size * 0.28
        base_y = size * 0.55

        painter.drawLine(QtCore.QPointF(center_x, margin), QtCore.QPointF(center_x, base_y))
        painter.drawLine(
            QtCore.QPointF(center_x, margin),
            QtCore.QPointF(center_x - arrow_height * 0.4, margin + arrow_height * 0.4),
        )
        painter.drawLine(
            QtCore.QPointF(center_x, margin),
            QtCore.QPointF(center_x + arrow_height * 0.4, margin + arrow_height * 0.4),
        )

        tray_width = size * 0.5
        tray_height = size * 0.22
        tray_rect = QtCore.QRectF(
            center_x - tray_width / 2,
            base_y,
            tray_width,
            tray_height,
        )
        painter.drawRoundedRect(tray_rect, tray_height * 0.5, tray_height * 0.5)
        painter.end()
        return pixmap


class VideoContainer(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            """
            background-color: #101010;
            border: 1px solid #2f2f2f;
            border-radius: 16px;
            """
        )

        self.placeholder = VideoPlaceholder()
        self.canvas = VideoCanvas()
        self.canvas.setStyleSheet("background-color: #000000; border-radius: 12px;")

        self.zoom_controls = ZoomControlPanel()
        self.zoom_controls.hide()

        self.stack = QtWidgets.QStackedLayout()
        self.stack.setContentsMargins(8, 8, 8, 8)
        self.stack.addWidget(self.placeholder)

        self.video_view = QtWidgets.QWidget()
        video_layout = QtWidgets.QGridLayout(self.video_view)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.addWidget(self.canvas, 0, 0)
        video_layout.addWidget(self.zoom_controls, 0, 0, alignment=QtCore.Qt.AlignTop | QtCore.Qt.AlignRight)

        self.stack.addWidget(self.video_view)
        self.setLayout(self.stack)

        # Signal wiring
        self.placeholder.request_load.connect(self._relay_load_request)
        self.canvas.zoom_changed.connect(self._handle_zoom_changed)
        self.zoom_controls.zoom_in_requested.connect(self.canvas.zoom_in)
        self.zoom_controls.zoom_out_requested.connect(self.canvas.zoom_out)
        self.zoom_controls.reset_requested.connect(self.canvas.reset_view)

        self.show_placeholder()

    request_load = QtCore.pyqtSignal()

    def _relay_load_request(self) -> None:
        self.request_load.emit()

    def _handle_zoom_changed(self, zoom: float) -> None:
        self.zoom_controls.set_percentage(zoom)
        if zoom <= 1.0:
            self.canvas.setCursor(QtCore.Qt.CrossCursor)

    def show_placeholder(self) -> None:
        self.stack.setCurrentWidget(self.placeholder)
        self.zoom_controls.hide()
        self.canvas.clear()

    def show_video(self) -> None:
        self.stack.setCurrentWidget(self.video_view)
        self.zoom_controls.show()

    def set_frame(self, pixmap: QtGui.QPixmap) -> None:
        self.canvas.set_frame(pixmap)
        self.show_video()

    def reset_view(self) -> None:
        self.canvas.reset_view()
