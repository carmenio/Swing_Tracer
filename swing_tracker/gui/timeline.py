from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

from PyQt5 import QtCore, QtGui, QtWidgets


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


class DetailedTimeline(QtWidgets.QWidget):
    seekRequested = QtCore.pyqtSignal(int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(36)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        self.total_frames: int = 0
        self.current_frame: int = 0
        self.viewport_range: Tuple[float, float] = (0.0, 100.0)
        self.markers: Dict[int, QtGui.QColor] = {}
        self.absence_ranges: List[Tuple[int, int]] = []

        self._dragging: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_total_frames(self, total: int) -> None:
        total = max(0, total)
        if self.total_frames != total:
            self.total_frames = total
            if self.current_frame >= total and total > 0:
                self.current_frame = total - 1
            self.update()

    def set_current_frame(self, frame_index: int) -> None:
        frame_index = max(0, frame_index)
        if self.current_frame != frame_index:
            self.current_frame = frame_index
            self.update()

    def set_viewport_range(self, start_percent: float, end_percent: float) -> None:
        start_percent = clamp(start_percent, 0.0, 100.0)
        end_percent = clamp(end_percent, start_percent + 0.001, 100.0)
        if self.viewport_range != (start_percent, end_percent):
            self.viewport_range = (start_percent, end_percent)
            self.update()

    def set_markers(self, markers: Iterable[Any]) -> None:
        new_markers: Dict[int, QtGui.QColor] = {}
        for marker in markers:
            if isinstance(marker, tuple) and len(marker) >= 2:
                frame = int(marker[0])
                color_value = marker[1]
                color = QtGui.QColor(color_value) if not isinstance(color_value, QtGui.QColor) else color_value
            else:
                frame = int(marker)  # type: ignore[arg-type]
                color = QtGui.QColor("#32ff8a")
            frame = max(0, frame)
            new_markers[frame] = color
        if self.markers != new_markers:
            self.markers = new_markers
            self.update()

    def set_absence_ranges(self, ranges: Iterable[Tuple[int, int]]) -> None:
        normalized: List[Tuple[int, int]] = []
        for start, end in ranges:
            start_i = max(0, int(start))
            end_i = max(start_i, int(end))
            normalized.append((start_i, end_i))
        normalized.sort()
        if self.absence_ranges != normalized:
            self.absence_ranges = normalized
            self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        rect = self.rect()
        painter.fillRect(rect, QtGui.QColor("#161616"))

        content = rect.adjusted(12, 10, -12, -10)
        painter.setBrush(QtGui.QColor("#2e2e2e"))
        painter.setPen(QtGui.QColor("#000000"))
        painter.drawRoundedRect(content, 8, 8)

        if self.total_frames <= 1:
            return

        start_frame, end_frame = self._viewport_frames()
        span = max(1.0, end_frame - start_frame)

        if self.absence_ranges:
            painter.save()
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(255, 140, 105, 90))
            for start, end in self.absence_ranges:
                if end < start_frame or start > end_frame:
                    continue
                clamped_start = max(start_frame, float(start))
                clamped_end = min(end_frame, float(end) + 1.0)
                if clamped_end <= clamped_start:
                    continue
                left = content.left() + ((clamped_start - start_frame) / span) * content.width()
                right = content.left() + ((clamped_end - start_frame) / span) * content.width()
                painter.drawRect(QtCore.QRectF(left, content.top(), max(1.0, right - left), content.height()))
            painter.restore()

        # Draw markers within viewport
        for marker_frame, color in self.markers.items():
            if marker_frame < start_frame or marker_frame > end_frame:
                continue
            percent = (marker_frame - start_frame) / span
            x = content.left() + percent * content.width()
            painter.setPen(QtGui.QPen(color, 2))
            painter.drawLine(QtCore.QPointF(x, content.top()), QtCore.QPointF(x, content.bottom()))

        # Draw playhead if inside viewport
        if start_frame <= self.current_frame <= end_frame:
            percent = (self.current_frame - start_frame) / span
            x = content.left() + percent * content.width()
            painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 2))
            painter.drawLine(QtCore.QPointF(x, content.top()), QtCore.QPointF(x, content.bottom()))

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self._dragging = True
            self._seek_at(event.pos())
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._dragging:
            self._seek_at(event.pos())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton and self._dragging:
            self._dragging = False
            self._seek_at(event.pos())
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        if not QtWidgets.QApplication.mouseButtons() & QtCore.Qt.LeftButton:
            self._dragging = False
        super().leaveEvent(event)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _viewport_frames(self) -> Tuple[float, float]:
        if self.total_frames <= 1:
            return (0.0, 1.0)
        total_span = float(self.total_frames - 1)
        start_percent, end_percent = self.viewport_range
        start_frame = (start_percent / 100.0) * total_span
        end_frame = (end_percent / 100.0) * total_span
        if end_frame <= start_frame:
            end_frame = start_frame + 1.0
        return start_frame, end_frame

    def _seek_at(self, pos: QtCore.QPoint) -> None:
        if self.total_frames <= 1:
            return
        rect = self.rect().adjusted(12, 10, -12, -10)
        if rect.width() <= 0:
            return
        start_frame, end_frame = self._viewport_frames()
        span = max(1.0, end_frame - start_frame)

        clamped_x = clamp(pos.x(), rect.left(), rect.right())
        percent = (clamped_x - rect.left()) / rect.width()
        target_frame = start_frame + percent * span
        target_frame = clamp(target_frame, 0.0, float(self.total_frames - 1))
        self.seekRequested.emit(int(round(target_frame)))


class OverviewTimeline(QtWidgets.QWidget):
    seekRequested = QtCore.pyqtSignal(int)
    viewportChanged = QtCore.pyqtSignal(float, float)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(46)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        try:
            self.grabGesture(QtCore.Qt.PinchGesture)
        except AttributeError:
            pass

        self.total_frames: int = 0
        self.current_frame: int = 0
        self.viewport_range: Tuple[float, float] = (0.0, 100.0)
        self.markers: Dict[int, QtGui.QColor] = {}
        self.absence_ranges: List[Tuple[int, int]] = []

        self._drag_mode: Optional[str] = None
        self._drag_start_x: float = 0.0
        self._drag_initial_range: Tuple[float, float] = (0.0, 100.0)
        self._min_width: float = 5.0
        self._max_width: float = 100.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_total_frames(self, total: int) -> None:
        total = max(0, total)
        if self.total_frames != total:
            self.total_frames = total
            if self.current_frame >= total and total > 0:
                self.current_frame = total - 1
            self.update()

    def set_current_frame(self, frame_index: int) -> None:
        frame_index = max(0, frame_index)
        if self.current_frame != frame_index:
            self.current_frame = frame_index
            self.update()

    def set_viewport_range(self, start_percent: float, end_percent: float) -> None:
        start_percent = clamp(start_percent, 0.0, 100.0)
        desired_width = end_percent - start_percent
        desired_width = clamp(desired_width, self._min_width, self._max_width)
        start_percent = clamp(start_percent, 0.0, 100.0 - desired_width)
        end_percent = start_percent + desired_width
        if self.viewport_range != (start_percent, end_percent):
            self.viewport_range = (start_percent, end_percent)
            self.update()

    def set_markers(self, markers: Iterable[Any]) -> None:
        new_markers: Dict[int, QtGui.QColor] = {}
        for marker in markers:
            if isinstance(marker, tuple) and len(marker) >= 2:
                frame = int(marker[0])
                color_value = marker[1]
                color = QtGui.QColor(color_value) if not isinstance(color_value, QtGui.QColor) else color_value
            else:
                frame = int(marker)  # type: ignore[arg-type]
                color = QtGui.QColor(50, 255, 138, 160)
            frame = max(0, frame)
            new_markers[frame] = color
        if self.markers != new_markers:
            self.markers = new_markers
            self.update()

    def set_absence_ranges(self, ranges: Iterable[Tuple[int, int]]) -> None:
        normalized: List[Tuple[int, int]] = []
        for start, end in ranges:
            start_i = max(0, int(start))
            end_i = max(start_i, int(end))
            normalized.append((start_i, end_i))
        normalized.sort()
        if self.absence_ranges != normalized:
            self.absence_ranges = normalized
            self.update()

    # ------------------------------------------------------------------
    # Event overrides (zoom support)
    # ------------------------------------------------------------------
    def event(self, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.NativeGesture:
            native = cast(QtGui.QNativeGestureEvent, event)
            if native.gestureType() == QtCore.Qt.ZoomNativeGesture:
                value = native.value()
                factor = 0.9 if value > 0 else 1.1
                anchor = self._percent_from_position(native.localPos())
                self._zoom_viewport(factor, anchor)
                return True
        elif event.type() == QtCore.QEvent.Gesture:
            gesture_event = cast(QtWidgets.QGestureEvent, event)
            pinch = gesture_event.gesture(QtCore.Qt.PinchGesture)
            if pinch and isinstance(pinch, QtWidgets.QPinchGesture):
                if pinch.changeFlags() & QtWidgets.QPinchGesture.ScaleFactorChanged:
                    scale = pinch.scaleFactor()
                    factor = 0.9 if scale > 1.0 else 1.1
                    center = pinch.centerPoint()
                    anchor = self._percent_from_position(center)
                    self._zoom_viewport(factor, anchor)
                gesture_event.accept(pinch)
                return True
        return super().event(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if event.modifiers() & QtCore.Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta != 0:
                factor = 0.9 if delta > 0 else 1.1
                anchor = self._percent_from_position(event.pos())
                self._zoom_viewport(factor, anchor)
            event.accept()
            return
        super().wheelEvent(event)

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        rect = self.rect()
        painter.fillRect(rect, QtGui.QColor("#141414"))

        content = rect.adjusted(12, 10, -12, -10)
        painter.setBrush(QtGui.QColor("#2a2a2a"))
        painter.setPen(QtGui.QColor("#000000"))
        painter.drawRoundedRect(content, 8, 8)

        if self.total_frames <= 1 or content.width() <= 0:
            return

        total_span = float(self.total_frames - 1)

        if self.absence_ranges:
            painter.save()
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(255, 140, 105, 70))
            for start, end in self.absence_ranges:
                if end < 0 or start > self.total_frames - 1:
                    continue
                clamped_start = max(0.0, float(start))
                clamped_end = min(float(self.total_frames - 1), float(end))
                if clamped_end < clamped_start:
                    continue
                left_ratio = (clamped_start / total_span) if total_span else 0.0
                right_ratio = ((clamped_end + 1.0) / total_span) if total_span else 1.0
                left = content.left() + left_ratio * content.width()
                right = content.left() + right_ratio * content.width()
                painter.drawRect(QtCore.QRectF(left, content.top(), max(1.0, right - left), content.height()))
            painter.restore()

        # Draw all markers with reduced opacity
        for marker, color in self.markers.items():
            percent = (marker / total_span) if total_span else 0.0
            x = content.left() + percent * content.width()
            painter.setPen(QtGui.QPen(color, 2))
            painter.drawLine(QtCore.QPointF(x, content.top()), QtCore.QPointF(x, content.bottom()))

        # Draw current playhead
        if self.current_frame <= total_span:
            play_percent = self.current_frame / total_span if total_span else 0.0
            play_x = content.left() + play_percent * content.width()
            painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 2))
            painter.drawLine(QtCore.QPointF(play_x, content.top()), QtCore.QPointF(play_x, content.bottom()))

        # Draw viewport window
        start_percent, end_percent = self.viewport_range
        view_left = content.left() + (start_percent / 100.0) * content.width()
        view_right = content.left() + (end_percent / 100.0) * content.width()
        viewport_rect = QtCore.QRectF(view_left, content.top(), view_right - view_left, content.height())

        painter.setBrush(QtGui.QColor(255, 255, 255, 35))
        painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1))
        painter.drawRoundedRect(viewport_rect, 6, 6)

        # Draw handles
        handle_width = 6
        painter.setBrush(QtGui.QColor("#ffffff"))
        painter.setPen(QtCore.Qt.NoPen)
        left_handle = QtCore.QRectF(view_left - handle_width / 2, content.top(), handle_width, content.height())
        right_handle = QtCore.QRectF(view_right - handle_width / 2, content.top(), handle_width, content.height())
        painter.drawRoundedRect(left_handle, 3, 3)
        painter.drawRoundedRect(right_handle, 3, 3)

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() != QtCore.Qt.LeftButton or self.total_frames <= 1:
            super().mousePressEvent(event)
            return

        content = self.rect().adjusted(12, 10, -12, -10)
        if content.width() <= 0:
            return

        x = clamp(event.pos().x(), content.left(), content.right())
        left, right = self._viewport_edges(content)
        handle_width = 12

        if left - handle_width <= x <= left + handle_width:
            self._drag_mode = "resize-left"
        elif right - handle_width <= x <= right + handle_width:
            self._drag_mode = "resize-right"
        elif left <= x <= right:
            self._drag_mode = "move"
        else:
            # seek to absolute timeline position
            percent = (x - content.left()) / content.width()
            target_frame = int(round(percent * (self.total_frames - 1)))
            self.seekRequested.emit(target_frame)
            self._drag_mode = None
            return

        self._drag_start_x = x
        self._drag_initial_range = self.viewport_range
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self._drag_mode:
            super().mouseMoveEvent(event)
            return

        content = self.rect().adjusted(12, 10, -12, -10)
        if content.width() <= 0:
            return

        x = clamp(event.pos().x(), content.left(), content.right())
        delta_px = x - self._drag_start_x
        delta_percent = (delta_px / content.width()) * 100.0

        start, end = self._drag_initial_range
        if self._drag_mode == "move":
            width = end - start
            new_start = clamp(start + delta_percent, 0.0, 100.0 - width)
            new_end = new_start + width
        elif self._drag_mode == "resize-left":
            new_start = clamp(start + delta_percent, 0.0, end - self._min_width)
            new_end = end
        elif self._drag_mode == "resize-right":
            new_end = clamp(end + delta_percent, start + self._min_width, 100.0)
            new_start = start
        else:
            return

        new_range = (new_start, new_end)
        if new_range != self.viewport_range:
            self.viewport_range = new_range
            self.viewportChanged.emit(*self.viewport_range)
            self.update()

        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton and self._drag_mode:
            self._drag_mode = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        if not QtWidgets.QApplication.mouseButtons() & QtCore.Qt.LeftButton:
            self._drag_mode = None
        super().leaveEvent(event)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _viewport_edges(self, content: QtCore.QRectF) -> Tuple[float, float]:
        start_percent, end_percent = self.viewport_range
        left = content.left() + (start_percent / 100.0) * content.width()
        right = content.left() + (end_percent / 100.0) * content.width()
        return left, right

    def _percent_from_position(self, pos: QtCore.QPointF) -> float:
        content = self.rect().adjusted(12, 10, -12, -10)
        if content.width() <= 0:
            return 50.0
        x = clamp(pos.x(), content.left(), content.right())
        return ((x - content.left()) / content.width()) * 100.0

    def _zoom_viewport(self, factor: float, anchor_percent: Optional[float] = None) -> None:
        start, end = self.viewport_range
        width = end - start
        width = clamp(width * factor, self._min_width, self._max_width)
        if anchor_percent is None:
            anchor_percent = (start + end) / 2
        anchor_percent = clamp(anchor_percent, 0.0, 100.0)
        start = clamp(anchor_percent - width / 2, 0.0, 100.0 - width)
        end = start + width
        if (start, end) != self.viewport_range:
            self.viewport_range = (start, end)
            self.viewportChanged.emit(start, end)
            self.update()

    def set_zoom_limits(self, min_width: float, max_width: float) -> None:
        self._min_width = max(1.0, min(100.0, min_width))
        self._max_width = max(self._min_width, min(100.0, max_width))
        start, end = self.viewport_range
        width = clamp(end - start, self._min_width, self._max_width)
        start = clamp(start, 0.0, 100.0 - width)
        end = start + width
        self.viewport_range = (start, end)
        self.viewportChanged.emit(start, end)
        self.update()
