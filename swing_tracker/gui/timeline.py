from bisect import bisect_left
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
        self.frame_map: List[int] = []
        self.frame_map: List[int] = []
        self.frame_map: List[int] = []
        self.frame_map: List[int] = []

        self._dragging: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_total_frames(self, total: int) -> None:
        total = max(0, total)
        if self.total_frames != total:
            self.total_frames = total
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

    def set_frame_map(self, frames: Iterable[int]) -> None:
        unique = sorted({max(0, int(frame)) for frame in frames})
        if self.frame_map != unique:
            self.frame_map = unique
            self.set_total_frames(len(unique))
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

        if len(self.frame_map) <= 1:
            return

        start_index, end_index = self._viewport_indices()
        span = max(1.0, end_index - start_index)

        if self.absence_ranges:
            painter.save()
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(255, 140, 105, 90))
            for start, end in self.absence_ranges:
                start_pos = self._position_for_frame(start)
                end_pos = self._position_for_frame(end + 1)
                if start_pos is None or end_pos is None:
                    continue
                if end_pos <= start_pos:
                    continue
                if end_pos < start_index or start_pos > end_index:
                    continue
                clamped_start = max(start_index, start_pos)
                clamped_end = min(end_index, end_pos)
                left = content.left() + ((clamped_start - start_index) / span) * content.width()
                right = content.left() + ((clamped_end - start_index) / span) * content.width()
                painter.drawRect(QtCore.QRectF(left, content.top(), max(1.0, right - left), content.height()))
            painter.restore()

        # Draw markers within viewport
        for marker_frame, color in self.markers.items():
            marker_pos = self._position_for_frame(marker_frame)
            if marker_pos is None or marker_pos < start_index or marker_pos > end_index:
                continue
            percent = (marker_pos - start_index) / span
            x = content.left() + percent * content.width()
            painter.setPen(QtGui.QPen(color, 2))
            painter.drawLine(QtCore.QPointF(x, content.top()), QtCore.QPointF(x, content.bottom()))

        # Draw playhead if inside viewport
        current_pos = self._position_for_frame(self.current_frame)
        if current_pos is not None and start_index <= current_pos <= end_index:
            percent = (current_pos - start_index) / span
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
    def _viewport_indices(self) -> Tuple[float, float]:
        if len(self.frame_map) <= 1:
            return (0.0, 1.0)
        total_span = float(len(self.frame_map) - 1)
        start_percent, end_percent = self.viewport_range
        start_index = (start_percent / 100.0) * total_span
        end_index = (end_percent / 100.0) * total_span
        if end_index <= start_index:
            end_index = start_index + 1.0
        return start_index, end_index

    def _position_for_frame(self, frame: float) -> Optional[float]:
        if not self.frame_map:
            return None
        target = float(frame)
        idx = bisect_left(self.frame_map, int(target))
        if idx < len(self.frame_map) and self.frame_map[idx] == int(target):
            return float(idx)
        if idx == 0 or idx >= len(self.frame_map):
            return None
        prev_frame = self.frame_map[idx - 1]
        next_frame = self.frame_map[idx]
        if next_frame == prev_frame:
            return float(idx)
        ratio = (target - prev_frame) / (next_frame - prev_frame)
        return (idx - 1) + max(0.0, min(1.0, ratio))

    def _frame_for_position(self, position: float) -> Optional[int]:
        if not self.frame_map:
            return None
        if len(self.frame_map) == 1:
            return self.frame_map[0]
        idx = int(round(position))
        idx = max(0, min(idx, len(self.frame_map) - 1))
        return self.frame_map[idx]

    def _seek_at(self, pos: QtCore.QPoint) -> None:
        if len(self.frame_map) <= 0:
            return
        rect = self.rect().adjusted(12, 10, -12, -10)
        if rect.width() <= 0:
            return
        start_index, end_index = self._viewport_indices()
        span = max(1.0, end_index - start_index)

        clamped_x = clamp(pos.x(), rect.left(), rect.right())
        percent = (clamped_x - rect.left()) / rect.width()
        target_position = start_index + percent * span
        target_position = clamp(target_position, 0.0, float(max(0, len(self.frame_map) - 1)))
        actual_frame = self._frame_for_position(target_position)
        if actual_frame is not None:
            self.seekRequested.emit(actual_frame)


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
        self.frame_map: List[int] = []

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

    def set_frame_map(self, frames: Iterable[int]) -> None:
        unique = sorted({max(0, int(frame)) for frame in frames})
        if self.frame_map != unique:
            self.frame_map = unique
            self.set_total_frames(len(unique))
            self.update()

    # ------------------------------------------------------------------
    # Event overrides (zoom support)
    # ------------------------------------------------------------------
    def event(self, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.NativeGesture:
            native = cast(QtGui.QNativeGestureEvent, event)
            if native.gestureType() == QtCore.Qt.ZoomNativeGesture:
                value = native.value()
                factor = 0.96 if value > 0 else 1.04
                anchor = self._percent_from_position(native.localPos())
                self._zoom_viewport(factor, anchor)
                return True
        elif event.type() == QtCore.QEvent.Gesture:
            gesture_event = cast(QtWidgets.QGestureEvent, event)
            pinch = gesture_event.gesture(QtCore.Qt.PinchGesture)
            if pinch and isinstance(pinch, QtWidgets.QPinchGesture):
                if pinch.changeFlags() & QtWidgets.QPinchGesture.ScaleFactorChanged:
                    scale = pinch.scaleFactor()
                    factor = 0.96 if scale > 1.0 else 1.04
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
                factor = 0.96 if delta > 0 else 1.04
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

        if len(self.frame_map) <= 1 or content.width() <= 0:
            return

        total_span = float(len(self.frame_map) - 1)

        if self.absence_ranges:
            painter.save()
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(255, 140, 105, 70))
            for start, end in self.absence_ranges:
                start_pos = self._position_for_frame(start)
                end_pos = self._position_for_frame(end + 1)
                if start_pos is None or end_pos is None or end_pos <= start_pos:
                    continue
                left_ratio = (start_pos / total_span) if total_span else 0.0
                right_ratio = (end_pos / total_span) if total_span else 1.0
                left = content.left() + left_ratio * content.width()
                right = content.left() + right_ratio * content.width()
                painter.drawRect(QtCore.QRectF(left, content.top(), max(1.0, right - left), content.height()))
            painter.restore()

        # Draw all markers with reduced opacity
        for marker, color in self.markers.items():
            marker_pos = self._position_for_frame(marker)
            if marker_pos is None:
                continue
            percent = (marker_pos / total_span) if total_span else 0.0
            x = content.left() + percent * content.width()
            painter.setPen(QtGui.QPen(color, 2))
            painter.drawLine(QtCore.QPointF(x, content.top()), QtCore.QPointF(x, content.bottom()))

        # Draw current playhead
        current_pos = self._position_for_frame(self.current_frame)
        if current_pos is not None:
            play_percent = current_pos / total_span if total_span else 0.0
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
        if event.button() != QtCore.Qt.LeftButton or len(self.frame_map) <= 1:
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
            target_position = percent * float(max(0, len(self.frame_map) - 1))
            actual_frame = self._frame_for_position(target_position)
            if actual_frame is not None:
                self.seekRequested.emit(actual_frame)
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

    def _position_for_frame(self, frame: float) -> Optional[float]:
        if not self.frame_map:
            return None
        target = float(frame)
        idx = bisect_left(self.frame_map, int(target))
        if idx < len(self.frame_map) and self.frame_map[idx] == int(target):
            return float(idx)
        if idx == 0 or idx >= len(self.frame_map):
            return None
        prev_frame = self.frame_map[idx - 1]
        next_frame = self.frame_map[idx]
        if next_frame == prev_frame:
            return float(idx)
        ratio = (target - prev_frame) / (next_frame - prev_frame)
        return (idx - 1) + max(0.0, min(1.0, ratio))

    def _frame_for_position(self, position: float) -> Optional[int]:
        if not self.frame_map:
            return None
        if len(self.frame_map) == 1:
            return self.frame_map[0]
        idx = int(round(position))
        idx = max(0, min(idx, len(self.frame_map) - 1))
        return self.frame_map[idx]

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
