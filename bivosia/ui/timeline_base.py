"""Shared frame coordinates, cursors, caching and time navigation."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from dataclasses import replace

from PySide6.QtCore import QEvent, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from bivosia.ui.plot_models import CursorSpec, PlotInteractions


TIMELINE_MINIMUM_SPAN = 3.0


class TimelineBase(QWidget):
    frameClicked = Signal(int)
    frameContextRequested = Signal(int, object)
    frameDoubleClicked = Signal(int)
    cursorSelected = Signal(str)
    cursorMoved = Signal(str, int)
    cursorReleased = Signal(str, int)
    viewChanged = Signal(float, float)

    def __init__(self, frame_count: int, *, interactions=None):
        super().__init__()
        if frame_count < 1:
            raise ValueError("A timeline needs at least one frame")
        self.frame_count = frame_count
        self.current_frame = 0
        self.view_start, self.view_end = 0.0, float(max(1, frame_count - 1))
        self.interactions = interactions or PlotInteractions()
        self.cursors = (CursorSpec("playhead", 0),)
        self.active_cursor = None
        self.left_margin = 0
        self.frame_axis_labels_visible = True
        self._revision = 0
        self._cache_key = None
        self._cache = None
        self._drag_mode = None
        self._drag_x = 0.0
        self._drag_view_start = 0.0
        self._drag_moved = False
        self.setMinimumHeight(90)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def sizeHint(self):
        return QSize(900, max(165, 6 * self.fontMetrics().height()))

    @staticmethod
    def frame_to_x(frame, plot, first_frame, last_frame):
        return plot.left() + (frame - first_frame) * plot.width() / max(1.0, last_frame - first_frame)

    @staticmethod
    def x_to_frame(x, plot, first_frame, last_frame):
        fraction = max(0.0, min(1.0, (x - plot.left()) / max(1.0, plot.width())))
        return first_frame + fraction * (last_frame - first_frame)

    def display_range(self):
        return self.view_start, self.view_end

    def _plot_rect(self):
        bottom = max(38, self.fontMetrics().height() + 20) if self.frame_axis_labels_visible else 18
        level_spacing = max(12, self.fontMetrics().height())
        extra = level_spacing * max((c.level for c in self.cursors if c.visible), default=0)
        return QRectF(12 + self.left_margin, 14 + extra, max(1, self.width() - 32 - self.left_margin),
                      max(1, self.height() - bottom - extra))

    def set_left_margin(self, width):
        self.left_margin = max(0, int(width))
        self.invalidate()

    def set_frame_axis_labels_visible(self, visible):
        self.frame_axis_labels_visible = bool(visible)
        self.invalidate()

    def set_interactions(self, interactions):
        self.interactions = interactions
        self._drag_mode = None

    def invalidate(self):
        self._revision += 1
        self.update()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.FontChange:
            self.invalidate()
            self.updateGeometry()

    def set_cursors(self, cursors):
        cursors = tuple(cursors)
        if len({c.key for c in cursors}) != len(cursors):
            raise ValueError("Cursor keys must be unique")
        old_rect = self._plot_rect()
        self.cursors = tuple(replace(c, frame=self.clamp_frame(c.frame)) for c in cursors)
        if self._plot_rect() != old_rect:
            self.invalidate()
        self.update()

    def set_active_cursor(self, key):
        self.active_cursor = key
        self.update()

    def clamp_frame(self, frame):
        return max(0, min(round(frame), self.frame_count - 1))

    def set_current_frame(self, frame):
        self.current_frame = self.clamp_frame(frame)
        self.cursors = tuple(replace(c, frame=self.current_frame) if c.key == "playhead" else c for c in self.cursors)
        if not self.view_start <= self.current_frame <= self.view_end:
            span = self.view_end - self.view_start
            self.set_view(self.current_frame - span / 2, self.current_frame + span / 2)
            self.viewChanged.emit(self.view_start, self.view_end)
        self.update()

    def set_view(self, start, end):
        extent = float(max(1, self.frame_count - 1))
        span = max(min(TIMELINE_MINIMUM_SPAN, extent), min(extent, end - start))
        self.view_start = max(0.0, min(start, extent - span))
        self.view_end = self.view_start + span
        self.update()

    def reset_zoom(self):
        self.set_view(0, max(1, self.frame_count - 1))
        self.viewChanged.emit(self.view_start, self.view_end)

    def _frame_at_x(self, x):
        return self.clamp_frame(self.x_to_frame(x, self._plot_rect(), *self.display_range()))

    def _cursor_owner(self, key):
        return next((c.owner or c.key for c in self.cursors if c.key == key), key)

    def _cursor_at(self, point):
        if not self.interactions.cursor_drag:
            return None
        plot = self._plot_rect()
        first, last = self.display_range()
        owner = self._cursor_owner(self.active_cursor)
        level_spacing = max(12, self.fontMetrics().height())
        hits = []
        for c in self.cursors:
            if not c.visible or not (c.draggable or c.selectable) or not first <= c.frame <= last:
                continue
            dx = abs(point.x() - self.frame_to_x(c.frame, plot, first, last))
            dy = abs(point.y() - (plot.top() - 6 - level_spacing * c.level))
            distance = dx if c.line and plot.top() - 12 <= point.y() <= plot.bottom() else (dx * dx + dy * dy) ** .5
            if distance <= 8:
                hits.append(((c.owner or c.key) != owner, distance, c.key != self.active_cursor, c.key))
        return min(hits)[-1] if hits else None

    def _move_cursor(self, key, x):
        frame = self._frame_at_x(x)
        self.cursors = tuple(replace(c, frame=frame) if c.key == key else c for c in self.cursors)
        if key == "playhead":
            self.current_frame = frame
            if self.interactions.seek:
                self.frameClicked.emit(frame)
        self.cursorMoved.emit(key, frame)
        self.update()

    def _paint_static(self, painter):
        plot = self._plot_rect()
        painter.fillRect(plot, QColor("#252a33"))
        if not self.frame_axis_labels_visible:
            return
        first, last = self.display_range()
        painter.setPen(QColor("#808895"))
        label_width = max(76, self.fontMetrics().horizontalAdvance(f"{self.frame_count - 1:,}") + 12)
        label_height = max(18, self.fontMetrics().height() + 4)
        frames = tuple(dict.fromkeys(
            self.clamp_frame(first + step / 4 * (last - first))
            for step in range(5)
        ))
        for frame in frames:
            x = self.frame_to_x(frame, plot, first, last)
            painter.drawText(QRectF(x - label_width / 2, plot.bottom() + 1,
                                    label_width, label_height),
                             Qt.AlignmentFlag.AlignCenter, f"{frame:,}")

    def _paint_overlay(self, painter):
        plot = self._plot_rect()
        first, last = self.display_range()
        owner = self._cursor_owner(self.active_cursor)
        level_spacing = max(12, self.fontMetrics().height())
        for cursor in sorted(self.cursors, key=lambda c: ((c.owner or c.key) == owner, c.key == self.active_cursor)):
            if not cursor.visible or not first <= cursor.frame <= last:
                continue
            x = self.frame_to_x(cursor.frame, plot, first, last)
            active = cursor.key == self.active_cursor
            if active and cursor.line:
                painter.setPen(QPen(QColor("white"), 5))
                painter.drawLine(int(x), int(plot.top() - 3), int(x), int(plot.bottom() + 3))
            painter.setPen(QPen(QColor(cursor.color), 3 if active else 2))
            if cursor.line:
                painter.drawLine(int(x), int(plot.top() - 3), int(x), int(plot.bottom() + 3))
            painter.setBrush(QColor(cursor.color))
            radius = 5 if active else 3
            if active:
                painter.setPen(QPen(QColor("white"), 1))
            if cursor.outline:
                painter.setPen(QPen(QColor(cursor.outline), 2))
            marker_y = plot.top() - 6 - level_spacing * cursor.level
            if cursor.dot:
                painter.drawEllipse(QRectF(x - radius, marker_y - radius, 2 * radius, 2 * radius))
            painter.setPen(QColor(cursor.color))
            if cursor.label:
                if cursor.dot or cursor.line:
                    painter.drawText(int(x + 6), int(plot.top() + 12 if cursor.line else marker_y + 4), cursor.label)
                else:
                    painter.save()
                    font = painter.font()
                    font.setBold(active)
                    painter.setFont(font)
                    label_width = max(14, self.fontMetrics().horizontalAdvance(cursor.label) + 4)
                    label_height = max(20, self.fontMetrics().height() + 4)
                    painter.drawText(QRectF(x - label_width / 2, marker_y - label_height / 2,
                                            label_width, label_height),
                                     Qt.AlignmentFlag.AlignCenter, cursor.label)
                    painter.restore()

    def paintEvent(self, event):
        key = (self.width(), self.height(), self.devicePixelRatioF(), self.display_range(), self._revision)
        if key != self._cache_key:
            ratio = max(1.0, self.devicePixelRatioF())
            self._cache = QImage(
                max(1, round(self.width() * ratio)),
                max(1, round(self.height() * ratio)),
                QImage.Format.Format_ARGB32_Premultiplied,
            )
            self._cache.setDevicePixelRatio(ratio)
            self._cache.fill(self.palette().color(self.backgroundRole()))
            painter = QPainter(self._cache)
            self._paint_static(painter)
            painter.end()
            self._cache_key = key
        painter = QPainter(self)
        painter.drawImage(0, 0, self._cache)
        self._paint_overlay(painter)

    def mousePressEvent(self, event):
        if (event.button() == Qt.MouseButton.RightButton and self.interactions.seek
                and not event.modifiers() & Qt.KeyboardModifier.ControlModifier
                and self._plot_rect().contains(event.position())):
            self.frameContextRequested.emit(self._frame_at_x(event.position().x()), event.globalPosition().toPoint())
            event.accept()
            return
        cursor = self._cursor_at(event.position())
        if event.button() != Qt.MouseButton.LeftButton or (cursor is None and not self._plot_rect().contains(event.position())):
            return
        self._drag_x = event.position().x()
        self._drag_view_start = self.view_start
        self._drag_moved = False
        self._drag_mode = ("cursor", cursor) if cursor else ("pan", None)
        if cursor:
            if not next(c.draggable for c in self.cursors if c.key == cursor):
                self._drag_mode = ("select", cursor)
            self.cursorSelected.emit(cursor)

    def mouseMoveEvent(self, event):
        if self._drag_mode is None or not event.buttons() & Qt.MouseButton.LeftButton:
            return
        self._drag_moved |= abs(event.position().x() - self._drag_x) >= 2
        mode, key = self._drag_mode
        if mode == "cursor":
            self._move_cursor(key, event.position().x())
        elif mode == "pan" and self.interactions.pan:
            span = self.view_end - self.view_start
            shift = -(event.position().x() - self._drag_x) * span / max(1, self._plot_rect().width())
            self.set_view(self._drag_view_start + shift, self._drag_view_start + shift + span)
            self.viewChanged.emit(self.view_start, self.view_end)

    def _click(self, event):
        if self.interactions.seek:
            self.frameClicked.emit(self._frame_at_x(event.position().x()))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._drag_mode is not None and self._drag_mode[0] == "cursor":
                key = self._drag_mode[1]
                cursor = next((c for c in self.cursors if c.key == key), None)
                if cursor is not None:
                    self.cursorReleased.emit(key, cursor.frame)
            if (self._drag_mode is not None and self._drag_mode[0] != "cursor"
                    and self._drag_mode[0] != "select"
                    and not self._drag_moved and self._plot_rect().contains(event.position())):
                self._click(event)
            self._drag_mode = None

    def mouseDoubleClickEvent(self, event):
        if (self.interactions.double_click and event.button() == Qt.MouseButton.LeftButton
                and self._plot_rect().contains(event.position())):
            self._drag_mode = None
            frame = self._frame_at_x(event.position().x())
            if self.interactions.seek:
                self.frameClicked.emit(frame)
            self.frameDoubleClicked.emit(frame)
            event.accept()

    def wheelEvent(self, event):
        plot = self._plot_rect()
        delta = event.angleDelta().y() or event.pixelDelta().y()
        if not self.interactions.zoom or not plot.contains(event.position()) or not delta:
            return
        old_span = self.view_end - self.view_start
        factor = .75 if delta > 0 else 1 / .75
        extent = float(max(1, self.frame_count - 1))
        span = max(min(TIMELINE_MINIMUM_SPAN, extent), min(extent, old_span * factor))
        anchor = self.x_to_frame(event.position().x(), plot, *self.display_range())
        fraction = max(0.0, min(1.0, (anchor - self.view_start) / max(1, old_span)))
        start = anchor - fraction * span
        self.set_view(start, start + span)
        self.viewChanged.emit(self.view_start, self.view_end)
        event.accept()
