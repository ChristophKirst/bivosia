"""Interactive interval tracks and recording overview."""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PySide6.QtGui import QPainter

from dataclasses import replace
from html import escape

from PySide6.QtCore import QEvent, QLineF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget
import numpy as np

from bivosia.ui.legend_panel import LegendPanel, legend_groups_html
from bivosia.ui.plot_models import IntervalEdit, IntervalHit, LegendGroup
from bivosia.ui.timeline_base import TimelineBase, TIMELINE_MINIMUM_SPAN
from bivosia.ui.zone_splits import ZoneSplitInteraction


class ZonePanel(QWidget):
    """Zone timeline and legend with optional row visibility checkboxes."""

    rowVisibilityChanged = Signal(str, bool)

    def __init__(self, timeline, *, checkable=False):
        super().__init__()
        self.timeline = timeline
        self.checkboxes = {}
        self._row_labels = {}
        self.legend_layout = QHBoxLayout()
        self.legend = LegendPanel(self.legend_layout)
        if checkable:
            timeline.rowsChanged.connect(self._refresh_legend)
            self._refresh_legend()
        else:
            self.legend_layout.addWidget(timeline.create_legend_label(), 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.legend)
        layout.addWidget(timeline, 1)

    def _refresh_legend(self):
        rows = self.timeline.rows
        if tuple(self.checkboxes) != tuple(row.key for row in rows):
            while self.legend_layout.count():
                item = self.legend_layout.takeAt(0)
                if item.widget() is not None:
                    item.widget().deleteLater()
            self.checkboxes.clear()
            self._row_labels.clear()
            for index, row in enumerate(rows):
                if index:
                    self.legend_layout.addWidget(QLabel('||'))
                checkbox = QCheckBox()
                checkbox.toggled.connect(lambda visible, key=row.key: self._toggle_row(key, visible))
                label = QLabel()
                label.setTextFormat(Qt.TextFormat.RichText)
                label.setMinimumWidth(0)
                self.checkboxes[row.key] = checkbox
                self._row_labels[row.key] = label
                self.legend_layout.addWidget(checkbox)
                self.legend_layout.addWidget(label)
            self.legend_layout.addStretch()
        for row in rows:
            checkbox = self.checkboxes[row.key]
            checkbox.setText(row.label if row.legend_title is None else row.legend_title)
            checkbox.blockSignals(True)
            checkbox.setChecked(row.visible)
            checkbox.blockSignals(False)
            label = self._row_labels[row.key]
            label.setText(self.timeline.row_legend_html(row))
            label.setVisible(bool(label.text()))

    def _toggle_row(self, key, visible):
        self.timeline.set_row_visible(key, visible)
        self.rowVisibilityChanged.emit(key, visible)


def _interval_pixel_runs(
    intervals: np.ndarray,
    first_frame: float,
    last_frame: float,
    pixel_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize inclusive intervals as cells centered on integer frames.

    A zone spans start - 0.5 through end + 0.5, so even a single-frame zone
    has a full frame's width. The playhead remains at the frame's center.
    Subpixel regions retain a minimum width of one screen pixel.
    """
    if intervals.size == 0 or pixel_width <= 0 or last_frame <= first_frame:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty
    visible = (
        (intervals[:, 1] + 0.5 > first_frame)
        & (intervals[:, 0] - 0.5 < last_frame)
    )
    values = intervals[visible]
    if not values.size:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty
    scale = pixel_width / (last_frame - first_frame)
    left = np.floor((values[:, 0] - 0.5 - first_frame) * scale).astype(np.int64)
    right = np.ceil((values[:, 1] + 0.5 - first_frame) * scale).astype(np.int64)
    left = np.clip(left, 0, pixel_width - 1)
    right = np.clip(right, left + 1, pixel_width)
    difference = np.zeros(pixel_width + 1, dtype=np.int32)
    np.add.at(difference, left, 1)
    np.add.at(difference, right, -1)
    occupied = np.cumsum(difference[:-1]) > 0
    changes = np.diff(np.pad(occupied.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1).astype(np.int32)
    ends = np.flatnonzero(changes == -1).astype(np.int32)
    return starts, ends


def _paint_interval_regions(
    painter: QPainter,
    plot: QRectF,
    intervals: np.ndarray,
    first_frame: float,
    last_frame: float,
    y: float,
    height: float,
    color: QColor,
) -> list[QRectF]:
    """Paint clipped interval rectangles for every manual or computed zone row."""
    starts, ends = _interval_pixel_runs(
        intervals, first_frame, last_frame, max(1, round(plot.width()))
    )
    rectangles: list[QRectF] = []
    row_clip = QRectF(plot.left(), y, plot.width(), height).intersected(plot)
    for start_pixel, end_pixel in zip(starts, ends):
        left = plot.left() + int(start_pixel)
        right = min(plot.right(), plot.left() + int(end_pixel))
        rectangle = QRectF(left, y, max(1.0, right - left), height).intersected(
            row_clip
        )
        if rectangle.isEmpty():
            continue
        painter.fillRect(rectangle, color)
        rectangles.append(rectangle)
    return rectangles


class ZoneTimeline(TimelineBase):
    """Render supplied rows in supplied order, with optional per-row interactions."""

    intervalSelected = Signal(object)
    intervalDoubleClicked = Signal(object)
    editRequested = Signal(object)
    rowsChanged = Signal()

    def __init__(self, frame_count, title="zones", *, rows=(), interactions=None):
        super().__init__(frame_count, interactions=interactions)
        self.title = title
        self.rows = ()
        self.selection = None
        self._arrays = {}
        self._legend_labels = []
        self._row_visibility = {}
        self._edit = None
        self._preview = None
        self.split_interaction = ZoneSplitInteraction(self)
        self.set_rows(rows)

    def set_rows(self, rows):
        rows = tuple(replace(row, visible=self._row_visibility.get(row.key, row.visible)) for row in rows)
        if len({row.key for row in rows}) != len(rows):
            raise ValueError("Row keys must be unique")
        arrays = {}
        for row in rows:
            if len({m.key for m in row.markers}) != len(row.markers):
                raise ValueError(f"Marker keys must be unique within row {row.key}")
            if len({t.key for t in row.tracks}) != len(row.tracks):
                raise ValueError(f"Track keys must be unique within row {row.key}")
            for track in row.tracks:
                if len({i.key for i in track.intervals}) != len(track.intervals):
                    raise ValueError(f"Interval keys must be unique within {row.key}/{track.key}")
                if any(i.end >= self.frame_count for i in track.intervals):
                    raise ValueError(f"Interval outside video in {row.key}/{track.key}")
                arrays[row.key, track.key] = np.asarray([
                    (i.start, i.end) for i in track.intervals if i.color is None and not i.bands
                ], dtype=np.int64).reshape(-1, 2)
        self.rows, self._arrays = rows, arrays
        self._edit = self._preview = None
        self.setMinimumHeight(self.minimum_content_height())
        self.set_left_margin(max(self.left_margin, self.required_left_margin()))
        self.invalidate()
        for label in self._legend_labels:
            label.setText(self.legend_html())
        self.rowsChanged.emit()

    def required_left_margin(self):
        return max((self.fontMetrics().horizontalAdvance(row.label) + 18 for row in self.rows if row.visible), default=0)

    def minimum_content_height(self):
        row_height = max(22, self.fontMetrics().height() + 6)
        axis_height = max(38, self.fontMetrics().height() + 20)
        return max(90, row_height * sum(r.visible for r in self.rows) + axis_height)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.FontChange and hasattr(self, "rows"):
            self.setMinimumHeight(self.minimum_content_height())
            self.set_left_margin(max(self.left_margin, self.required_left_margin()))

    def set_row_visible(self, key, visible):
        self._row_visibility[key] = visible
        self.set_rows(self.rows)

    def set_selection(self, selection):
        self.selection = selection
        self.invalidate()

    def row_rects(self):
        plot = self._plot_rect()
        visible = [r for r in self.rows if r.visible]
        total = sum(r.height for r in visible)
        y = plot.top()
        for row in visible:
            height = plot.height() * row.height / total
            yield row, QRectF(plot.left(), y, plot.width(), height)
            y += height

    def row_at(self, point):
        return next((row for row, rect in self.row_rects() if rect.contains(point)), None)

    def marker_rects(self):
        first, last = self.display_range()
        for row, rect in self.row_rects():
            for marker in row.markers:
                if first <= marker.frame <= last:
                    x = self.frame_to_x(marker.frame, self._plot_rect(), first, last)
                    yield row, marker, QRectF(x - 4, rect.top() + 1, 8, 8)

    def marker_at(self, point):
        hits = [(row, marker, rect) for row, marker, rect in self.marker_rects()
                if row.on_marker_double_click is not None and rect.adjusted(-2, -2, 2, 2).contains(point)]
        return min(hits, key=lambda hit: abs(hit[2].center().x() - point.x()))[:2] if hits else None

    def _begin_marker(self, event):
        hit = self.marker_at(event.position()) if event.button() == Qt.MouseButton.LeftButton else None
        if hit is None:
            return False
        self._edit = self._preview = self._drag_mode = None
        return True

    def hit_at(self, point):
        row = self.row_at(point)
        if row is None:
            return None
        frame = self._frame_at_x(point.x())
        # Last drawn track is on top. Within it, pick the smallest containing interval.
        for track in reversed(row.tracks):
            if not track.visible:
                continue
            hits = [i for i in track.intervals if i.start <= frame <= i.end]
            if hits:
                interval = min(hits, key=lambda i: i.end - i.start)
                return IntervalHit(row.key, track.key, interval, frame)
        return None

    def _paint_static(self, painter):
        super()._paint_static(painter)
        plot = self._plot_rect()
        first, last = self.display_range()
        for row, rect in self.row_rects():
            painter.setPen(self.palette().text().color())
            painter.drawText(QRectF(4, rect.top(), max(1, plot.left() - 14), rect.height()),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, row.label)
            y, height = rect.top() + 1, max(1, rect.height() - 2)
            borders = []
            for track in row.tracks:
                if not track.visible:
                    continue
                color = QColor(track.color)
                color.setAlphaF(color.alphaF() * max(0., min(1., track.opacity)))
                _paint_interval_regions(painter, plot, self._arrays[row.key, track.key], first, last, y, height, color)
                for interval in track.intervals:
                    if interval.end + .5 < first or interval.start - .5 > last:
                        continue
                    data = np.asarray(((interval.start, interval.end),))
                    fill = QColor(interval.color or track.color)
                    fill.setAlphaF(fill.alphaF() * max(0., min(1., track.opacity)))
                    selected = self.selection == (row.key, track.key, interval.key)
                    if not (interval.color or interval.bands or selected or row.show_interval_labels):
                        continue
                    rectangles = _paint_interval_regions(
                        painter, plot, data, first, last, y, height,
                        fill if interval.color or interval.bands else QColor("transparent"),
                    )
                    for band in interval.bands:
                        band_color = QColor(band.color)
                        band_color.setAlphaF(band_color.alphaF() * max(0., min(1., track.opacity)))
                        top, band_height = y + height * band.offset, height * band.height
                        band_data = data + (band.frame_inset, -band.frame_inset)
                        _paint_interval_regions(painter, plot, band_data, first, last, top, band_height, band_color)
                        if band.border_color is not None:
                            border_color = QColor(band.border_color)
                            border_color.setAlphaF(border_color.alphaF() * max(0., min(1., track.opacity)))
                            for frame in (band_data[0, 0] - .5, band_data[0, 1] + .5):
                                if first <= frame <= last:
                                    x = self.frame_to_x(frame, plot, first, last)
                                    borders.append((QLineF(x, top, x, top + band_height), border_color))
                    for rectangle in rectangles:
                        if selected:
                            painter.setPen(QPen(QColor("white"), 2))
                            painter.drawRect(rectangle.adjusted(1, 1, -1, -1))
                        if row.show_interval_labels and rectangle.width() > 24:
                            painter.setPen(QColor("#101216"))
                            painter.drawText(rectangle.adjusted(3, 0, -2, 0), Qt.AlignmentFlag.AlignVCenter, interval.label)
            # Paint edges after all fills so overlapping intervals retain their borders.
            if borders:
                painter.save()
                painter.setClipRect(plot)
                for line, color in borders:
                    painter.setPen(QPen(color, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
                    painter.drawLine(line)
                painter.restore()

    def _paint_overlay(self, painter):
        super()._paint_overlay(painter)
        self.split_interaction.paint(painter)
        for row, marker, rect in self.marker_rects():
            painter.setPen(QPen(QColor("#252a33"), 1))
            painter.setBrush(QColor(marker.color))
            painter.drawRect(rect)
        if self._preview is not None:
            request = self._preview
            rect = next((rect for row, rect in self.row_rects() if row.key == request.hit.row_key), None)
            if rect is not None:
                frame = request.frame + (-.5 if request.action in ("start", "move") else .5)
                x = self.frame_to_x(frame, self._plot_rect(), *self.display_range())
                painter.setPen(QPen(QColor("white"), 2, Qt.PenStyle.DashLine))
                painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))

    def _click(self, event):
        row = self.row_at(event.position())
        hit = self.hit_at(event.position())
        if row is not None and hit is not None and row.selectable:
            self.set_selection(hit.selection)
            self.intervalSelected.emit(hit)
            if row.on_select is not None:
                row.on_select(hit)
        super()._click(event)

    def _begin_edit(self, event):
        row = self.row_at(event.position())
        if row is None or event.button() != Qt.MouseButton.LeftButton:
            return False
        if row.edit_requires_alt and not event.modifiers() & Qt.KeyboardModifier.AltModifier:
            return False
        first, last = self.display_range()
        endpoints = []
        if row.editable:
            for track in row.tracks:
                if not track.visible:
                    continue
                for interval in track.intervals:
                    for action, frame in (("start", interval.start - .5), ("end", interval.end + .5)):
                        x = self.frame_to_x(frame, self._plot_rect(), first, last)
                        endpoints.append((abs(x - event.position().x()), action, track.key, interval))
        candidate = min(endpoints, key=lambda item: item[0]) if endpoints else None
        hit = self.hit_at(event.position())
        if candidate is not None and candidate[0] <= 6:
            _, action, track_key, interval = candidate
            hit = IntervalHit(row.key, track_key, interval, self._frame_at_x(event.position().x()))
        elif row.draggable and hit is not None:
            action = "move"
        else:
            return False
        self._edit = (hit, action, self.x_to_frame(event.position().x(), self._plot_rect(), first, last))
        self._drag_x = event.position().x()
        self._drag_moved = False
        return True

    def mousePressEvent(self, event):
        if self._begin_marker(event):
            return
        if event.button() == Qt.MouseButton.LeftButton and self._cursor_at(event.position()) is not None:
            return TimelineBase.mousePressEvent(self, event)
        if not self._begin_edit(event):
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._edit is None:
            return super().mouseMoveEvent(event)
        if event.buttons() & Qt.MouseButton.LeftButton:
            hit, action, initial_frame = self._edit
            self._drag_moved |= abs(event.position().x() - self._drag_x) >= 2
            pointer = self.x_to_frame(event.position().x(), self._plot_rect(), *self.display_range())
            origin = hit.interval.end if action == "end" else hit.interval.start
            self._preview = IntervalEdit(hit, action, self.clamp_frame(origin + pointer - initial_frame))
            self.update()

    def mouseReleaseEvent(self, event):
        if self._edit is None:
            return super().mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            request = self._preview if self._drag_moved else None
            self._edit = self._preview = None
            if request is not None:
                row = next(r for r in self.rows if r.key == request.hit.row_key)
                self.editRequested.emit(request)
                if row.on_edit is not None:
                    row.on_edit(request)
            else:
                self._click(event)
            self.update()

    def mouseDoubleClickEvent(self, event):
        marker = self.marker_at(event.position())
        if marker is not None and self.interactions.double_click and event.button() == Qt.MouseButton.LeftButton:
            self._edit = self._preview = self._drag_mode = None
            marker[0].on_marker_double_click(marker[1])
            return
        row = self.row_at(event.position())
        if row is not None and not row.double_click:
            return
        if self.interactions.double_click and event.button() == Qt.MouseButton.LeftButton:
            self._edit = self._preview = None
            hit = self.hit_at(event.position())
            if hit is not None:
                self.intervalDoubleClicked.emit(hit)
                if row.on_double_click is not None:
                    row.on_double_click(hit)
                    return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._edit = self._preview = self._drag_mode = None
            self.update()
        else:
            super().keyPressEvent(event)

    @staticmethod
    def row_legend_html(row):
        """Format one row's existing color keys without its title."""
        groups = row.legend if row.legend is not None else (
            LegendGroup(items=tuple((t.label, t.color) for t in row.tracks if t.visible)),)
        return legend_groups_html(groups)

    def legend_html(self):
        rows = []
        for row in self.rows:
            text = self.row_legend_html(row)
            if row.visible and text:
                title = row.label if row.legend_title is None else row.legend_title
                rows.append((escape(title) + ':&nbsp;' if title else '') + text)
        return " &nbsp;||&nbsp; ".join(rows)

    def create_legend_label(self):
        label = QLabel(self.legend_html())
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        label.setMinimumWidth(0)
        self._legend_labels.append(label)
        label.destroyed.connect(lambda: self._legend_labels.remove(label) if label in self._legend_labels else None)
        return label


class TimelineOverview(ZoneTimeline):
    """The same row renderer at full extent, with a movable detail viewport."""

    windowMoved = Signal(float, float)

    def __init__(self, frame_count, *, rows=(), interactions=None):
        super().__init__(frame_count, rows=rows, interactions=interactions)
        self._window_drag = None
        self.viewChanged.connect(self.windowMoved.emit)

    def display_range(self):
        return 0., float(max(1, self.frame_count - 1))

    def set_current_frame(self, frame):
        # Seek within the existing detail range.
        self.current_frame = self.clamp_frame(frame)
        self.cursors = tuple(replace(c, frame=self.current_frame) if c.key == "playhead" else c for c in self.cursors)
        self.update()

    def _window_rect(self):
        plot = self._plot_rect()
        left = self.frame_to_x(self.view_start, plot, *self.display_range())
        right = self.frame_to_x(self.view_end, plot, *self.display_range())
        return QRectF(left, plot.top() - 2, max(5., right - left), plot.height() + 4)

    def _paint_overlay(self, painter):
        window = self._window_rect()
        painter.fillRect(window, QColor(255, 255, 255, 35))
        painter.setPen(QPen(QColor("white"), 2))
        painter.drawRect(window)
        super()._paint_overlay(painter)

    def mousePressEvent(self, event):
        if self._begin_marker(event):
            return
        if event.button() != Qt.MouseButton.LeftButton or self._cursor_at(event.position()) is not None:
            return TimelineBase.mousePressEvent(self, event)
        if not self._plot_rect().contains(event.position()):
            return
        if self._begin_edit(event):
            return
        if self._cursor_at(event.position()) is not None or not self.interactions.pan:
            return TimelineBase.mousePressEvent(self, event)
        x = event.position().x()
        frame = self.x_to_frame(x, self._plot_rect(), *self.display_range())
        window = self._window_rect()
        mode = "left" if abs(x - window.left()) <= 7 else "right" if abs(x - window.right()) <= 7 else "move"
        self._window_drag = (mode, frame, self.view_start, self.view_end)
        self._drag_x, self._drag_moved = x, False
        if not window.contains(event.position()) and mode == "move":
            span = self.view_end - self.view_start
            self.set_view(frame - span / 2, frame + span / 2)
            self._window_drag = (mode, frame, self.view_start, self.view_end)
            self.windowMoved.emit(self.view_start, self.view_end)

    def mouseMoveEvent(self, event):
        if self._window_drag is None:
            return super().mouseMoveEvent(event)
        if event.buttons() & Qt.MouseButton.LeftButton:
            mode, anchor, start, end = self._window_drag
            self._drag_moved |= abs(event.position().x() - self._drag_x) >= 2
            frame = self.x_to_frame(event.position().x(), self._plot_rect(), *self.display_range())
            shift = frame - anchor
            minimum = min(TIMELINE_MINIMUM_SPAN, max(1, self.frame_count - 1))
            if mode == "left":
                self.set_view(min(start + shift, end - minimum), end)
            elif mode == "right":
                self.set_view(start, max(start + minimum, end + shift))
            else:
                self.set_view(start + shift, end + shift)
            self.windowMoved.emit(self.view_start, self.view_end)

    def mouseReleaseEvent(self, event):
        if self._window_drag is None:
            return super().mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self._window_drag = None
            if not self._drag_moved:
                self._click(event)
