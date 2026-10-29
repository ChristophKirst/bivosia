"""Render and drag supplied internal split boundaries on a zone timeline."""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QColor, QPen


class ZoneSplitInteraction(QObject):
    """Handle split gestures without making the zone's outer endpoints editable."""

    def __init__(self, timeline):
        super().__init__(timeline)
        self.timeline = timeline
        self.drag = None
        self.consume_release = False
        timeline.installEventFilter(self)

    def paint(self, painter):
        plot = self.timeline
        first, last = plot.display_range()
        painter.setPen(QPen(QColor('white'), 2))
        for row, rectangle in plot.row_rects():
            for boundary in row.boundaries:
                frame = (self.drag[2] if self.drag is not None
                         and self.drag[0].key == row.key and self.drag[1].key == boundary.key
                         else boundary.frame)
                if first <= frame - .5 <= last:
                    x = plot.frame_to_x(frame - .5, plot._plot_rect(), first, last)
                    painter.drawLine(int(x), int(rectangle.top()), int(x), int(rectangle.bottom()))

    def eventFilter(self, watched, event):
        kind = event.type()
        plot = self.timeline
        if kind == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            row = plot.row_at(event.position())
            if row is None:
                return False
            control = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            candidates = [(abs(event.position().x() - plot.frame_to_x(
                b.frame - .5, plot._plot_rect(), *plot.display_range())), b) for b in row.boundaries]
            closest = min(candidates, key=lambda item: item[0]) if candidates else None
            if closest is not None and closest[0] <= 6:
                boundary = closest[1]
                plot.setFocus()
                if control and row.on_boundary_activate is not None:
                    row.on_boundary_activate(boundary.key)
                    self.consume_release = True
                elif boundary.movable and row.on_boundary_move is not None:
                    self.drag = (row, boundary, boundary.frame)
                else:
                    self.consume_release = True
                return True
            if control and row.on_split is not None:
                hit = plot.hit_at(event.position())
                if hit is not None:
                    plot.setFocus()
                    self.consume_release = True
                    row.on_split(hit)
                    return True
        elif kind == QEvent.Type.MouseMove and self.drag is not None:
            row, boundary, _ = self.drag
            frame = plot._frame_at_x(event.position().x())
            self.drag = row, boundary, max(boundary.minimum, min(frame, boundary.maximum))
            plot.update()
            return True
        elif kind == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
            if self.drag is not None:
                row, boundary, frame = self.drag
                self.drag = None
                if frame != boundary.frame:
                    row.on_boundary_move(boundary.key, frame)
                plot.update()
                return True
            if self.consume_release:
                self.consume_release = False
                return True
        elif kind == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape and self.drag is not None:
            self.drag = None
            plot.update()
            return True
        return False
