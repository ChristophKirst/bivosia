"""Cached two-dimensional scatter plots with zoom, pan and sample selection."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from bivosia.ui.embedding_colors import embedding_bounds


class Embedding2DPlot(QWidget):
    """Render supplied coordinates and keep the visible extent in embedding units."""

    pointSelected = Signal(int)

    def __init__(self):
        super().__init__()
        self.points = self.colors = self._image = self._positions = None
        self.bounds = (np.zeros(2), np.ones(2))
        self.selected = ()
        self._press = self._drag_bounds = None
        self._dragged = False
        self.setMinimumSize(240, 240)
        self.setToolTip('Click: select sample · drag: pan · wheel: zoom')

    def plot_rect(self):
        return QRectF(48, 16, max(1, self.width() - 64), max(1, self.height() - 62))

    def set_points(self, points, colors):
        changed = self.points is not points
        self.points, self.colors = points, colors
        self.selected = ()
        if changed and points is not None and len(points):
            self.bounds = embedding_bounds(points)
        self.invalidate()

    def invalidate(self):
        self._image = self._positions = None
        self.update()

    def set_selected(self, indices):
        self.selected = tuple(indices)
        self.update()

    def set_view(self, low, high):
        self.bounds = np.asarray(low).copy(), np.asarray(high).copy()
        self.invalidate()

    def screen_positions(self):
        low, high = self.bounds
        box = self.plot_rect()
        scaled = (self.points - low) / (high - low)
        return np.column_stack((box.left() + scaled[:, 0] * box.width(),
                                box.bottom() - scaled[:, 1] * box.height()))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.plot_rect().contains(event.position()):
            self._press = event.position()
            self._drag_bounds = tuple(value.copy() for value in self.bounds)
            self._dragged = False
            event.accept()

    def mouseMoveEvent(self, event):
        if self._press is None:
            return
        delta = event.position() - self._press
        if not self._dragged and abs(delta.x()) + abs(delta.y()) < 4:
            return
        self._dragged = True
        low, high = self._drag_bounds
        box = self.plot_rect()
        shift = np.array((-delta.x() / box.width(), delta.y() / box.height())) * (high - low)
        self.set_view(low + shift, high + shift)

    def mouseReleaseEvent(self, event):
        if self._press is not None and not self._dragged and self.points is not None and len(self.points):
            positions = self.screen_positions()
            distance = ((positions - (event.position().x(), event.position().y())) ** 2).sum(axis=1)
            index = int(np.argmin(distance))
            if distance[index] <= self.pick_radius() ** 2:
                self.pointSelected.emit(index)
        self._press = None

    def pick_radius(self):
        return 8

    def wheelEvent(self, event):
        box = self.plot_rect()
        if not box.contains(event.position()) or not event.angleDelta().y():
            return
        low, high = self.bounds
        fraction = np.array(((event.position().x() - box.left()) / box.width(),
                             (box.bottom() - event.position().y()) / box.height()))
        anchor = low + fraction * (high - low)
        factor = 1.2 ** (-event.angleDelta().y() / 120)
        self.set_view(anchor + (low - anchor) * factor, anchor + (high - anchor) * factor)
        event.accept()

    def resizeEvent(self, event):
        self.invalidate()
        super().resizeEvent(event)

    def _draw_points(self, painter, box):
        if self._image is None:
            width, height = self.width(), self.height()
            pixels = np.zeros((height, width, 4), dtype=np.uint8)
            self._positions = self.screen_positions()
            visible = np.all((self.points >= self.bounds[0]) & (self.points <= self.bounds[1]), axis=1)
            x, y = np.rint(self._positions[visible]).astype(int).T
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    pixels[np.clip(y + dy, 0, height - 1), np.clip(x + dx, 0, width - 1)] = self.colors[visible]
            self._image = QImage(pixels.data, width, height, pixels.strides[0], QImage.Format.Format_RGBA8888).copy()
        painter.drawImage(0, 0, self._image)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().base())
        painter.setPen(self.palette().text().color())
        box = self.plot_rect()
        painter.drawRect(box)
        painter.drawText(QRectF(box.left(), box.bottom() + 8, box.width(), 24), Qt.AlignmentFlag.AlignCenter, 'UMAP 1')
        painter.save()
        painter.translate(18, box.center().y())
        painter.rotate(-90)
        painter.drawText(QRectF(-box.height() / 2, -12, box.height(), 24), Qt.AlignmentFlag.AlignCenter, 'UMAP 2')
        painter.restore()
        if self.points is None or not len(self.points):
            return
        painter.setClipRect(box)
        self._draw_points(painter, box)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(self.palette().text().color(), 2))
        positions = self.screen_positions() if self._positions is None else self._positions
        for index in self.selected:
            painter.drawEllipse(QPointF(*positions[index]), 6, 6)
