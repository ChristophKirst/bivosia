"""Shared two-dimensional scatter display for saved embeddings."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QImage, QPainter, QPalette
from PySide6.QtWidgets import QWidget


class EmbeddingPlot(QWidget):
    """Map supplied coordinates onto a cached scatter image without fitting models."""

    def __init__(self, axis_labels=('UMAP 1', 'UMAP 2')):
        super().__init__()
        self.axis_labels = axis_labels
        self.points = None
        self._image = None
        self.setMinimumSize(240, 240)

    def set_points(self, points):
        self.points = points
        self._image = None
        self.update()

    def resizeEvent(self, event):
        self._image = None
        super().resizeEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().base())
        box = QRectF(48, 16, max(1, self.width() - 64), max(1, self.height() - 62))
        painter.setPen(self.palette().color(QPalette.ColorRole.Text))
        painter.drawRect(box)
        painter.drawText(QRectF(box.left(), box.bottom() + 8, box.width(), 24),
                         Qt.AlignmentFlag.AlignCenter, self.axis_labels[0])
        painter.save()
        painter.translate(18, box.center().y())
        painter.rotate(-90)
        painter.drawText(QRectF(-box.height() / 2, -12, box.height(), 24),
                         Qt.AlignmentFlag.AlignCenter, self.axis_labels[1])
        painter.restore()
        if self.points is None or not len(self.points):
            return
        if self._image is None:
            width, height = max(1, int(box.width()) - 2), max(1, int(box.height()) - 2)
            low, high = self.points.min(axis=0), self.points.max(axis=0)
            span = np.maximum(high - low, np.finfo(float).eps)
            scaled = .025 + .95 * (self.points - low) / span
            x = np.clip(np.rint(scaled[:, 0] * (width - 1)).astype(int), 0, width - 1)
            y = np.clip(np.rint((1 - scaled[:, 1]) * (height - 1)).astype(int), 0, height - 1)
            pixels = np.empty((height, width, 4), dtype=np.uint8)
            pixels[:] = self.palette().color(QPalette.ColorRole.Base).getRgb()
            color = self.palette().color(QPalette.ColorRole.Highlight).getRgb()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    pixels[np.clip(y + dy, 0, height - 1), np.clip(x + dx, 0, width - 1)] = color
            self._image = QImage(pixels.data, width, height, pixels.strides[0],
                                  QImage.Format.Format_RGBA8888).copy()
        painter.drawImage(box.topLeft() + QPointF(1, 1), self._image)
