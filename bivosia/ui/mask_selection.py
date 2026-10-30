"""Shared, zoom-aware outline for a selected animal mask."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import cv2
from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF

from bivosia.ui.video_panels import ZoomableVideoLabel


class MaskSelectionLabel(ZoomableVideoLabel):
    """Outline one mask with the same screen-space stroke at every zoom level."""

    def __init__(self):
        super().__init__()
        self.selection = QPainterPath()

    def set_selection(self, mask, animal):
        """Select an animal in label-image coordinates, or clear with None."""
        self.selection = QPainterPath()
        if mask is not None and animal is not None and animal > 0:
            height, width = mask.shape
            contours, _ = cv2.findContours((mask == animal).astype('uint8'),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                self.selection.addPolygon(QPolygonF([
                    QPointF(x / width, y / height) for x, y in contour[:, 0]]))
                self.selection.closeSubpath()
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._image is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setClipRect(self._image_rect)
        outline = self.source_transform().map(self.selection)
        for color, width in (('black', 7), ('white', 3)):
            painter.setPen(QPen(QColor(color), width))
            painter.drawPath(outline)
