"""Mask selection, point prompts and zoom-aware brush interaction."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import cv2
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF

from bivosia.ui.video_panels import ZoomableVideoLabel


class MaskCanvas(ZoomableVideoLabel):
    maskClicked = Signal(float, float)
    pointAdded = Signal(float, float)
    strokeStarted = Signal(float, float)
    strokeMoved = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.tool, self.radius = "Prompt", 3
        self.prompts, self.colors = [], []
        self.selection = QPainterPath()
        self.pointer = None
        self._stroke = False
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setToolTip("Select: click mask · Prompt: click point · Paint/Erase: left-drag · Shift+drag: pan · right-drag or wheel: zoom")

    def set_tool(self, tool):
        self.tool = tool
        self.update()

    def set_radius(self, radius):
        self.radius = radius
        self.update()

    def pixel(self, point):
        x, y = self._normalized_source_point(point)
        return min(self._image.width() - 1, x * self._image.width()), min(self._image.height() - 1, y * self._image.height())

    def set_selection(self, mask, animal):
        """Outline the selected animal with a fixed screen-space stroke."""
        self.selection = QPainterPath()
        if animal is not None and animal > 0:
            height, width = mask.shape
            contours, _ = cv2.findContours((mask == animal).astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                self.selection.addPolygon(QPolygonF([QPointF(x / width, y / height) for x, y in contour[:, 0]]))
                self.selection.closeSubpath()
        self.update()

    def brush_rect(self):
        """Return the brush footprint in widget coordinates, or None outside painting."""
        if (self._image is None or self.pointer is None or self.tool not in ("Paint", "Erase")
                or self._drag_mode is not None or not self._image_rect.contains(self.pointer.toPoint())):
            return None
        x, y = map(round, self.pixel(self.pointer.toPoint()))
        width, height = self._image.width(), self._image.height()
        return self.source_transform().mapRect(QRectF((x - self.radius) / width, (y - self.radius) / height,
                                                      2 * self.radius / width, 2 * self.radius / height))

    def mousePressEvent(self, event):
        self.setFocus()
        self.pointer = event.position()
        if (self._image is not None and self.tool != "Pan"
                and event.button() == Qt.MouseButton.LeftButton
                and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier
                and self._image_rect.contains(event.position().toPoint())):
            x, y = self.pixel(event.position().toPoint())
            if self.tool == "Select":
                self.maskClicked.emit(x, y)
            elif self.tool == "Prompt":
                self.pointAdded.emit(x, y)
            else:
                self._stroke = True
                self.strokeStarted.emit(x, y)
            self.update()
            return
        super().mousePressEvent(event)
        self.update()

    def mouseMoveEvent(self, event):
        self.pointer = event.position()
        if self._stroke:
            if self._image_rect.contains(event.position().toPoint()):
                self.strokeMoved.emit(*self.pixel(event.position().toPoint()))
        else:
            super().mouseMoveEvent(event)
        self.update()

    def mouseReleaseEvent(self, event):
        self.pointer = event.position()
        if self._stroke and event.button() == Qt.MouseButton.LeftButton:
            if self._image_rect.contains(event.position().toPoint()):
                self.strokeMoved.emit(*self.pixel(event.position().toPoint()))
            self._stroke = False
        else:
            super().mouseReleaseEvent(event)
        self.update()

    def leaveEvent(self, event):
        self.pointer = None
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._image is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.save()
        painter.setClipRect(self._image_rect)
        outline = self.source_transform().map(self.selection)
        for color, width in (("black", 7), ("white", 3)):
            painter.setPen(QPen(QColor(color), width))
            painter.drawPath(outline)
        painter.restore()
        for x, y, animal in self.prompts:
            point = self.source_point(x / self._image.width(), y / self._image.height())
            if point is None:
                continue
            color = self.colors[animal - 1] if animal else QColor("white")
            painter.setPen(QPen(QColor("black"), 4))
            painter.drawEllipse(point, 5, 5)
            painter.setPen(QPen(color, 2))
            painter.drawEllipse(point, 5, 5)
            painter.drawText(int(point.x() + 7), int(point.y() - 4), str(animal) if animal else "B")
        brush = self.brush_rect()
        if brush is not None:
            painter.setClipRect(self._image_rect)
            for color, width in (("black", 3), ("white", 1)):
                painter.setPen(QPen(QColor(color), width))
                painter.drawEllipse(brush)
