"""Interactive mask-link diagrams and video endpoints."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QWidget
import numpy as np

from bivosia.ui.video_panels import VideoViewGroup, ZoomableVideoLabel, image_from_bgr, overlay_mask
from bivosia.ui.colors import bgr_colors, identity_colors


class LinkDiagram(QWidget):
    dragStarted = Signal(str, int)
    dragTarget = Signal(int)
    dragFinished = Signal(bool)

    def __init__(self, n):
        super().__init__()
        self.n = n
        self.mapping = tuple(range(n))
        self.colors = {side: identity_colors(n, .27) for side in ("left", "right")}
        self.present = {side: (True,) * n for side in ("left", "right")}
        self.missing_color = QColor("#69717c")
        self._drag = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(200, max(95, 24 * n + 24))
        self.setMaximumHeight(max(130, 30 * n + 30))
        self.setToolTip("Drag either endpoint onto another disk on the same side; return to its original disk to restore the mapping.")

    def point(self, side, label):
        return QPointF(30 if side == "left" else self.width() - 30, 20 + (label + .5) * max(1, self.height() - 30) / self.n)

    def set_mapping(self, mapping):
        self.mapping = tuple(mapping)
        self.update()

    def set_masks(self, colors, present):
        self.colors, self.present = colors, present
        self.update()

    def disk_color(self, side, label):
        return self.colors[side][label] if self.present[side][label] else self.missing_color

    def link_color(self, left, right):
        return self.colors["left"][left] if self.present["left"][left] and self.present["right"][right] else self.missing_color

    def _hit(self, point, side):
        for label in range(self.n):
            if (point - self.point(side, label)).manhattanLength() <= 16:
                return label
        return None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#171c24"))
        painter.setPen(QColor("#d8dee9"))
        painter.drawText(12, 14, "Left")
        painter.drawText(self.width() - 55, 14, "Right")
        for i, j in enumerate(self.mapping):
            painter.setPen(QPen(self.link_color(i, j), 2))
            painter.drawLine(self.point("left", i), self.point("right", j))
        for side in ("left", "right"):
            for label in range(self.n):
                painter.setBrush(self.disk_color(side, label))
                painter.setPen(QPen(QColor("white"), 1))
                painter.drawEllipse(self.point(side, label), 9, 9)
                painter.setPen(QColor("black"))
                painter.drawText(QRectF(self.point(side, label) - QPointF(8, 8), self.point(side, label) + QPointF(8, 8)), Qt.AlignmentFlag.AlignCenter, str(label + 1))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            for side in ("left", "right"):
                label = self._hit(event.position(), side)
                if label is not None:
                    self._drag = side
                    self.dragStarted.emit(side, label)
                    return

    def mouseMoveEvent(self, event):
        if self._drag:
            label = self._hit(event.position(), self._drag)
            if label is not None:
                self.dragTarget.emit(label)

    def mouseReleaseEvent(self, event):
        if self._drag and event.button() == Qt.MouseButton.LeftButton:
            label = self._hit(event.position(), self._drag)
            if label is not None:
                self.dragTarget.emit(label)
            self._drag = None
            self.dragFinished.emit(label is not None)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self._drag:
            self._drag = None
            self.dragFinished.emit(False)
        else:
            super().keyPressEvent(event)


class MaskPane(ZoomableVideoLabel):
    dragStarted = Signal(str, int)
    dragTarget = Signal(int)
    dragFinished = Signal(bool)

    def __init__(self, side, source):
        super().__init__()
        self.setAlignment((Qt.AlignmentFlag.AlignRight if side == "left" else Qt.AlignmentFlag.AlignLeft
                           if side == "right" else Qt.AlignmentFlag.AlignHCenter)
                          | Qt.AlignmentFlag.AlignVCenter)
        self.side, self.source = side, source
        self.masks = None
        self.centers = ()
        self.colors = []
        self.active = False
        self.linking = False
        self._link_drag = False
        self._original = None
        self._opacity = None
        self._video_visible = None
        self._outlines = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_frame(
        self, original, masks, centers, colors, active, linking, opacity,
        video_visible, outlines,
    ):
        image_changed = (original is not self._original or masks is not self.masks
                         or colors != self.colors or opacity != self._opacity
                         or video_visible != self._video_visible
                         or outlines != self._outlines)
        if self._original is None or active != self.active:
            self.setStyleSheet("QLabel { background: black; border: 2px solid " + ("#eee" if active else "#555") + "; }")
        if active != self.active or linking != self.linking:
            self.update()
        self.masks, self.centers, self.colors = masks, centers, colors
        self.active, self.linking = active, linking
        self._original = original
        self._opacity, self._video_visible = opacity, video_visible
        self._outlines = outlines
        if image_changed:
            self.set_image(image_from_bgr(overlay_mask(
                original, masks, opacity, outlines, None, bgr_colors(colors),
                video_visible=video_visible,
            )))

    def center_point(self, label):
        if self.masks is None or label >= len(self.centers) or self.centers[label] is None:
            return None
        x, y = self.centers[label]
        return self.source_point(x / self.masks.shape[1], y / self.masks.shape[0])

    def _mask_at(self, point):
        if self.masks is None or not self._image_rect.contains(point.toPoint()):
            return None
        x, y = self._normalized_source_point(point.toPoint())
        row = min(self.masks.shape[0] - 1, max(0, int(y * self.masks.shape[0])))
        column = min(self.masks.shape[1] - 1, max(0, int(x * self.masks.shape[1])))
        value = int(self.masks[row, column])
        return value - 1 if 1 <= value <= len(self.colors) else None

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.active or not self.linking:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for label, color in enumerate(self.colors):
            point = self.center_point(label)
            if point is not None:
                painter.setBrush(color)
                painter.setPen(QPen(QColor("white"), 2))
                painter.drawEllipse(point, 7, 7)
                painter.drawText(point + QPointF(9, 0), str(label + 1))

    def mousePressEvent(self, event):
        if self.active and self.linking and event.button() == Qt.MouseButton.LeftButton:
            label = next((i for i in range(len(self.centers)) if self.center_point(i) is not None
                          and (event.position() - self.center_point(i)).manhattanLength() <= 10),
                         self._mask_at(event.position()))
            if label is not None:
                self._link_drag = True
                self.dragStarted.emit(self.side, label)
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._link_drag:
            label = self._mask_at(event.position())
            if label is not None:
                self.dragTarget.emit(label)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._link_drag and event.button() == Qt.MouseButton.LeftButton:
            label = self._mask_at(event.position())
            if label is not None:
                self.dragTarget.emit(label)
            self._link_drag = False
            self.dragFinished.emit(label is not None)
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self._link_drag:
            self._link_drag = False
            self.dragFinished.emit(False)
        else:
            super().keyPressEvent(event)


class _ConnectionOverlay(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)

    def paintEvent(self, event):
        grid = self.parent()
        if not grid.linking:
            return
        left, right = (grid.panes[(side, grid.sources[side])] for side in ("left", "right"))
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for i, j in enumerate(grid.mapping):
            a, b = left.center_point(i), right.center_point(j)
            if a is not None and b is not None:
                a += QPointF(left.mapTo(grid, left.rect().topLeft()))
                b += QPointF(right.mapTo(grid, right.rect().topLeft()))
                painter.setPen(QPen(left.colors[i], 2))
                painter.drawLine(a, b)


class MaskVideoGrid(QWidget):
    def __init__(self, sides):
        super().__init__()
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(0)
        self.panes, self.headers = {}, {}
        self.sources = {side: "forward" for side in sides}
        self.mapping = ()
        self.linking = False
        for column, side in enumerate(sides):
            self.headers[side] = QHBoxLayout()
            layout.addLayout(self.headers[side], 0, column)
            for row, source in enumerate(("forward", "reverse")):
                pane = MaskPane(side, source)
                self.panes[side, source] = pane
                layout.addWidget(pane, row + 1, column)
                layout.setRowStretch(row + 1, 1)
            layout.setColumnStretch(column, 1)
        self.overlay = _ConnectionOverlay(self)
        self.navigation = VideoViewGroup(self.panes.values(), self.overlay.update)

    def set_mapping(self, mapping):
        self.mapping = tuple(mapping)
        self.overlay.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.overlay.setGeometry(self.rect())
        self.overlay.raise_()
