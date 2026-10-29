"""Shared video pane, coordinate transforms and mask compositing."""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QMouseEvent, QWheelEvent

from PySide6.QtCore import QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QTransform
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QRubberBand, QSizePolicy, QSlider, QStyle,
    QWidget,
)
import cv2
import numpy as np

from bivosia.ui.colors import MASK_COLORS_BGR
from bivosia.ui.keypoints import overlay_keypoints


class MaskDisplayControl(QWidget):
    """Shared mask-opacity, outline and video-background controls."""

    valueChanged = Signal()

    def __init__(self, *, keypoints=False, global_colors=False):
        super().__init__()
        self.additive = False
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(45)
        self.slider.setFixedWidth(150)
        self.label = QLabel("45%")
        self.video = QCheckBox("Video")
        self.video.setChecked(True)
        self.outline = QCheckBox("Outline")
        self.outline.setToolTip(
            "Draw opaque one-pixel white boundaries around and between masks"
        )
        layout.addWidget(self.slider)
        layout.addWidget(QLabel("Mask opacity"))
        layout.addWidget(self.label)
        layout.addWidget(self.video)
        layout.addWidget(self.outline)
        self.slider.valueChanged.connect(self._changed)
        self.video.toggled.connect(lambda _checked: self.valueChanged.emit())
        self.outline.toggled.connect(lambda _checked: self.valueChanged.emit())
        self.color_mode = None
        if global_colors:
            layout.addWidget(QLabel('Colors'))
            self.color_mode = QComboBox()
            self.color_mode.addItem('Labels', 'labels')
            self.color_mode.addItem('Global', 'global')
            self.color_mode.setToolTip(
                'Colors masks and assigned keypoints using the editor\'s global link preview. '
                'Unlinked regions have separate palettes; original track colors stay unchanged.')
            layout.addWidget(self.color_mode)
            self.color_mode.currentIndexChanged.connect(lambda _index: self.valueChanged.emit())
        self.keypoints = self.keypoint_mode = self.all_keypoints = None
        if keypoints:
            self.keypoints = QCheckBox('Keypoints')
            self.keypoints.setChecked(True)
            self.keypoint_mode = QComboBox()
            for label, value in (('Original label', 'original'), ('Assigned label', 'assigned'), ('Both', 'both')):
                self.keypoint_mode.addItem(label, value)
            self.keypoint_mode.setToolTip(
                'Assigned colors use the annotation-selected masks in that pane. '
                'Both: original inner disk, assigned outer disk when available.')
            layout.addWidget(self.keypoints)
            self.all_keypoints = QCheckBox('All keypoints')
            self.all_keypoints.setToolTip('Include keypoints excluded from the saved alignment analysis.')
            layout.addWidget(self.all_keypoints)
            layout.addWidget(self.keypoint_mode)
            self.keypoints.toggled.connect(self.keypoint_mode.setEnabled)
            self.keypoints.toggled.connect(self.all_keypoints.setEnabled)
            self.keypoints.toggled.connect(lambda _checked: self.valueChanged.emit())
            self.all_keypoints.toggled.connect(lambda _checked: self.valueChanged.emit())
            self.keypoint_mode.currentIndexChanged.connect(lambda _index: self.valueChanged.emit())

    @property
    def opacity(self):
        return self.slider.value() / 100.0

    @property
    def video_visible(self):
        return self.video.isChecked()

    @property
    def outlines_visible(self):
        return self.outline.isChecked()

    def background(self, original):
        """Return the video frame or a black mask-only background."""
        return original if self.video_visible else np.zeros_like(original)

    def set_additive(self, enabled):
        """Add mask color without dimming the original video pixels."""
        if self.additive != bool(enabled):
            self.additive = bool(enabled)
            self.valueChanged.emit()

    def composite(self, original, masks, palette, label_colors=None):
        """Apply this control's display state through the shared compositor."""
        return overlay_mask(
            original,
            masks,
            self.opacity,
            self.outlines_visible,
            palette,
            label_colors,
            video_visible=self.video_visible,
            additive=self.additive,
        )

    def _changed(self, value):
        self.label.setText(f"{value}%")
        self.valueChanged.emit()

    def composite_keypoints(self, image, data, frame, source, label_colors=None):
        """Apply optional keypoints using this control's shared display state."""
        if self.keypoints is None or not self.keypoints.isChecked() or data is None:
            return image
        points = data[frame]
        if not self.all_keypoints.isChecked():
            selected = [name not in data.excluded_keypoints for name in data.node_names]
            points = points[:, selected, :]
        assignment = np.full(len(points), -1, dtype=np.intp)
        if data.alignment is not None and data.source[frame] == ('forward', 'reverse').index(source):
            assignment = data.alignment.assignment[frame]
        return overlay_keypoints(image, points, assignment, data.image_shape,
                                 self.keypoint_mode.currentData(), label_colors)


class ZoomableVideoLabel(QLabel):
    """Image pane that requests normalized wheel or rectangle zooms."""

    wheelZoomRequested = Signal(float, float, float)
    rectangleZoomRequested = Signal(QRectF)
    panRequested = Signal(float, float)

    def __init__(self) -> None:
        super().__init__("Loading…")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setIndent(0)
        self.setScaledContents(False)
        self.setMinimumSize(180, 120)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setStyleSheet("QLabel { background: black; color: white; }")
        self._image: QImage | None = None
        self._view = QRectF(0.0, 0.0, 1.0, 1.0)
        self._image_rect = QRect()
        self._drag_start: QPoint | None = None
        self._drag_last: QPoint | None = None
        self._drag_mode: str | None = None
        self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.setToolTip("Left-drag: pan · right-drag: rectangle zoom · wheel: zoom")

    def set_image(self, image: QImage) -> None:
        self._image = image
        self._refresh()

    def set_view(self, view: QRectF) -> None:
        self._view = QRectF(view)
        self._refresh()

    def _refresh(self) -> None:
        if self._image is None or self.width() < 1 or self.height() < 1:
            return
        source = QRect(
            round(self._view.left() * self._image.width()),
            round(self._view.top() * self._image.height()),
            max(1, round(self._view.width() * self._image.width())),
            max(1, round(self._view.height() * self._image.height())),
        ).intersected(self._image.rect())
        available = self.contentsRect()
        image = self._image if source == self._image.rect() else self._image.copy(source)
        pixmap = QPixmap.fromImage(image).scaled(
            available.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(pixmap)
        self._image_rect = QStyle.alignedRect(self.layoutDirection(), self.alignment(), pixmap.size(), available)

    def _normalized_source_point(self, point: QPoint) -> tuple[float, float]:
        rect = self._image_rect
        local_x = max(0.0, min(1.0, (point.x() - rect.left()) / max(1, rect.width())))
        local_y = max(0.0, min(1.0, (point.y() - rect.top()) / max(1, rect.height())))
        return (
            self._view.left() + local_x * self._view.width(),
            self._view.top() + local_y * self._view.height(),
        )

    def source_transform(self):
        """Map normalized image coordinates into the widget, including clipped regions."""
        transform = QTransform()
        transform.translate(self._image_rect.left(), self._image_rect.top())
        transform.scale(self._image_rect.width() / self._view.width(),
                        self._image_rect.height() / self._view.height())
        transform.translate(-self._view.left(), -self._view.top())
        return transform

    def source_point(self, x, y):
        """Map normalized image coordinates to the visible widget viewport."""
        point = QPointF(x, y)
        return self.source_transform().map(point) if self._view.contains(point) else None

    def wheelEvent(self, event: QWheelEvent) -> None:
        if not self._image_rect.contains(event.position().toPoint()):
            return
        x, y = self._normalized_source_point(event.position().toPoint())
        factor = 0.8 if event.angleDelta().y() > 0 else 1.25
        self.wheelZoomRequested.emit(factor, x, y)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        point = event.position().toPoint()
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton) and self._image_rect.contains(point):
            self._drag_start = point
            self._drag_last = point
            self._drag_mode = "pan" if event.button() == Qt.MouseButton.LeftButton else "rectangle"
            if self._drag_mode == "rectangle":
                self._rubber_band.setGeometry(QRect(point, QSize()))
                self._rubber_band.show()
            else:
                self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_mode == "rectangle" and self._drag_start is not None:
            selection = QRect(self._drag_start, event.position().toPoint()).normalized().intersected(self._image_rect)
            self._rubber_band.setGeometry(selection)
        elif self._drag_mode == "pan" and self._drag_last is not None:
            point = event.position().toPoint()
            dx = -(point.x() - self._drag_last.x()) * self._view.width() / max(1, self._image_rect.width())
            dy = -(point.y() - self._drag_last.y()) * self._view.height() / max(1, self._image_rect.height())
            self._drag_last = point
            self.panRequested.emit(dx, dy)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton) or self._drag_start is None:
            return
        if self._drag_mode == "pan":
            self._drag_start = None
            self._drag_last = None
            self._drag_mode = None
            self.unsetCursor()
            return
        selection = self._rubber_band.geometry().intersected(self._image_rect)
        self._rubber_band.hide()
        self._drag_start = None
        self._drag_last = None
        self._drag_mode = None
        if selection.width() < 8 or selection.height() < 8:
            return
        left, top = self._normalized_source_point(selection.topLeft())
        right, bottom = self._normalized_source_point(selection.bottomRight())
        self.rectangleZoomRequested.emit(QRectF(left, top, right - left, bottom - top))

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        self._refresh()


def overlay_mask(
    original, painted, opacity: float, outlines: bool, palette: np.ndarray | None,
    label_colors: np.ndarray | None = None, *, video_visible: bool = True,
    additive: bool = False,
):
    """Overlay mask pixels, optionally adding color to the undimmed video."""
    label_masks = painted.ndim == 2
    if label_masks:
        colors = label_colors if label_colors is not None else MASK_COLORS_BGR
        return _overlay_labels(
            original, painted, opacity, outlines, colors, video_visible, additive
        )
    if painted.shape[:2] != original.shape[:2]:
        painted = cv2.resize(painted, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR)
    mask = cv2.cvtColor(painted, cv2.COLOR_BGR2GRAY) > 8
    painted_color = painted
    if palette is None:
        raise ValueError("A learned palette is required for painted mask videos")
    colors_float = painted.astype(np.float32)
    assignments = ((colors_float[:, :, None, :] - palette[None, None, :, :]) ** 2).sum(axis=3).argmin(axis=2)
    identities = list(range(len(palette)))
    result = original.copy() if video_visible else np.zeros_like(original)
    blended = (
        cv2.addWeighted(original, 1.0 if additive else 1.0 - opacity, painted_color, opacity, 0)
        if video_visible else painted_color
    )
    result[mask] = blended[mask]
    if outlines:
        edges = np.zeros(mask.shape, dtype=bool)
        kernel = np.ones((3, 3), np.uint8)
        # Outline every learned color identity independently. Touching masks
        # therefore receive a boundary on each side of their shared edge.
        for identity in identities:
            individual = ((assignments == identity) & mask).astype(np.uint8)
            eroded = cv2.erode(individual, kernel, iterations=1).astype(bool)
            edges |= individual.astype(bool) & ~eroded
        result[edges] = (255, 255, 255)
    return result


def _overlay_labels(original, masks, opacity, outlines, colors, video_visible, additive):
    """Composite only mask bounding boxes, keeping full-resolution pixels and outlines."""
    height, width = original.shape[:2]
    mask_height, mask_width = masks.shape
    labels = masks if masks.shape == original.shape[:2] else cv2.resize(
        masks, (width, height), interpolation=cv2.INTER_NEAREST)
    result = original.copy() if video_visible else np.zeros_like(original)
    for identity in np.unique(masks):
        if identity <= 0:
            continue
        x, y, w, h = cv2.boundingRect((masks == identity).astype(np.uint8))
        # Include one background pixel so erosion also outlines each box edge.
        x0, y0 = max(0, int(x * width / mask_width) - 1), max(0, int(y * height / mask_height) - 1)
        x1 = min(width, int(np.ceil((x + w) * width / mask_width)) + 1)
        y1 = min(height, int(np.ceil((y + h) * height / mask_height)) + 1)
        mask = (labels[y0:y1, x0:x1] == identity).astype(np.uint8)
        view = result[y0:y1, x0:x1]
        color = np.empty_like(view)
        color[:] = colors[(int(identity) - 1) % len(colors)]
        blended = (
            cv2.addWeighted(
                original[y0:y1, x0:x1], 1.0 if additive else 1.0 - opacity, color, opacity, 0
            )
            if video_visible else color
        )
        cv2.copyTo(blended, mask, view)
        if outlines:
            view[mask > cv2.erode(mask, None)] = 255
    return result


def image_from_bgr(frame):
    # PySide retains the buffer; no full-frame channel conversion or copy is needed.
    frame = np.ascontiguousarray(frame)
    height, width = frame.shape[:2]
    return QImage(frame.data, width, height, frame.strides[0], QImage.Format.Format_BGR888)


class VideoViewGroup:
    """One normalized zoom/pan viewport shared by any collection of video panes."""

    def __init__(self, panes, on_changed=None, *, lock_aspect=False, minimum_span=.02):
        self.panes = ()
        self.on_changed = on_changed
        self.lock_aspect = bool(lock_aspect)
        self.minimum_span = float(minimum_span)
        self.view = QRectF(0, 0, 1, 1)
        for pane in panes:
            self.add_pane(pane)

    def add_pane(self, pane):
        """Join a video pane to the current shared zoom and pan viewport."""
        if pane in self.panes:
            return
        self.panes = (*self.panes, pane)
        pane.wheelZoomRequested.connect(self.zoom)
        pane.rectangleZoomRequested.connect(self.set_view)
        pane.panRequested.connect(self.pan)
        pane.set_view(self.view)

    def set_view(self, rect):
        rect = rect.normalized()
        if self.lock_aspect:
            width = height = min(1., max(self.minimum_span, rect.width(), rect.height()))
            left = rect.center().x() - width / 2
            top = rect.center().y() - height / 2
        else:
            width = min(1., max(self.minimum_span, rect.width()))
            height = min(1., max(self.minimum_span, rect.height()))
            left, top = rect.left(), rect.top()
        self.view = QRectF(
            max(0., min(1 - width, left)),
            max(0., min(1 - height, top)),
            width,
            height,
        )
        for pane in self.panes:
            pane.set_view(self.view)
        if self.on_changed:
            self.on_changed()

    def zoom(self, factor, x, y):
        self.set_view(QRectF(x - (x - self.view.left()) * factor, y - (y - self.view.top()) * factor,
                            self.view.width() * factor, self.view.height() * factor))

    def pan(self, dx, dy):
        self.set_view(self.view.translated(dx, dy))

    def reset(self):
        self.set_view(QRectF(0, 0, 1, 1))

    def fit_masks(self, masks, image_shape, border=50):
        """Fit all foreground masks with padding measured in original image pixels."""
        bounds = []
        for mask in masks:
            foreground = mask > 0 if mask.ndim == 2 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) > 8
            if not foreground.any():
                continue
            x, y, width, height = cv2.boundingRect(foreground.astype(np.uint8))
            h, w = mask.shape[:2]
            bounds.append((x / w, y / h, (x + width) / w, (y + height) / h))
        if not bounds:
            return
        h, w = image_shape[:2]
        left = max(0., min(b[0] for b in bounds) - border / w)
        top = max(0., min(b[1] for b in bounds) - border / h)
        right = min(1., max(b[2] for b in bounds) + border / w)
        bottom = min(1., max(b[3] for b in bounds) + border / h)
        self.set_view(QRectF(left, top, right - left, bottom - top))
