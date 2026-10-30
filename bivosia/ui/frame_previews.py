"""Selectable frame thumbnails with labeling and boundary indicators."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import cv2
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QListWidget, QListWidgetItem

from bivosia.ui.colors import KEY_FRAME_COLORS, bgr_colors, identity_colors
from bivosia.ui.video_panels import image_from_bgr, overlay_mask


class FramePreviews(QListWidget):
    frameSelected = Signal(int)

    def __init__(self):
        super().__init__()
        self.setIconSize(QSize(160, 100))
        self.images, self.items = {}, {}
        self._videos = None
        self._pending = []
        self.masks, self.crop, self.colors = {}, (), None
        self.itemClicked.connect(lambda item: self.frameSelected.emit(item.data(Qt.ItemDataRole.UserRole)))

    def set_video(self, videos):
        self._videos = videos
        self.images.clear()
        self.clear_pending()

    def set_session(self, session):
        self.set_frames(session.frames, session.masks, session.crop, bgr_colors(identity_colors(session.animals, .27)))

    def set_frames(self, frames, masks, crop, colors):
        self.masks, self.crop, self.colors = masks, crop, colors
        self.clear()
        self.items = {}
        self.images = {f: image for f, image in self.images.items() if f in frames}
        self._pending = [frame for frame in frames if frame not in self.images]
        for frame in frames:
            labeled = frame in masks
            status = "Labeled" if labeled else "Unlabeled"
            item = QListWidgetItem(f"{frame:,}\n{status}")
            item.setData(Qt.ItemDataRole.UserRole, frame)
            item.setForeground(QColor(KEY_FRAME_COLORS["labeled" if labeled else "proposed"]))
            if frame in crop:
                item.setToolTip("Crop boundary" + (" · labeled" if labeled else ""))
            self.addItem(item)
            self.items[frame] = item
            self._render(frame)

    def clear_pending(self):
        self._pending.clear()

    def next_thumbnail(self):
        """Return the next worker callable; the window decides when to run it."""
        if self._videos is None or not self._pending:
            return None
        videos, frame = self._videos, self._pending.pop(0)
        size = self.iconSize()
        width, height = size.width(), size.height()
        def read():
            image = videos.read('original', frame)
            scale = min(width / image.shape[1], height / image.shape[0])
            image = cv2.resize(image, (max(1, round(image.shape[1] * scale)),
                                       max(1, round(image.shape[0] * scale))))
            return videos, frame, image
        return read

    def accept_thumbnail(self, result):
        videos, frame, image = result
        if videos is self._videos:
            self.set_image(frame, image)

    def select(self, frame):
        item = self.items.get(frame)
        if item is not None:
            self.setCurrentItem(item)

    def set_image(self, frame, image):
        if frame in self.items:
            self.images[frame] = image
            self._render(frame)

    def _render(self, frame):
        image = self.images.get(frame)
        pixmap = QPixmap(self.iconSize())
        pixmap.fill(Qt.GlobalColor.black)
        painter = QPainter(pixmap)
        if image is not None:
            if frame in self.masks:
                image = overlay_mask(image, self.masks[frame], .45, False, None, self.colors)
            displayed = image_from_bgr(image).scaled(self.iconSize(), Qt.AspectRatioMode.KeepAspectRatio,
                                                     Qt.TransformationMode.SmoothTransformation)
            painter.drawImage((pixmap.width() - displayed.width()) // 2,
                              (pixmap.height() - displayed.height()) // 2, displayed)
        status = "labeled" if frame in self.masks else "proposed"
        painter.setPen(QPen(QColor(KEY_FRAME_COLORS[status]), 3))
        painter.drawRect(pixmap.rect().adjusted(3, 3, -4, -4))
        if frame in self.crop:
            painter.setPen(QPen(QColor(KEY_FRAME_COLORS["boundary"]), 2))
            painter.drawRect(pixmap.rect().adjusted(1, 1, -2, -2))
        painter.end()
        self.items[frame].setIcon(QIcon(pixmap))
