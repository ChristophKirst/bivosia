"""Saved annotation intervals and frame-indexed embedding samples."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QImage, QPen

from bivosia.ui.plot_models import RowSpec
from bivosia.ui.revision_plots import annotation_row
from bivosia.ui.zone_plot import ZoneTimeline


class EmbeddingTimeline(ZoneTimeline):
    """Rasterize sample dots once per view, keeping selection separate from the cache."""

    pointSelected = Signal(int)

    def __init__(self):
        self.frames = self.identities = np.empty(0, dtype=int)
        self.colors = np.empty((0, 4), dtype=np.uint8)
        self.selected = ()
        self.animals = 1
        super().__init__(1, title='Annotations')

    def set_data(self, frame_count, zones, frames, identities, colors):
        changed = self.frame_count != frame_count
        self.frame_count = frame_count
        self.frames, self.identities, self.colors = frames, identities, colors
        self.animals = max(1, int(identities.max())) if len(identities) else 1
        self.selected = ()
        self.set_rows((annotation_row(zones), RowSpec('umap', 'UMAP', height=2)))
        if changed:
            self.reset_zoom()

    def set_selected(self, indices):
        self.selected = tuple(indices)
        self.update()

    def _sample_positions(self, indices=None):
        row = next((rect for row, rect in self.row_rects() if row.key == 'umap'), None)
        if row is None or not len(self.frames):
            return np.empty(0, dtype=int), np.empty((0, 2))
        start, end = self.display_range()
        if indices is None:
            first = np.searchsorted(self.frames, start, side='left')
            last = np.searchsorted(self.frames, end, side='right')
            indices = np.arange(first, last)
        else:
            indices = np.asarray(indices, dtype=int)
            indices = indices[(self.frames[indices] >= start) & (self.frames[indices] <= end)]
        x = self.frame_to_x(self.frames[indices], self._plot_rect(), start, end)
        y = row.top() + (self.identities[indices] - .5) / self.animals * row.height()
        return indices, np.column_stack((x, y))

    def _paint_static(self, painter):
        super()._paint_static(painter)
        indices, positions = self._sample_positions()
        if not len(indices):
            return
        pixels = np.zeros((self.height(), self.width(), 4), dtype=np.uint8)
        x, y = np.rint(positions).astype(int).T
        for dy, dx in ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)):
            pixels[np.clip(y + dy, 0, self.height() - 1),
                   np.clip(x + dx, 0, self.width() - 1)] = self.colors[indices]
        image = QImage(pixels.data, self.width(), self.height(), pixels.strides[0],
                       QImage.Format.Format_RGBA8888)
        painter.drawImage(0, 0, image)

    def _paint_overlay(self, painter):
        super()._paint_overlay(painter)
        indices, positions = self._sample_positions(self.selected)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(Qt.GlobalColor.white, 2))
        for position in positions:
            painter.drawEllipse(QPointF(*position), 5, 5)

    def _click(self, event):
        indices, positions = self._sample_positions()
        if len(indices):
            distance = ((positions - (event.position().x(), event.position().y())) ** 2).sum(axis=1)
            nearest = int(np.argmin(distance))
            if distance[nearest] <= 36:
                self.pointSelected.emit(int(indices[nearest]))
                return
        super()._click(event)
