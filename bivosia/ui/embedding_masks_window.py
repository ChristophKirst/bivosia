"""Display representative mask silhouettes on a grid of a visible 2D embedding."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtWidgets import QMainWindow

from bivosia.ui.embedding_2d_plot import Embedding2DPlot
from bivosia.ui.mask_thumbnails import mask_images
from bivosia.ui.task import Task


def grid_samples(points, low, high, columns, rows):
    """Choose the nearest-to-center sample inside each occupied display cell.

    >>> import numpy as np
    >>> from bivosia.ui.embedding_masks_window import grid_samples
    >>> points = np.array([[.1, .1], [.24, .24], [.8, .8], [2., 2.]])
    >>> indices, centers = grid_samples(points, np.zeros(2), np.ones(2), 2, 2)
    >>> indices.tolist()
    [1, 2]
    >>> centers.tolist()
    [[0.25, 0.25], [0.75, 0.75]]
    """
    visible = np.flatnonzero(np.all((points >= low) & (points <= high), axis=1))
    cells = np.array((columns, rows))
    scaled = (points[visible] - low) / (high - low) * cells
    xy = np.clip(np.floor(scaled).astype(int), 0, cells - 1)
    keys = xy[:, 1] * columns + xy[:, 0]
    distance = ((scaled - xy - .5) ** 2).sum(axis=1)
    order = np.lexsort((visible, distance, keys))
    _, first = np.unique(keys[order], return_index=True)
    chosen = order[first]
    return visible[chosen], low + (xy[chosen] + .5) / cells * (high - low)


class _MaskGridPlot(Embedding2DPlot):
    def __init__(self, cell_size):
        super().__init__()
        self.images = {}
        self.cell_size = cell_size

    def _pixel_size(self):
        box = self.plot_rect()
        return self.cell_size / (self.bounds[1] - self.bounds[0]) * (box.width(), box.height()) * .8

    def _draw_points(self, painter, box):
        self._positions = self.screen_positions()
        size = self._pixel_size()
        painter.setRenderHint(painter.RenderHint.SmoothPixmapTransform)
        for index, image in self.images.items():
            scale = min(size[0] / image.width(), size[1] / image.height())
            width, height = image.width() * scale, image.height() * scale
            x, y = self._positions[index]
            painter.drawImage(QRectF(x - width / 2, y - height / 2, width, height), image)

    def pick_radius(self):
        return float(min(self._pixel_size()) / 2)


class EmbeddingMasksWindow(QMainWindow):
    """Load only the grid's representative masks, without changing the source plot."""

    sampleSelected = Signal(int, int)

    def __init__(self, dataset, info, settings, plot, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._closed = False
        self.setWindowTitle(f'UMAP masks · {dataset.directory.name}')
        low, high = (value.copy() for value in plot.bounds)
        box = plot.plot_rect()
        columns, rows = max(1, int(box.width() // 64)), max(1, int(box.height() // 64))
        indices, centers = grid_samples(plot.points, low, high, columns, rows)
        self.frames, self.identities = info['frames'][indices], info['identities'][indices]
        colors = plot.colors[indices].copy()
        self.plot = _MaskGridPlot((high - low) / (columns, rows))
        self.plot.set_points(centers, colors)
        self.plot.set_view(low, high)
        self.plot.pointSelected.connect(self._selected)
        self.setCentralWidget(self.plot)
        self.resize(plot.width(), plot.height() + self.statusBar().sizeHint().height())
        self.statusBar().showMessage('Loading masks…')
        def load():
            return mask_images(dataset, self.frames, self.identities, colors, settings,
                               info['source'], self.task.isInterruptionRequested)
        self.task = Task(load, self)
        self.task.connect_callbacks(ready=self._ready, failed=self.statusBar().showMessage, finished=self._finished)
        self.task.start()

    def _selected(self, index):
        self.plot.set_selected([index])
        self.sampleSelected.emit(int(self.frames[index]), int(self.identities[index]))

    def _ready(self, images):
        if not self._closed:
            self.plot.images = images
            self.plot.update()
            self.statusBar().showMessage(f'{len(images)} masks')

    def _finished(self):
        self.task = None

    def closeEvent(self, event):
        self._closed = True
        if self.task is not None:
            self.task.requestInterruption()
        event.accept()
