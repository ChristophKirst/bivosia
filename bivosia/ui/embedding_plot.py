"""Selectable Qt scatter displays for saved two- and three-dimensional embeddings."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QStackedWidget

from bivosia.ui.embedding_colors import embedding_colors
from bivosia.ui.embedding_2d_plot import Embedding2DPlot


class EmbeddingPlot(QStackedWidget):
    """Keep a common sample-selection interface for 2D and native Qt 3D plots."""

    pointSelected = Signal(int)

    def __init__(self):
        super().__init__()
        self.points = None
        self.colors = np.empty((0, 4), dtype=np.uint8)
        self.selected = ()
        self.two = Embedding2DPlot()
        self.two.pointSelected.connect(self.pointSelected)
        self.three = None
        self.addWidget(self.two)

    def set_points(self, points, identities=None, mode='sample'):
        self.points = points
        self.selected = ()
        if points is None or not len(points):
            self.colors = np.empty((0, 4), dtype=np.uint8)
            self.two.set_points(None, self.colors)
            if self.three is not None:
                self.three.set_points(None, None, None, mode)
            self.setCurrentWidget(self.two)
            return
        identities = np.ones(len(points), dtype=int) if identities is None else identities
        self.colors = embedding_colors(points, identities, mode)
        if points.shape[1] == 3:
            if self.three is None:
                from bivosia.ui.embedding_3d_plot import Embedding3DPlot
                self.three = Embedding3DPlot()
                self.three.pointSelected.connect(self.pointSelected)
                self.addWidget(self.three)
            self.three.set_points(points, identities, self.colors, mode)
            self.setCurrentWidget(self.three)
        else:
            self.two.set_points(points, self.colors)
            self.setCurrentWidget(self.two)

    def set_selected(self, indices):
        self.selected = tuple(indices)
        self.currentWidget().set_selected(np.asarray(indices, dtype=int))
