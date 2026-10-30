"""Qt scatter view with sample selection and frame highlights."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QVector3D
from PySide6.QtWidgets import QVBoxLayout, QWidget
from PySide6.QtDataVisualization import (
    Q3DCamera, Q3DScatter, Q3DTheme, QAbstract3DGraph, QAbstract3DSeries,
    QScatter3DSeries, QScatterDataItem,
)

from bivosia.ui.embedding_colors import embedding_bounds, sample_gradient


class Embedding3DPlot(QWidget):
    pointSelected = Signal(int)

    def __init__(self):
        super().__init__()
        self.graph = Q3DScatter()
        self.graph.setShadowQuality(QAbstract3DGraph.ShadowQuality.ShadowQualityNone)
        self.graph.setOptimizationHints(QAbstract3DGraph.OptimizationHint.OptimizationStatic)
        self.graph.setSelectionMode(QAbstract3DGraph.SelectionFlag.SelectionItem)
        theme = self.graph.activeTheme()
        theme.setWindowColor(self.palette().base().color())
        theme.setBackgroundEnabled(False)
        theme.setLabelBackgroundEnabled(False)
        theme.setLabelTextColor(self.palette().text().color())
        for index, axis in enumerate((self.graph.axisX(), self.graph.axisY(), self.graph.axisZ()), 1):
            axis.setTitle(f'UMAP {index}')
            axis.setTitleVisible(True)
            axis.setLabelFormat('%.2g')
        self.graph.scene().activeCamera().setCameraPreset(Q3DCamera.CameraPreset.CameraPresetIsometricRight)
        self.container = QWidget.createWindowContainer(self.graph, self)
        self.container.setToolTip('Left-click: select sample · right-drag: rotate · wheel: zoom')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.container)
        self.points = None
        self.series = []
        self.highlight = None
        self._generation = 0

    def _series(self, points, indices, color):
        series = QScatter3DSeries()
        series.setMesh(QAbstract3DSeries.Mesh.MeshSphere)
        series.setItemSize(0.)
        series.setBaseColor(color)
        series.setItemLabelVisible(False)
        generation = self._generation
        series.selectedItemChanged.connect(lambda index: self.pointSelected.emit(int(indices[index]))
            if generation == self._generation and 0 <= index < len(indices) else None,
            Qt.ConnectionType.QueuedConnection)
        series.dataProxy().resetArray([QScatterDataItem(QVector3D(*map(float, point))) for point in points])
        self.graph.addSeries(series)
        return series

    def set_points(self, points, identities, colors, mode):
        self._generation += 1
        self.graph.clearSelection()
        for series in self.graph.seriesList():
            self.graph.removeSeries(series)
            series.deleteLater()
        self.series, self.highlight = [], None
        self.points = points
        if points is None or not len(points):
            return
        low, high = embedding_bounds(points)
        self.set_bounds(low, high)
        groups = [np.arange(len(points))] if mode == 'sample' else [np.flatnonzero(identities == value)
                                                                  for value in np.unique(identities)]
        for indices in groups:
            series = self._series(points[indices], indices, QColor(*map(int, colors[indices[0]])))
            if mode == 'sample':
                series.setBaseGradient(sample_gradient())
                series.setColorStyle(Q3DTheme.ColorStyle.ColorStyleRangeGradient)
            self.series.append(series)

    def set_bounds(self, low, high):
        """Use the same coordinate bounds for scatter and mask displays."""
        for axis, start, end in zip((self.graph.axisX(), self.graph.axisY(), self.graph.axisZ()), low, high):
            axis.setRange(float(start), float(end))

    def set_selected(self, indices):
        self.graph.clearSelection()
        if self.highlight is not None:
            self.graph.removeSeries(self.highlight)
            self.highlight.deleteLater()
            self.highlight = None
        if self.points is not None and len(indices):
            self.highlight = self._series(self.points[indices], indices, self.palette().text().color())
            self.highlight.setMesh(QAbstract3DSeries.Mesh.MeshSphere)
            self.highlight.setItemSize(.13)
