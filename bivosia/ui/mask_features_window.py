"""Inspect Zernike-based UMAP embeddings of consensus masks.

Examples
--------
>>> from PySide6.QtWidgets import QApplication
>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.ui.window_manager import DatasetWindows
>>> app = QApplication.instance() or QApplication([])  # doctest: +SKIP
>>> windows = DatasetWindows(app)  # doctest: +SKIP
>>> window = windows.open('features', Dataset('/path/to/dataset'))  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from bivosia.dataset.mask_features import shape_analysis_inputs
from bivosia.ui.analysis_window import AnalysisWindow
from bivosia.ui.embedding_view import EmbeddingView


class MaskFeaturesWindow(AnalysisWindow):
    """Calculate consensus-mask descriptors and display their saved UMAP."""

    def __init__(self, dataset, windows):
        view = EmbeddingView()
        super().__init__(dataset, windows, 'Mask features', 'shape_features',
                         input_loader=shape_analysis_inputs, result_widget=view,
                         result_controls=view.controls,
                         required_positive=('radius',), processing=True)
        self.view, self.plot = view, view.plot
        view.failed.connect(self.statusBar().showMessage)
        self.parameters.changed.connect(self.refresh)

    def _ready(self, info):
        if self._closed or self._refresh_pending:
            return
        self.parameters.store.suggest(self.operation, degree=info['degree'], radius=info['radius'])
        super()._ready(info)
        self.view.set_data(info, self.dataset, self.parameters.values)

    def refresh(self):
        if hasattr(self, 'view'):
            self.view.clear_samples()
        super().refresh()

    def closeEvent(self, event):
        self.view.close()
        super().closeEvent(event)
