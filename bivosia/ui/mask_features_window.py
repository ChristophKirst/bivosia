"""Inspect Zernike-based UMAP embeddings of exported consensus masks.

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

from bivosia.config import ANALYSIS_DEFAULTS
from bivosia.dataset.shape_features import shape_analysis_inputs
from bivosia.ui.analysis_window import AnalysisWindow
from bivosia.ui.embedding_plot import EmbeddingPlot


class MaskFeaturesWindow(AnalysisWindow):
    """Calculate consensus-mask descriptors and display their saved UMAP."""

    def __init__(self, dataset, windows):
        defaults = dict(ANALYSIS_DEFAULTS['shape_features'], **ANALYSIS_DEFAULTS['mask_preprocessing'], radius=0)
        options = (
            ('preprocess_masks', 'Preprocess masks', 0, 1),
            ('filter_by_area', 'Filter by area', 0, 1),
            ('main_body_area', 'Main body area (pixels²)', 0, 2147483647),
            ('filter_by_distance', 'Filter by distance', 0, 1),
            ('fragment_distance', 'Fragment distance (pixels)', 0, 2147483647),
            ('apply_opening', 'Morphological opening', 0, 1),
            ('opening_radius', 'Opening radius (pixels)', 0, 2147483647),
            ('degree', 'Zernike degree', 0, 2147483647),
            ('radius', 'Zernike radius (pixels)', 0, 2147483647),
            ('z_score', 'Z-score features before UMAP', 0, 1),
        )
        plot = EmbeddingPlot()
        super().__init__(dataset, windows, 'Mask features', 'shape_features', options, defaults,
                         input_loader=shape_analysis_inputs, result_widget=plot,
                         required_positive=('radius',), help_text={
            'preprocess_masks': 'Select body components and remove thin protrusions before shape analysis. Originals are unchanged.',
            'filter_by_area': 'Keep only the largest component if its area exceeds Main body area; otherwise leave components unchanged.',
            'main_body_area': 'Keep only the largest component when its pixel area exceeds this threshold.',
            'filter_by_distance': 'Keep at most the two largest remaining components when nearby; otherwise keep only the largest.',
            'fragment_distance': 'Keep the two largest remaining components when their centroids are within this distance.',
            'apply_opening': 'Remove thin protrusions using erosion followed by dilation with a disk.',
            'opening_radius': 'Disk radius for morphological opening at the input mask resolution. Zero skips opening.',
            'degree': 'Maximum degree of the absolute Zernike moments. Mahotas defaults to 8.',
            'radius': 'Fixed disk radius around each mask centroid. It must contain every foreground pixel.',
            'z_score': 'Center and scale each feature column across all samples before standard UMAP.',
        })
        self.plot = plot
        for name in ('degree', 'radius'):
            self.parameters.controls[name].valueChanged.connect(self.refresh)
        for name, value in ANALYSIS_DEFAULTS['mask_preprocessing'].items():
            widget = self.parameters.controls[name]
            signal = widget.toggled if isinstance(value, bool) else widget.valueChanged
            signal.connect(self._preprocessing_changed)
        self.parameters.controls['z_score'].toggled.connect(self.refresh)
        self.parameters.controls['preprocess_masks'].toggled.connect(self._preprocessing_changed)
        self._enable_preprocessing()

    def _enable_preprocessing(self):
        enabled = self.parameters.controls['preprocess_masks'].isChecked()
        for name in ANALYSIS_DEFAULTS['mask_preprocessing']:
            self.parameters.controls[name].setEnabled(enabled)
        for step, parameter in (('filter_by_area', 'main_body_area'),
                                ('filter_by_distance', 'fragment_distance'),
                                ('apply_opening', 'opening_radius')):
            self.parameters.controls[parameter].setEnabled(enabled and self.parameters.controls[step].isChecked())

    def _preprocessing_changed(self):
        self._enable_preprocessing()
        self.refresh()

    def _ready(self, info):
        if self._closed:
            return
        for name in ('degree', 'radius'):
            widget = self.parameters.controls[name]
            widget.blockSignals(True)
            widget.setValue(info[name])
            widget.blockSignals(False)
        self.plot.set_points(info['coordinates'])
        super()._ready(info)

    def refresh(self):
        if hasattr(self, 'plot'):
            self.plot.set_points(None)
        super().refresh()
