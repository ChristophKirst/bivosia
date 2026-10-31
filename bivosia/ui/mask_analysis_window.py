"""Choose where to calculate mask overlaps, assignments and quality scores.

Examples
--------
>>> from PySide6.QtWidgets import QApplication
>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.ui.window_manager import DatasetWindows
>>> app = QApplication.instance() or QApplication([])  # doctest: +SKIP
>>> windows = DatasetWindows(app)  # doctest: +SKIP
>>> window = windows.open('mask_analysis', Dataset('/path/to/dataset'))  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from bivosia.dataset.workflow.processing import processing_inputs
from bivosia.ui.analysis_window import AnalysisWindow


class MaskAnalysisWindow(AnalysisWindow):
    """Run mask quantification at the selected local or SSH dataset location."""

    def __init__(self, dataset, windows):
        super().__init__(dataset, windows, 'Mask analysis', 'quantify',
                         processing=True,
                         input_loader=lambda dataset, settings: processing_inputs(dataset, 'quantify', settings))
