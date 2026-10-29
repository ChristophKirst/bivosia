"""Configure keypoint alignment to a dataset's annotation-selected masks.

Examples
--------
>>> from PySide6.QtWidgets import QApplication
>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.ui.window_manager import DatasetWindows
>>> app = QApplication.instance() or QApplication([])  # doctest: +SKIP
>>> windows = DatasetWindows(app)  # doctest: +SKIP
>>> window = windows.open('alignment', Dataset('/path/to/dataset'))  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from bivosia.config import ANALYSIS_DEFAULTS, PROCESSING_DEFAULTS
from bivosia.ui.analysis_window import AnalysisWindow


class KeypointAlignmentWindow(AnalysisWindow):
    """Select an annotation, keypoints and the existing tie-handling option."""

    def __init__(self, dataset, windows):
        defaults = dict(ANALYSIS_DEFAULTS['alignment'], n_animals=0,
                        workers=PROCESSING_DEFAULTS['alignment']['workers'] or 0)
        options = (
            ('n_animals', 'Animals', 0, 2147483647),
            ('workers', 'Workers', 0, 2147483647),
            ('fallback_previous', 'Reuse previous assignments for ties', 0, 1),
        )
        super().__init__(dataset, windows, 'Keypoint alignment', 'align', options, defaults,
                         keypoints=True)
