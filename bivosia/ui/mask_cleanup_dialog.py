"""Choose contour filtering and overlap-removal parameters for mask cleanup.

Examples
--------
>>> from PySide6.QtWidgets import QApplication
>>> from bivosia.config import ANALYSIS_DEFAULTS
>>> from bivosia.ui.mask_cleanup_dialog import MaskCleanupDialog
>>> app = QApplication.instance() or QApplication([])  # doctest: +SKIP
>>> dialog = MaskCleanupDialog(ANALYSIS_DEFAULTS['masks']['cleanup'])  # doctest: +SKIP
>>> dialog.exec()  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout

from bivosia.ui.parameters import ParameterForm


class MaskCleanupDialog(QDialog):
    """Edit cleanup settings without changing the current mask or preview."""

    def __init__(self, parameters, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Mask cleanup settings')
        layout = QVBoxLayout(self)
        self.parameters = ParameterForm((
            ('contour_min_area', 'Minimum contour area (px²)', 0, 2147483647),
            ('contour_max_area', 'Maximum contour area (px²)', 0, 2147483647),
            ('overlap_kernel', 'Overlap kernel (px)', 1, 2147483647),
        ), parameters, {
            'contour_min_area': 'Inclusive minimum contour area in full-resolution pixels squared.',
            'contour_max_area': 'Inclusive maximum contour area in full-resolution pixels squared.',
            'overlap_kernel': 'Positive odd side length of the square used to exclude overlapping masks.',
        })
        self.parameters.controls['overlap_kernel'].setSingleStep(2)
        layout.addWidget(self.parameters)
        self.status = QLabel()
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accept(self):
        values = self.parameters.values
        if values['contour_min_area'] > values['contour_max_area']:
            self.status.setText('Minimum contour area must not exceed maximum contour area.')
        elif values['overlap_kernel'] % 2 == 0:
            self.status.setText('Overlap kernel must be a positive odd integer.')
        else:
            super().accept()
