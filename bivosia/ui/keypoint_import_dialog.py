"""Select the source image dimensions for importing keypoint tracks.

Examples
--------
>>> from PySide6.QtWidgets import QApplication
>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.ui.keypoint_import_dialog import KeypointImportDialog
>>> app = QApplication.instance() or QApplication([])  # doctest: +SKIP
>>> dialog = KeypointImportDialog(Dataset('/path/to/dataset'), '/path/to/tracks.h5')  # doctest: +SKIP
>>> dialog.exec()  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from functools import partial

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QLineEdit, QVBoxLayout

from bivosia.dataset.keypoints import keypoint_image_shape, select_keypoints
from bivosia.dataset.layout import Dataset
from bivosia.ui.parameters import ParameterForm
from bivosia.ui.task import Task


class KeypointImportDialog(QDialog):
    """Validate and copy a selected SLEAP file without blocking the GUI."""

    def __init__(self, dataset, path, parent=None):
        super().__init__(parent)
        self.dataset, self.path = dataset, path
        self.task = None
        self.setWindowTitle('Select keypoints')
        layout = QVBoxLayout(self)
        source = QLineEdit(str(path))
        source.setReadOnly(True)
        source.setToolTip(str(path))
        layout.addWidget(source)
        shape = keypoint_image_shape(dataset)
        self.parameters = ParameterForm((
            ('height', 'Source image height (px)', 0, 2147483647),
            ('width', 'Source image width (px)', 0, 2147483647),
        ), dict(zip(('height', 'width'), shape)), {
            'height': 'Height of the original images used for keypoint tracking.',
            'width': 'Width of the original images used for keypoint tracking.',
        })
        for control in self.parameters.controls.values():
            control.setSpecialValueText(' ')
        self.parameters.changed.connect(self._update_save)
        layout.addWidget(self.parameters)
        self.status = QLabel('Files outside the dataset keypoints folder are copied there. '
                             'The source file is preserved.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save |
                                        QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self._save)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._update_save()

    def _update_save(self):
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(
            all(value > 0 for value in self.parameters.values.values()))

    def _save(self):
        values = self.parameters.values
        shape = (values['height'], values['width'])
        self.error = None
        self.parameters.setEnabled(False)
        self.buttons.setEnabled(False)
        self.status.setText('Validating and copying keypoints…')
        operation = partial(select_keypoints, Dataset(self.dataset.directory), self.path, shape)
        self.task = Task(operation, self)
        self.task.connect_callbacks(failed=self._failed, finished=self._finished)
        self.task.start()

    def _failed(self, message):
        self.error = message

    def _finished(self):
        self.parameters.setEnabled(True)
        self.buttons.setEnabled(True)
        self.task = None
        if self.error is not None:
            self.status.setText(self.error)
        else:
            self.dataset.reload()
            self.accept()

    def reject(self):
        if self.task is None:
            super().reject()

    def closeEvent(self, event):
        if self.task is not None:
            event.ignore()
        else:
            super().closeEvent(event)
