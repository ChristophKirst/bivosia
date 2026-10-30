"""Select a segmentation model and display its installation progress.

Examples
--------
>>> from PySide6.QtWidgets import QApplication
>>> from bivosia.config import KEY_FRAME_BACKENDS
>>> from bivosia.ui.model_install_dialog import ModelInstallDialog
>>> app = QApplication.instance() or QApplication([])  # doctest: +SKIP
>>> dialog = ModelInstallDialog(KEY_FRAME_BACKENDS, 'sam2')  # doctest: +SKIP
>>> dialog.exec()  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import sys

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QComboBox, QDialog, QFormLayout, QHBoxLayout,
                               QLabel, QPlainTextEdit, QPushButton, QVBoxLayout)

from bivosia.config import MODEL_INSTALLATIONS
from bivosia.processing.model_install import install_local
from bivosia.ui.task import Task


class ModelInstallDialog(QDialog):
    """Run the same installer locally or through an existing SSH installation."""

    message = Signal(str)
    installed = Signal(dict)

    def __init__(self, models, model, *, remote=None, parent=None):
        super().__init__(parent)
        self.remote, self.task = remote, None
        self.setWindowTitle('Install model')
        self.resize(620, 400)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.addRow('Run on', QLabel(remote.host if remote else 'Local'))
        form.addRow('Python', QLabel(remote.python if remote else sys.executable))
        self.model = QComboBox()
        for name, spec in models.items():
            if name in MODEL_INSTALLATIONS:
                self.model.addItem(spec['label'], name)
        self.model.setCurrentIndex(max(0, self.model.findData(model)))
        form.addRow('Model', self.model)
        layout.addLayout(form)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        layout.addWidget(self.log)
        self.message.connect(self.log.appendPlainText)
        self.status = QLabel('Source and weights are stored in external/. Packages use the Python environment above.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        buttons.addStretch()
        self.install_button = QPushButton('Install')
        self.install_button.clicked.connect(self.start)
        buttons.addWidget(self.install_button)
        self.close_button = QPushButton('Close')
        self.close_button.clicked.connect(self.reject)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)

    def start(self):
        if self.task is not None:
            return
        model = self.model.currentData()
        self.log.clear()
        self.status.setText('Installing…')
        self.model.setEnabled(False)
        self.install_button.setEnabled(False)
        self.close_button.setEnabled(False)
        install = self.remote.install_model if self.remote else install_local
        self.task = Task(lambda: install(model, self.message.emit), self)
        self.task.connect_callbacks(ready=self._ready, failed=self.status.setText, finished=self._finished)
        self.task.start()

    def _ready(self, result):
        self.status.setText('Installed.' if self.remote else
                           'Installed. Restart BiVOSIA to load newly installed packages.')
        self.installed.emit(result)

    def _finished(self):
        self.task = None
        self.model.setEnabled(True)
        self.install_button.setEnabled(True)
        self.close_button.setEnabled(True)

    def reject(self):
        if self.task is None:
            super().reject()

    def closeEvent(self, event):
        if self.task is not None:
            event.ignore()
        else:
            super().closeEvent(event)
