"""Shared local or SSH execution choice for dataset processing windows."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QFormLayout, QVBoxLayout, QWidget

from bivosia.ui.remote_target import RemoteTarget


class ProcessingTarget(QWidget):
    changed = Signal()

    def __init__(self, dataset, parent=None, *, require_dataset=True):
        super().__init__(parent)
        self.require_dataset = require_dataset
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        self.target = QComboBox()
        self.target.addItems(('Local', 'SSH'))
        form.addRow('Processing', self.target)
        layout.addLayout(form)
        self.remote_panel = RemoteTarget(self, include_dataset=require_dataset)
        self.remote_panel.set_dataset(dataset.directory)
        layout.addWidget(self.remote_panel)
        self.remote_panel.hide()
        self.target.currentIndexChanged.connect(self._changed)
        self.remote_panel.changed.connect(self.changed)

    def _changed(self):
        self.remote_panel.setVisible(self.target.currentText() == 'SSH')
        self.changed.emit()

    def remote(self):
        return self.remote_panel.remote(require_dataset=self.require_dataset) if self.target.currentText() == 'SSH' else None
