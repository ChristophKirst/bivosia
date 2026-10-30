"""Shared VOS backend parameters for its editor and workflow settings."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLineEdit, QMessageBox, QPushButton, QVBoxLayout, QWidget

from bivosia.config import VOS_BACKENDS
from bivosia.ui.icons import action_icon
from bivosia.ui.model_install_dialog import ModelInstallDialog
from bivosia.ui.parameters import ParameterForm
from bivosia.ui.step_settings import step_store


class VosParameters(QWidget):
    changed = Signal()

    def __init__(self, dataset, remote, parent=None):
        super().__init__(parent)
        self.remote = remote
        self.store = None
        self._updating = False
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        self.model = QComboBox()
        for key, backend in VOS_BACKENDS.items():
            self.model.addItem(backend['label'], key)
        row = QHBoxLayout()
        row.addWidget(self.model, 1)
        install = QPushButton('Install')
        install.clicked.connect(self.install_model)
        row.addWidget(install)
        form.addRow('Model', row)
        self.checkpoint = QLineEdit()
        row = QHBoxLayout()
        row.addWidget(self.checkpoint, 1)
        self.browse = QPushButton('Open')
        self.browse.setIcon(action_icon('open'))
        self.browse.clicked.connect(self.choose_checkpoint)
        row.addWidget(self.browse)
        form.addRow('Checkpoint', row)
        self.device = QComboBox()
        self.device.setEditable(True)
        self.device.addItems(['auto', 'cuda', 'cpu', 'mps'])
        form.addRow('Device', self.device)
        self.direction = self._combo((('Both', 'both'), ('Forward', 'forward'), ('Reverse', 'reverse')))
        form.addRow('Direction', self.direction)
        self.references = self._combo((('Start key frame', 'start'), ('Both ends', 'ends'), ('All key frames', 'all')))
        self.references.setToolTip('Both ends / All key frames require consistent animal IDs across references')
        form.addRow('References', self.references)
        self.parameter_layout = QVBoxLayout()
        self.parameter_layout.setContentsMargins(0, 0, 0, 0)
        self.parameters = None
        form.addRow(self.parameter_layout)
        self.scale = self._combo((('Native', 1.), ('Half', .5), ('Quarter', .25)))
        form.addRow('Output size', self.scale)
        self.set_dataset(dataset)
        self.model.currentIndexChanged.connect(self._model_changed)
        for combo in (self.device, self.direction, self.references, self.scale):
            combo.currentTextChanged.connect(self._edited)
        self.checkpoint.editingFinished.connect(self._edited)

    @staticmethod
    def _combo(items):
        combo = QComboBox()
        for label, value in items:
            combo.addItem(label, value)
        return combo

    def set_dataset(self, dataset):
        if self.store is not None:
            self.store.changed.disconnect(self._shared_changed)
        self.store = step_store(dataset)
        self.set_values(self.store.get('vos'))
        self.store.changed.connect(self._shared_changed)

    @property
    def values(self):
        return dict(model=self.model.currentData(), checkpoint=self.checkpoint.text().strip(),
                    device=self.device.currentText(), direction=self.direction.currentData(),
                    references=self.references.currentData(), output_scale=self.scale.currentData(),
                    parameters=self.parameters.values)

    def options(self):
        values = self.values
        if not values['checkpoint']:
            values.pop('checkpoint')
        return values

    def set_values(self, values):
        self._updating = True
        self.model.setCurrentIndex(self.model.findData(values['model']))
        self.checkpoint.setText(values['checkpoint'])
        self.device.setCurrentText(values['device'])
        self.direction.setCurrentIndex(self.direction.findData(values['direction']))
        self.references.setCurrentIndex(self.references.findData(values['references']))
        self.scale.setCurrentIndex(self.scale.findData(values['output_scale']))
        self._backend(values['parameters'])
        self._updating = False
        self.target_changed()

    def _backend(self, values):
        backend = VOS_BACKENDS[self.model.currentData()]
        if self.parameters is not None:
            self.parameter_layout.removeWidget(self.parameters)
            self.parameters.deleteLater()
        self.parameters = ParameterForm(backend['options'], {**backend['defaults'], **values}, backend.get('help'))
        self.parameter_layout.addWidget(self.parameters)
        self.parameters.changed.connect(self._edited)
        for index in range(self.references.count()):
            self.references.model().item(index).setEnabled(self.references.itemData(index) in backend['reference_modes'])

    def _model_changed(self):
        if self._updating:
            return
        self._backend({})
        self.references.setCurrentIndex(0)
        self.target_changed()
        self._edited()

    def _edited(self, *args):
        if not self._updating:
            self.store.set('vos', self.values)
            self.changed.emit()

    def _shared_changed(self, operation, values):
        if operation == 'vos' and values != self.values:
            self.set_values(values)
            self.changed.emit()

    def target_changed(self):
        self.checkpoint.setPlaceholderText('Default checkpoint for the selected installation')
        try:
            remote = self.remote()
            self.browse.setEnabled(remote is None)
            if remote is None:
                self.checkpoint.setPlaceholderText(str(VOS_BACKENDS[self.model.currentData()]['checkpoint']))
        except ValueError:
            self.browse.setEnabled(False)

    def install_model(self):
        try:
            ModelInstallDialog(VOS_BACKENDS, self.model.currentData(), remote=self.remote(), parent=self).exec()
        except Exception as error:
            QMessageBox.warning(self, 'VOS', str(error))

    def choose_checkpoint(self):
        suffixes = VOS_BACKENDS[self.model.currentData()]['checkpoint_suffixes']
        path, _ = QFileDialog.getOpenFileName(self, 'Checkpoint', self.checkpoint.text(),
                                            'Weights (' + ' '.join('*' + suffix for suffix in suffixes) + ')')
        if path:
            self.checkpoint.setText(path)
            self._edited()
