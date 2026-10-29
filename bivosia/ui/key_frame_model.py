"""Select a key-frame segmentation model in a local or SSH installation.

Examples
--------
>>> from PySide6.QtWidgets import QApplication
>>> from bivosia.ui.key_frame_model import KeyFrameModelSelector
>>> app = QApplication.instance() or QApplication([])  # doctest: +SKIP
>>> selector = KeyFrameModelSelector(model="sam2")  # doctest: +SKIP
>>> selector.show()  # doctest: +SKIP
>>> app.exec()  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication, QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLineEdit, QPushButton, QWidget

from bivosia.config import KEY_FRAME_BACKENDS
from bivosia.processing.remote_images import check_models
from bivosia.processing.remote_session import RemoteSession
from bivosia.segmentation.image import model_availability
from bivosia.ui.icons import action_icon
from bivosia.ui.model_install_dialog import ModelInstallDialog
from bivosia.ui.remote_target import RemoteTarget
from bivosia.ui.task import Task


class ModelCombo(QComboBox):
    opening = Signal()

    def showPopup(self):
        self.opening.emit()
        super().showPopup()


class KeyFrameModelSelector(QWidget):
    changed = Signal()

    def __init__(self, model=None, checkpoint=None, device="auto", parameters=None):
        super().__init__()
        model = model or next(iter(KEY_FRAME_BACKENDS))
        self.device = device
        self.overrides = {model: parameters or {}}
        self.checkpoints = {name: spec['checkpoint'] for name, spec in KEY_FRAME_BACKENDS.items()}
        if checkpoint is not None:
            self.checkpoints[model] = Path(checkpoint).expanduser()
        self._checkpoint_sets = {None: self.checkpoints}
        self._probe = None
        self._generation = 0
        self._busy = self._target_pending = False
        self.availability = {}
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        self.target = QComboBox()
        self.target.addItems(['Local', 'SSH'])
        form.addRow('Run on', self.target)
        self.remote_panel = RemoteTarget(include_dataset=False)
        self.remote_panel.hide()
        form.addRow(self.remote_panel)
        self.model = ModelCombo()
        for name in self.checkpoints:
            self.model.addItem(KEY_FRAME_BACKENDS[name]['label'], name)
        self.model.setCurrentIndex(self.model.findData(model))
        model_row = QHBoxLayout()
        model_row.addWidget(self.model, 1)
        install = QPushButton('Install')
        install.clicked.connect(self.install_model)
        model_row.addWidget(install)
        form.addRow("Model", model_row)
        self.checkpoint = QLineEdit()
        self.browse = QPushButton("Open")
        self.browse.setIcon(action_icon('open'))
        self.browse.clicked.connect(self.choose_checkpoint)
        row = QHBoxLayout()
        row.addWidget(self.checkpoint, 1)
        row.addWidget(self.browse)
        form.addRow("Checkpoint", row)
        self.model.currentIndexChanged.connect(self._selected)
        self.model.opening.connect(self.refresh)
        self.checkpoint.editingFinished.connect(self._checkpoint_changed)
        self.target.currentIndexChanged.connect(self._target_changed)
        self.remote_panel.changed.connect(self._target_changed)
        QApplication.instance().aboutToQuit.connect(self.shutdown)
        self.refresh()

    @property
    def remote(self):
        return self.remote_panel.remote(require_dataset=False) if self.target.currentText() == 'SSH' else None

    def set_busy(self, busy):
        self._busy = busy
        self.setEnabled(not busy)
        if not busy and self._target_pending:
            self._target_changed()

    def _target_changed(self):
        if self._busy:
            self._target_pending = True
            return
        self._target_pending = False
        ssh = self.target.currentText() == 'SSH'
        self.remote_panel.setVisible(ssh)
        self.browse.setEnabled(not ssh)
        profile = self.remote_panel.profile if ssh else None
        key = tuple(profile[field] for field in ('host', 'python', 'code')) if profile else ('SSH',) if ssh else None
        if key not in self._checkpoint_sets:
            self._checkpoint_sets[key] = {name: profile['checkpoints'].get(name, '') if profile else ''
                                          for name in KEY_FRAME_BACKENDS}
        self.checkpoints = self._checkpoint_sets[key]
        self.refresh()

    @property
    def reason(self):
        return self.availability[self.model.currentData()][1]

    @property
    def parameters(self):
        """Return the selected (model, checkpoint, device), or None if unavailable."""
        name = self.model.currentData()
        device, reason = self.availability[name]
        return None if reason else (name, self.checkpoints[name], device)

    def refresh(self):
        self.shutdown()
        try:
            remote = self.remote
        except ValueError as error:
            self._apply_availability({name: (None, str(error)) for name in self.checkpoints})
            return
        if remote is None:
            self._apply_availability({name: model_availability(name, path, self.device, self.overrides.get(name))
                                      for name, path in self.checkpoints.items()})
            return
        self._apply_availability({name: (None, 'Checking remote models…') for name in self.checkpoints})
        session = RemoteSession(remote, 'image_session')
        checkpoints = self.checkpoints.copy()
        task = Task(lambda: check_models(session, checkpoints, self.device, self.overrides), self)
        generation = self._generation
        self._probe = task, session
        task.ready.connect(lambda result: self._checked(generation, result))
        task.failed.connect(lambda error: self._checked(generation, {name: (None, error) for name in checkpoints}))
        task.finished.connect(lambda: self._check_finished(generation))
        task.start()

    def _checked(self, generation, availability):
        if generation == self._generation:
            self._apply_availability(availability)

    def _check_finished(self, generation):
        if generation == self._generation and self._probe is not None:
            self._probe[0].deleteLater()
            self._probe = None

    def shutdown(self):
        """Cancel pending availability checks before replacing or closing this selector."""
        self._generation += 1
        if self._probe is not None:
            task, session = self._probe
            task.requestInterruption()
            session.close()
            task.wait()
            task.deleteLater()
            self._probe = None

    def _apply_availability(self, availability):
        self.availability = availability
        for index in range(self.model.count()):
            name = self.model.itemData(index)
            device, reason = self.availability[name]
            item = self.model.model().item(index)
            item.setEnabled(not reason)
            item.setToolTip(reason or f"Available · {device.upper()}")
        if self.reason:
            available = next((name for name, (_, reason) in self.availability.items() if not reason), None)
            if available is not None:
                self.model.blockSignals(True)
                self.model.setCurrentIndex(self.model.findData(available))
                self.model.blockSignals(False)
        self._selected()

    def _selected(self):
        name = self.model.currentData()
        self.checkpoint.setText(str(self.checkpoints[name]))
        self.model.setToolTip("\n".join(f"{KEY_FRAME_BACKENDS[key]['label']}: {reason or 'Available · ' + device.upper()}"
                                       for key, (device, reason) in self.availability.items()))
        self.changed.emit()

    def _checkpoint_changed(self):
        path = self.checkpoint.text().strip()
        self.checkpoints[self.model.currentData()] = (Path(path).expanduser() if self.target.currentText() == 'Local'
                                                     else path)
        self.refresh()

    def install_model(self):
        try:
            remote = self.remote
        except ValueError:
            self.refresh()
            return
        self.shutdown()
        dialog = ModelInstallDialog(KEY_FRAME_BACKENDS, self.model.currentData(), remote=remote, parent=self)
        dialog.exec()
        self.refresh()

    def choose_checkpoint(self):
        filters = {spec['label'] + ' checkpoint (' + ' '.join('*' + suffix for suffix in spec['checkpoint_suffixes']) + ')': name
                   for name, spec in KEY_FRAME_BACKENDS.items()}
        selected = next(label for label, name in filters.items() if name == self.model.currentData())
        path, chosen = QFileDialog.getOpenFileName(self, "Checkpoint", self.checkpoint.text(),
                                                 ";;".join(filters), selected)
        if path:
            name = filters[chosen]
            self.checkpoints[name] = Path(path)
            self.model.blockSignals(True)
            self.model.setCurrentIndex(self.model.findData(name))
            self.model.blockSignals(False)
            self.refresh()
