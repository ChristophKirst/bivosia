"""Manage and validate saved remote BiVOSIA installations.

Examples
--------
>>> from PySide6.QtWidgets import QApplication
>>> from bivosia.ui.remote_installations_dialog import RemoteInstallationsDialog
>>> app = QApplication.instance() or QApplication([])  # doctest: +SKIP
>>> dialog = RemoteInstallationsDialog()  # doctest: +SKIP
>>> dialog.exec()  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget)

from bivosia.processing.remote.client import Remote
from bivosia.dataset.storage.locations import remote_path
from bivosia.ui.remote_settings import installation_label, remote_settings
from bivosia.ui.task import Task


class RemoteInstallationsDialog(QDialog):
    """Save installations only after the chosen SSH interpreter verifies them."""

    def __init__(self, parent=None, store=None):
        super().__init__(parent)
        self.store = store if store is not None else remote_settings()
        self.installation_id = self.selected_id = self.task = None
        self.setWindowTitle('Remote installations')
        self.resize(640, 420)
        layout = QVBoxLayout(self)
        self.entries = QListWidget()
        self.entries.currentItemChanged.connect(self._select)
        layout.addWidget(self.entries)
        self.form = QWidget()
        form = QFormLayout(self.form)
        self.fields = {}
        for key, label in (('host', 'Host'), ('python', 'Python'), ('code', 'Code'),
                           ('data_folder', 'Default data folder')):
            field = QLineEdit()
            self.fields[key] = field
            form.addRow(label, field)
        self.fields['host'].setToolTip('SSH alias or user@host; use your SSH configuration for keys and ports')
        self.fields['code'].setToolTip('Remote inner bivosia package folder containing config.py and processing/')
        self.fields['data_folder'].setToolTip('Optional absolute folder on the workstation; datasets get their own subfolder')
        layout.addWidget(self.form)
        self.status = QLabel('Install BiVOSIA on the server, then enter its host, Python executable and code folder. '
                             'Save checks the installation; a dataset and model weights are not required.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        self.new_button = QPushButton('New')
        self.new_button.clicked.connect(self._new)
        self.save_button = QPushButton('Save')
        self.save_button.clicked.connect(self._save)
        self.remove_button = QPushButton('Remove')
        self.remove_button.clicked.connect(self._remove)
        self.close_button = QPushButton('Close')
        self.close_button.clicked.connect(self.reject)
        for button in (self.new_button, self.save_button, self.remove_button, self.close_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self._reload()
        self._new()

    def _reload(self):
        self.entries.blockSignals(True)
        self.entries.clear()
        for key, profile in self.store.installations.items():
            item = QListWidgetItem(installation_label(profile), self.entries)
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setToolTip(item.text())
            if key == self.installation_id:
                self.entries.setCurrentItem(item)
        self.entries.blockSignals(False)

    def _select(self, item, previous=None):
        if item is None:
            return
        self.installation_id = item.data(Qt.ItemDataRole.UserRole)
        profile = self.store.installations[self.installation_id]
        for key, field in self.fields.items():
            field.setText(profile.get(key, ''))
        self.remove_button.setEnabled(True)

    def _new(self):
        self.entries.setCurrentRow(-1)
        self.installation_id = None
        for key, field in self.fields.items():
            field.setText('python' if key == 'python' else '')
        self.remove_button.setEnabled(False)

    def _save(self):
        if self.task is not None:
            return
        try:
            self._data_folder = self.fields['data_folder'].text().strip()
            if self._data_folder:
                self._data_folder = remote_path(self._data_folder)
            remote = Remote(**{key: self.fields[key].text().strip() for key in ('host', 'python', 'code')})
        except ValueError as error:
            self.status.setText(str(error))
            return
        self._candidate = remote
        self._busy(True)
        self.status.setText('Checking installation…')
        self.task = Task(remote.check_installation, self)
        self.task.connect_callbacks(ready=self._saved, failed=self.status.setText, finished=self._finished)
        self.task.start()

    def _saved(self, details):
        try:
            self.installation_id = self.store.save(self._candidate.host, details, self.installation_id,
                                                   data_folder=self._data_folder)
            self.selected_id = self.installation_id
            self._reload()
            self._select(self.entries.currentItem())
            self.status.setText('Installation checked and saved.')
        except Exception as error:
            self.status.setText(str(error))

    def _remove(self):
        try:
            self.store.remove(self.installation_id)
            if self.selected_id == self.installation_id:
                self.selected_id = None
            self._new()
            self._reload()
            self.status.setText('Removed from the saved list.')
        except Exception as error:
            self.status.setText(str(error))

    def _busy(self, busy):
        for widget in (self.entries, self.form, self.new_button, self.save_button, self.close_button):
            widget.setEnabled(not busy)
        self.remove_button.setEnabled(not busy and self.installation_id is not None)

    def _finished(self):
        self.task = None
        self._busy(False)

    def reject(self):
        if self.task is None:
            super().reject()

    def closeEvent(self, event):
        event.setAccepted(self.task is None)
