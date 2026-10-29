"""Edit a dataset's mounted storage folders and remote workstation locations.

Examples
--------
>>> from PySide6.QtWidgets import QApplication
>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.ui.dataset_locations_dialog import DatasetLocationsDialog
>>> app = QApplication.instance() or QApplication([])  # doctest: +SKIP
>>> dialog = DatasetLocationsDialog(Dataset('/data/trial'))  # doctest: +SKIP
>>> dialog.exec()  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QPushButton, QVBoxLayout)

from bivosia.dataset.layout import Dataset
from bivosia.dataset.locations import location_for, remote_locations, remote_path, save_locations
from bivosia.ui.icons import action_icon
from bivosia.ui.remote_settings import remote_settings


class DatasetLocationsDialog(QDialog):
    """Save location references without moving or deleting their files."""

    def __init__(self, dataset, parent=None, store=None):
        super().__init__(parent)
        self.dataset = dataset
        self.store = store if store is not None else remote_settings()
        self.setWindowTitle('Dataset locations')
        self.resize(700, 400)
        layout = QVBoxLayout(self)
        root = QLabel(str(dataset.directory))
        root.setTextFormat(Qt.TextFormat.PlainText)
        root.setWordWrap(True)
        layout.addWidget(root)
        self.entries = QListWidget()
        self.entries.currentItemChanged.connect(self._select)
        layout.addWidget(self.entries)
        form = QFormLayout()
        self.location = QComboBox()
        self.location.addItem('Local / mounted folder', None)
        hosts = {}
        for profile in self.store.installations.values():
            hosts.setdefault(profile['host'], profile)
        for entry in remote_locations(dataset):
            hosts.setdefault(entry['host'], {'host': entry['host']})
        for host, profile in hosts.items():
            self.location.addItem(host, profile)
        self.location.currentIndexChanged.connect(self._location_changed)
        form.addRow('Location', self.location)
        row = QHBoxLayout()
        self.path = QLineEdit()
        row.addWidget(self.path, 1)
        self.browse = QPushButton('Open')
        self.browse.setIcon(action_icon('open'))
        self.browse.clicked.connect(self._browse)
        row.addWidget(self.browse)
        form.addRow('Dataset folder', row)
        layout.addLayout(form)
        self.status = QLabel('Removing a location leaves its files in place.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        for label, callback in (('New', self._new), ('Save', self._save),
                                ('Remove', self._remove), ('Close', self.accept)):
            button = QPushButton(label)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self._reload()
        self._location_changed()

    def _reload(self):
        current = Dataset(self.dataset.directory)
        self.local = list(current.settings.get('storage_folders', []))
        self.remote = remote_locations(current)
        self.entries.clear()
        for path in self.local:
            item = QListWidgetItem(str(path), self.entries)
            item.setData(Qt.ItemDataRole.UserRole, {'path': path})
        for entry in self.remote:
            item = QListWidgetItem(f"{entry['host']}: {entry['path']}", self.entries)
            item.setData(Qt.ItemDataRole.UserRole, entry)

    def _select(self, item, previous=None):
        if item is None:
            return
        entry = item.data(Qt.ItemDataRole.UserRole)
        index = next((i for i in range(1, self.location.count())
                      if self.location.itemData(i)['host'] == entry.get('host')), 0)
        self.location.setCurrentIndex(index)
        self.path.setText(entry['path'])

    def _location_changed(self):
        profile = self.location.currentData()
        self.browse.setEnabled(profile is None)
        self.path.setText(location_for(self.dataset, profile) if profile else '')

    def _browse(self):
        path = QFileDialog.getExistingDirectory(self, 'Dataset storage folder', self.path.text())
        if path:
            self.path.setText(path)

    def _new(self):
        self.entries.setCurrentRow(-1)
        self._location_changed()

    def _without_selected(self):
        item = self.entries.currentItem()
        entry = item.data(Qt.ItemDataRole.UserRole) if item else None
        local = [path for path in self.local if entry != {'path': path}]
        remote = [value for value in self.remote if value != entry]
        return local, remote

    def _save(self):
        try:
            path = self.path.text().strip()
            if not path:
                raise ValueError('Specify a dataset folder')
            local, remote = self._without_selected()
            profile = self.location.currentData()
            if profile is None:
                selected = self.entries.currentItem()
                entry = selected.data(Qt.ItemDataRole.UserRole) if selected else {}
                index = self.local.index(entry['path']) if entry and 'host' not in entry else len(local)
                local.insert(index, path)
            else:
                remote = [entry for entry in remote if entry['host'] != profile['host']]
                remote.append({'host': profile['host'], 'path': remote_path(path)})
            save_locations(self.dataset, local, remote)
            self.store.changed.emit()
            self._reload()
            self.status.setText('Location saved.')
        except Exception as error:
            self.status.setText(str(error))

    def _remove(self):
        if self.entries.currentItem() is None:
            return
        try:
            save_locations(self.dataset, *self._without_selected())
            self.store.changed.emit()
            self._reload()
            self.status.setText('Location removed. Files are unchanged.')
        except Exception as error:
            self.status.setText(str(error))
