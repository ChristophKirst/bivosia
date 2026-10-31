"""Select a saved installation and resolve its dataset location.

Examples
--------
>>> from PySide6.QtWidgets import QApplication
>>> from bivosia.ui.remote_target import RemoteTarget
>>> app = QApplication.instance() or QApplication([])  # doctest: +SKIP
>>> target = RemoteTarget()  # doctest: +SKIP
>>> target.set_dataset('/data/recording')  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QFormLayout, QHBoxLayout, QLabel, QPushButton, QWidget

from bivosia.dataset.layout import Dataset
from bivosia.dataset.storage.locations import location_for, remember_location
from bivosia.processing.remote.client import Remote
from bivosia.ui.dataset_locations_dialog import DatasetLocationsDialog
from bivosia.ui.remote_installations_dialog import RemoteInstallationsDialog
from bivosia.ui.remote_settings import installation_label, remote_settings


class RemoteTarget(QWidget):
    """Share the remote installation selector and dataset preferences."""

    changed = Signal()

    def __init__(self, parent=None, store=None, *, include_dataset=True):
        super().__init__(parent)
        self.include_dataset = include_dataset
        self.store = store if store is not None else remote_settings()
        self.directory = self.installation_id = None
        self.dataset = None
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.installations = QComboBox()
        self.installations.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.installations.setMinimumContentsLength(10)
        row.addWidget(self.installations, 1)
        manage = QPushButton('Manage')
        manage.clicked.connect(self._manage)
        row.addWidget(manage)
        form.addRow('Installation', row)
        self.path = QLabel(self)
        self.path.setWordWrap(True)
        if include_dataset:
            row = QHBoxLayout()
            row.addWidget(self.path, 1)
            self.locations_button = QPushButton('Locations…')
            self.locations_button.clicked.connect(self._locations)
            row.addWidget(self.locations_button)
            form.addRow('Remote dataset', row)
        else:
            self.path.hide()
        self.installations.currentIndexChanged.connect(self._selected)
        self.store.changed.connect(self._reload)
        self._reload()

    @property
    def profile(self):
        return self.store.installations.get(self.installation_id)

    def _reload(self):
        self.installations.blockSignals(True)
        self.installations.clear()
        self.installations.addItem('Select installation', None)
        for key, profile in self.store.installations.items():
            self.installations.addItem(installation_label(profile), key)
        self.installations.setCurrentIndex(max(0, self.installations.findData(self.installation_id)))
        self.installations.blockSignals(False)
        self._selected()

    def _selected(self):
        self.installation_id = self.installations.currentData()
        self.installations.setToolTip(self.installations.currentText())
        self.remember()
        self._update_path()
        self.changed.emit()

    def _update_path(self):
        path = location_for(self.dataset, self.profile) if self.dataset and self.profile else ''
        self.path.setText(path)
        if self.include_dataset:
            self.locations_button.setEnabled(self.dataset is not None)

    def remember(self):
        self.store.remember(self.directory, self.installation_id)

    def set_dataset(self, directory):
        self.directory = directory
        self.dataset = Dataset(directory) if directory is not None else None
        entry = self.store.dataset(directory)
        self.installations.blockSignals(True)
        self.installations.setCurrentIndex(max(0, self.installations.findData(entry.get('installation'))))
        self.installations.blockSignals(False)
        self.installation_id = self.installations.currentData()
        self._update_path()
        self.changed.emit()

    def remote(self, *, require_dataset=True):
        profile = self.profile
        if profile is None:
            raise ValueError('Select a remote installation. Use Manage to register and check one.')
        self._update_path()
        path = self.path.text()
        if require_dataset and self.include_dataset:
            if not path:
                raise ValueError('Add a remote location using Locations, or set the installation’s default data folder')
            remember_location(self.dataset, profile['host'], path)
        self.remember()
        return Remote(**{key: profile[key] for key in ('host', 'python', 'code')}, dataset=path)

    def _locations(self):
        if self.dataset is not None:
            DatasetLocationsDialog(self.dataset, self, self.store).exec()
            self._update_path()

    def _manage(self):
        dialog = RemoteInstallationsDialog(self, self.store)
        dialog.exec()
        if dialog.selected_id in self.store.installations:
            self.installations.setCurrentIndex(self.installations.findData(dialog.selected_id))
