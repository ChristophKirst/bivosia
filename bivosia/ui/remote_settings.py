"""Saved remote installations and dataset selections in application preferences."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import json
import uuid

from PySide6.QtCore import QObject, QSettings, Signal
from PySide6.QtWidgets import QApplication

from bivosia.config import APPLICATION_SETTINGS, REMOTE_SETTINGS
from bivosia.dataset.locations import remote_path


class RemoteSettings(QObject):
    """Share saved installation choices across the application's windows."""

    changed = Signal()

    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.settings = settings if settings is not None else QSettings(*APPLICATION_SETTINGS)

    def _read(self, key):
        self.settings.sync()
        value = json.loads(self.settings.value(REMOTE_SETTINGS[key], '{}'))
        if not isinstance(value, dict):
            raise ValueError('Invalid remote installation preferences')
        return value

    def _write(self, key, value):
        self.settings.setValue(REMOTE_SETTINGS[key], json.dumps(value))
        self.settings.sync()
        if self.settings.status() != QSettings.Status.NoError:
            raise OSError('Could not save remote installation preferences')

    @property
    def installations(self):
        return self._read('installations')

    def save(self, host, details, installation_id=None, *, data_folder=''):
        """Save details returned by a successful remote installation check."""
        entries = self.installations
        profile = {key: details[key] for key in ('python', 'code', 'checkpoints')}
        profile['host'] = host
        profile['data_folder'] = remote_path(data_folder) if data_folder else ''
        for key, value in entries.items():
            if key != installation_id and all(value[field] == profile[field] for field in ('host', 'python', 'code')):
                raise ValueError('This remote installation is already registered')
        installation_id = installation_id or uuid.uuid4().hex
        entries[installation_id] = profile
        self._write('installations', entries)
        self.changed.emit()
        return installation_id

    def remove(self, installation_id):
        entries = self.installations
        entries.pop(installation_id, None)
        self._write('installations', entries)
        self.changed.emit()

    def dataset(self, directory):
        return self._read('datasets').get(str(directory), {}) if directory is not None else {}

    def remember(self, directory, installation_id):
        """Remember the selected software installation; paths belong to the dataset."""
        if directory is None or installation_id is None:
            return
        entries = self._read('datasets')
        entries[str(directory)] = {'installation': installation_id}
        self._write('datasets', entries)


def remote_settings():
    """Return the application's shared remote preferences."""
    app = QApplication.instance()
    if not hasattr(app, '_bivosia_remote_settings'):
        app._bivosia_remote_settings = RemoteSettings(app)
    return app._bivosia_remote_settings


def installation_label(profile):
    return f"{profile['host']} · {profile['python']} · {profile['code']}"
