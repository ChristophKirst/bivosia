"""Share dataset processing parameters between editors and settings dialogs."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from bivosia.dataset.processing_settings import step_settings, save_step_settings
from bivosia.dataset.layout import Dataset
from bivosia.ui.window_manager import notify_dataset_changed


class StepSettings(QObject):
    changed = Signal(str, object)

    def __init__(self, dataset):
        super().__init__(QApplication.instance())
        self.dataset = dataset
        self.values = {}

    def get(self, operation):
        if operation not in self.values:
            self.values[operation] = step_settings(self.dataset, operation)
        return dict(self.values[operation])

    def set(self, operation, values):
        if values == self.get(operation):
            return
        save_step_settings(self.dataset, operation, values)
        self.values[operation] = dict(values)
        self.changed.emit(operation, dict(values))
        notify_dataset_changed(self.dataset)

    def suggest(self, operation, **values):
        saved = Dataset(self.dataset.directory).settings.get('processing', {}).get(operation, {})
        current = self.get(operation)
        updated = {**current, **{key: value for key, value in values.items() if key not in saved}}
        if updated != current:
            self.values[operation] = updated
            self.changed.emit(operation, dict(updated))


def step_store(dataset):
    app = QApplication.instance()
    if not hasattr(app, '_step_settings'):
        app._step_settings = {}
    key = dataset.directory
    if key not in app._step_settings:
        app._step_settings[key] = StepSettings(dataset)
    return app._step_settings[key]
