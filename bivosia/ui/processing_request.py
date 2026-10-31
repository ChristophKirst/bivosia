"""Prepare a processing location in the dataset viewer, then submit its job."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import uuid

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QMessageBox

from bivosia.dataset.layout import Dataset
from bivosia.processing.jobs import check_inputs, inspect, submit
from bivosia.ui.task import Task


class ProcessingRequest(QObject):
    started = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, parent, dataset, operation, settings, *, remote=None, windows=None, annotation=None):
        super().__init__(QApplication.instance())
        self.window, self.dataset = parent, dataset
        self.operation, self.settings = operation, dict(settings)
        self.remote, self.windows, self.annotation = remote, windows, annotation
        self.task = None
        self.viewer = None
        self.job_id = uuid.uuid4().hex
        self.synced = False
        self.closed = False
        self._continuation = None
        parent.destroyed.connect(self.cancel)

    def _background(self, function, ready):
        task = self.task = Task(function, self)
        def delivered(result):
            self._continuation = lambda: ready(result)
        task.connect_callbacks(ready=delivered, failed=self._failed, finished=self._finished_task)
        task.start()

    def _finished_task(self):
        self.task = None
        if self.closed:
            self.deleteLater()
            return
        continuation, self._continuation = self._continuation, None
        if continuation is not None and not self.closed:
            continuation()

    def start(self):
        if self.closed:
            return
        if not self.dataset.description.is_file():
            self.dataset.save()
        def check():
            if self.remote:
                return self.remote.call('preflight', operation=self.operation, settings=self.settings,
                                        annotation=self.annotation)
            dataset = Dataset(self.dataset.directory)
            if self.annotation:
                dataset.settings['annotation'] = self.annotation
            return check_inputs(dataset, self.operation, self.settings)
        self._background(check, self._checked)

    def _checked(self, result):
        if result['missing']:
            if self.synced:
                self._failed('Inputs are still unavailable after synchronization: ' + result['message'])
                return
            try:
                if self.windows is not None:
                    self.viewer = self.windows.open('files', self.dataset)
                else:
                    from bivosia.ui.dataset_viewer import DatasetWidget
                    self.viewer = DatasetWidget(self.dataset)
                    self.window._dataset_viewer = self.viewer
                    self.viewer.show()
                location = ({'host': self.remote.host, 'path': self.remote.dataset} if self.remote else
                            {'host': '', 'path': str(self.dataset.directory)})
                self.viewer.prepare_processing_sync(self.operation, location, self._synchronized,
                                                     annotation=self.annotation, settings=self.settings)
            except Exception as error:
                self._failed(str(error))
            return
        if result.get('replace_analysis'):
            answer = QMessageBox.question(self.window, 'Mask analysis', 'Replace the existing mask analysis results?',
                                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                                          QMessageBox.StandardButton.Cancel)
            if answer != QMessageBox.StandardButton.Yes:
                self.cancel()
                return
        def launch():
            if self.remote:
                return self.remote.call('submit', operation=self.operation, settings=self.settings,
                                        job_id=self.job_id, annotation=self.annotation)
            dataset = Dataset(self.dataset.directory)
            if self.annotation:
                dataset.select_annotation(self.annotation)
            return inspect(submit(dataset, self.operation, self.settings, job_id=self.job_id))
        self._background(launch, self._started)

    def _synchronized(self, success, message=''):
        if self.closed:
            return
        if success:
            self.synced = True
            self.start()
        elif message:
            self._failed(message)
        else:
            self.cancel()

    def _started(self, response):
        self.started.emit(response)
        self.cancel()

    def _failed(self, message):
        self.failed.emit(message)
        self.cancel()

    def cancel(self):
        if not self.closed:
            self.closed = True
            self.finished.emit()
            if self.task is None:
                self.deleteLater()
