"""Embedded processing-job status, progress, cancellation and console output."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QTextCursor

from bivosia.config import JOB_TERMINAL_STATES
from bivosia.dataset.jobs import Job, list_jobs
from bivosia.processing.jobs import inspect
from bivosia.ui.processing_output import ProcessingOutput
from bivosia.ui.task import Task
from bivosia.ui.window_manager import notify_dataset_changed


class JobMonitor(ProcessingOutput):
    """Read saved jobs and incremental logs for one execution location."""

    completed = Signal()
    updated = Signal(object)

    def __init__(self, dataset, *, operation=None, parent=None):
        super().__init__(parent, job_selection=True)
        self.remote = None
        self.operation = operation
        self.dataset = dataset
        self.task = None
        self._generation = 0
        self._pending_job = None
        self.offset = 0
        self._closed = False
        self._job_state = None
        self._pending_cancel = None
        self.cancel_requested.connect(self.cancel_job)
        self.jobs.currentIndexChanged.connect(self._select_job)
        self.timer = QTimer(self)
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self._poll)
        self.timer.start()
        QTimer.singleShot(0, self._poll)

    def set_source(self, dataset, remote=None):
        if dataset.directory == self.dataset.directory and remote == self.remote:
            return
        self.dataset, self.remote = dataset, remote
        self._pending_job = None
        self.jobs.blockSignals(True)
        self.jobs.clear()
        self.jobs.blockSignals(False)
        self._select_job()

    def select_job(self, job_id, *, remote=None):
        self.set_source(self.dataset, remote)
        self._pending_job = job_id
        self._select_job()

    def _select_job(self, *args):
        self._generation += 1
        self.offset = 0
        self._job_state = None
        self.log.clear()
        self._poll()

    def _poll(self):
        if self._closed or self.task is not None:
            return
        remote = self.remote
        selected = self._pending_job or self.jobs.currentData()
        offset, generation, dataset = self.offset, self._generation, self.dataset
        def read():
            if remote:
                return remote.call('console', job_id=selected, offset=offset, operation=self.operation)
            names = list_jobs(dataset, operation=self.operation)
            name = selected if selected in names else next(iter(names), None)
            return {'jobs': [(n, Job(dataset, n).settings['operation']) for n in names],
                    'result': inspect(Job(dataset, name), offset if selected == name else 0) if name else None}
        self._background(read, lambda result: self._ready(result, generation))

    def _background(self, function, ready):
        task = self.task = Task(function, self)
        task.connect_callbacks(ready=ready, failed=self.job_status.setText, finished=self._finished)
        task.start()

    def _finished(self):
        self.task = None
        if not self._closed and self._pending_cancel is not None:
            cancel, self._pending_cancel = self._pending_cancel, None
            self._background(cancel, lambda _: self.job_status.setText('Cancellation requested'))

    def _ready(self, response, generation):
        if self._closed or generation != self._generation:
            return
        self._pending_job = None
        result = response['result']
        selected = result['id'] if result else None
        old_selected = self.jobs.currentData()
        self.jobs.blockSignals(True)
        self.jobs.clear()
        for name, operation in response['jobs']:
            self.jobs.addItem(f'{operation} · {name}', name)
        self.jobs.setCurrentIndex(self.jobs.findData(selected))
        self.jobs.blockSignals(False)
        if old_selected != selected:
            self.log.clear()
            self._job_state = None
        self.cancel_button.setEnabled(False)
        if result is None:
            self.job_status.setText('No saved jobs')
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            return
        self.offset = result['offset']
        if result['log']:
            cursor = QTextCursor(self.log.document())
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertText(result['log'])
        status = result['status']
        state = status['state']
        total, completed = int(status.get('total', 0)), int(status.get('completed', 0))
        self.progress.setRange(0, total or (1 if state in JOB_TERMINAL_STATES else 0))
        self.progress.setValue(completed)
        stage = status.get('stage', '')
        self.job_status.setText((stage + ' · ' if stage else '') + f"{state}: {completed:,} / {total:,}" +
                               (f"\n{status['error']}" if status.get('error') else ''))
        self.cancel_button.setEnabled(state not in JOB_TERMINAL_STATES)
        self.updated.emit(result)
        old, self._job_state = self._job_state, state
        if old != state and state in JOB_TERMINAL_STATES:
            self.completed.emit()
            notify_dataset_changed(self.dataset)

    def cancel_job(self):
        name = self.jobs.currentData()
        if name is None:
            return
        try:
            remote, dataset = self.remote, self.dataset
            cancel = lambda: remote.call('cancel', job_id=name) if remote else Job(dataset, name).cancel()
            self.cancel_button.setEnabled(False)
            if self.task is not None:
                self._pending_cancel = cancel
            else:
                self._background(cancel, lambda _: self.job_status.setText('Cancellation requested'))
        except Exception as error:
            self.job_status.setText(str(error))

    def shutdown(self):
        self._closed = True
        self.timer.stop()
        if self.task is not None:
            self.task.requestInterruption()
