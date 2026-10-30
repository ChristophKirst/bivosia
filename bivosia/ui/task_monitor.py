"""Show an interactive task's progress and logs in the shared processing layout."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import logging

from PySide6.QtCore import Signal

from bivosia.ui.processing_output import ProcessingOutput


class TaskMonitor(ProcessingOutput):
    """Follow one interactive calculation without creating a saved batch job."""

    completed = Signal()
    updated = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.task = None
        self.error = ''
        self.cancelled = False
        self.formatter = logging.Formatter('%(levelname)s %(name)s: %(message)s')
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.cancel_requested.connect(self.cancel)

    def follow(self, task):
        self.task = task
        self.error = ''
        self.cancelled = False
        self.log.clear()
        self.job_status.setText('Running')
        self.progress.setRange(0, 0)
        self.cancel_button.setEnabled(True)
        task.logged.connect(self._logged)
        task.failed.connect(self._failed)
        task.finished.connect(self._finished)
        self.updated.emit({'status': {'state': 'running'}})

    def _logged(self, record):
        self.log.appendPlainText(self.formatter.format(record))
        if hasattr(record, 'completed') and hasattr(record, 'total'):
            self.progress.setRange(0, record.total or 1)
            self.progress.setValue(record.completed)
            if self.task is not None and not self.task.isInterruptionRequested():
                self.job_status.setText(f'{record.stage}: {record.completed} / {record.total}')

    def _failed(self, error):
        self.error = error

    def _finished(self):
        self.task = None
        state = 'cancelled' if self.cancelled else 'failed' if self.error else 'complete'
        self.job_status.setText(self.error or state.capitalize())
        self.cancel_button.setEnabled(False)
        self.progress.setRange(0, max(1, self.progress.maximum()))
        if state == 'complete':
            self.progress.setValue(self.progress.maximum())
        self.updated.emit({'status': {'state': state}})
        self.completed.emit()

    def cancel(self):
        if self.task is not None:
            self.cancelled = True
            self.task.requestInterruption()
            self.cancel_button.setEnabled(False)
            self.job_status.setText('Cancellation requested')

    def shutdown(self):
        self.cancel()
