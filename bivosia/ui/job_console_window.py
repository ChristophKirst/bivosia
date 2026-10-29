"""Display saved processing logs and live progress for one dataset.

Examples
--------
>>> from PySide6.QtWidgets import QApplication
>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.ui.job_console_window import JobConsole
>>> app = QApplication.instance() or QApplication([])  # doctest: +SKIP
>>> console = JobConsole(Dataset('/path/to/dataset'))  # doctest: +SKIP
>>> console.show()  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QMainWindow,
                               QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout, QWidget)

from bivosia.config import JOB_TERMINAL_STATES
from bivosia.dataset.jobs import Job, list_jobs
from bivosia.processing.jobs import inspect
from bivosia.ui.icons import action_icon


class JobConsole(QMainWindow):
    """Read job logs incrementally; closing the window leaves processing running."""

    completed = Signal()

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.offset = 0
        self._closed = False
        self._job_state = None
        self.setWindowTitle(f'Console · {dataset.directory.name}')
        self.resize(850, 480)
        body = QWidget()
        layout = QVBoxLayout(body)
        jobs = QHBoxLayout()
        jobs.addWidget(QLabel('Job'))
        self.jobs = QComboBox()
        jobs.addWidget(self.jobs, 1)
        self.cancel_button = QPushButton('Cancel job')
        self.cancel_button.setIcon(action_icon('cancel'))
        self.cancel_button.clicked.connect(self.cancel_job)
        jobs.addWidget(self.cancel_button)
        layout.addLayout(jobs)
        self.job_status = QLabel()
        self.job_status.setWordWrap(True)
        self.job_status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.job_status)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)
        self.setCentralWidget(body)
        self.jobs.currentIndexChanged.connect(self._select_job)
        self.timer = QTimer(self)
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self._poll)
        self.timer.start()
        self._poll()

    def select_job(self, job_id):
        self._poll()
        self.jobs.setCurrentIndex(self.jobs.findData(job_id))

    def _select_job(self, *args):
        self.offset = 0
        self._job_state = None
        self.log.clear()
        self._poll()

    def _poll(self):
        if self._closed:
            return
        try:
            names = list_jobs(self.dataset)
            if names != [self.jobs.itemData(index) for index in range(self.jobs.count())]:
                selected = self.jobs.currentData()
                self.jobs.blockSignals(True)
                self.jobs.clear()
                for name in names:
                    job = Job(self.dataset, name)
                    self.jobs.addItem(f"{job.settings['operation']} · {name}", name)
                index = self.jobs.findData(selected)
                self.jobs.setCurrentIndex(max(0, index) if names else -1)
                self.jobs.blockSignals(False)
                if selected != self.jobs.currentData():
                    self.offset, self._job_state = 0, None
                    self.log.clear()
            name = self.jobs.currentData()
            self.cancel_button.setEnabled(False)
            if name is None:
                self.job_status.setText('No saved jobs')
                self.progress.setRange(0, 1)
                self.progress.setValue(0)
                return
            job = Job(self.dataset, name)
            result = inspect(job, self.offset)
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
            self.job_status.setText(f"{state}: {completed:,} / {total:,}" +
                                   (f"\n{status['error']}" if status.get('error') else ''))
            self.cancel_button.setEnabled(state not in JOB_TERMINAL_STATES and not job.cancelled)
            old, self._job_state = self._job_state, state
            if old is not None and old != state and state in JOB_TERMINAL_STATES:
                self.completed.emit()
        except Exception as error:
            self.job_status.setText(str(error))

    def cancel_job(self):
        name = self.jobs.currentData()
        if name is not None:
            try:
                Job(self.dataset, name).cancel()
                self._poll()
            except Exception as error:
                self.job_status.setText(str(error))

    def closeEvent(self, event):
        self._closed = True
        self.timer.stop()
        event.accept()
