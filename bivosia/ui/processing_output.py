"""Shared processing status, progress, cancellation and console layout."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QPlainTextEdit,
                               QProgressBar, QPushButton, QVBoxLayout, QWidget)

from bivosia.ui.icons import action_icon
from bivosia.ui.log_highlighter import LogHighlighter


class ProcessingOutput(QWidget):
    """Display processing output with an optional saved-job selector."""

    cancel_requested = Signal()

    def __init__(self, parent=None, *, job_selection=False):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        controls = QHBoxLayout()
        self.jobs = None
        if job_selection:
            controls.addWidget(QLabel('Job'))
            self.jobs = QComboBox()
            controls.addWidget(self.jobs, 1)
        else:
            controls.addStretch()
        self.cancel_button = QPushButton('Cancel job')
        self.cancel_button.setIcon(action_icon('cancel'))
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_requested)
        controls.addWidget(self.cancel_button)
        layout.addLayout(controls)
        self.job_status = QLabel()
        self.job_status.setWordWrap(True)
        self.job_status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.job_status)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log_highlighter = LogHighlighter(self.log)
        layout.addWidget(self.log, 1)
