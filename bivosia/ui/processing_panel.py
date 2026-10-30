"""Shared processing location and embedded job output for dataset windows."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from bivosia.config import JOB_TERMINAL_STATES
from bivosia.ui.dock_panel import DockPanel
from bivosia.ui.job_monitor import JobMonitor
from bivosia.ui.processing_target import ProcessingTarget


class ProcessingPanel(QWidget):
    """Own one execution-location selector and job monitor."""

    changed = Signal()

    def __init__(self, dataset, *, operation=None, monitor=None, require_dataset=True, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.active = False
        self.dock_panel = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.target = ProcessingTarget(dataset, require_dataset=require_dataset)
        layout.addWidget(self.target)
        self._saved_jobs = monitor is None
        self.monitor = JobMonitor(dataset, operation=operation) if monitor is None else monitor
        self.monitor.updated.connect(self._updated)
        layout.addWidget(self.monitor, 1)
        self.target.changed.connect(self._changed)

    def dock(self, host):
        """Place the controls and logs in one detachable Processing panel."""
        self.dock_panel = DockPanel('Processing', self, host)
        host.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.dock_panel)

    def remote(self):
        return self.target.remote()

    def _updated(self, result):
        self.active = result['status']['state'] not in JOB_TERMINAL_STATES
        self.target.setEnabled(not self.active)

    def _changed(self):
        if self._saved_jobs:
            try:
                self.monitor.set_source(self.dataset, self.remote())
            except Exception as error:
                self.monitor.job_status.setText(str(error))
        self.changed.emit()

    def set_dataset(self, dataset):
        self.dataset = dataset
        self.target.remote_panel.set_dataset(dataset.directory)
        self._changed()

    def select_job(self, result, *, remote=None):
        self.monitor.select_job(result['id'], remote=remote)

    def shutdown(self):
        self.monitor.shutdown()
        if self.dock_panel is not None:
            self.dock_panel.hide()
