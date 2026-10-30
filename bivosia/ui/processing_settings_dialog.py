"""Open a step's shared parameter panel without loading its editor."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QScrollArea, QVBoxLayout

from bivosia.config import WORKFLOW_STEPS
from bivosia.dataset.keypoints import open_keypoints
from bivosia.ui.analysis_parameters import AnalysisParameters
from bivosia.ui.task import Task
from bivosia.ui.vos_parameters import VosParameters


class ProcessingSettingsDialog(QDialog):
    def __init__(self, dataset, operation, remote, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle(f"{WORKFLOW_STEPS[operation]['label']} settings · {dataset.directory.name}")
        self.resize(520, 520)
        self.task = None
        layout = QVBoxLayout(self)
        self.parameters = (VosParameters(dataset, remote) if operation == 'vos' else
                           AnalysisParameters(dataset, operation))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.parameters)
        layout.addWidget(scroll)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)
        if operation == 'align':
            try:
                source = remote()
            except Exception as error:
                self.status.setText(str(error))
                return
            def names():
                return source.call('keypoint_names')['names'] if source else list(open_keypoints(dataset).node_names)
            task = self.task = Task(names, self)
            task.connect_callbacks(ready=self.parameters.set_nodes, failed=self.status.setText, finished=self._finished)
            task.start()

    def _finished(self):
        self.task = None

    def closeEvent(self, event):
        if self.task is not None:
            self.task.requestInterruption()
        event.accept()
