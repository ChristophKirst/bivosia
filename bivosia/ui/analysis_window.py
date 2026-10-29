"""Shared settings window for dataset analysis jobs.

Examples
--------
>>> from PySide6.QtWidgets import QApplication
>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.ui.window_manager import DatasetWindows
>>> app = QApplication.instance() or QApplication([])  # doctest: +SKIP
>>> windows = DatasetWindows(app)  # doctest: +SKIP
>>> window = windows.open('features', Dataset('/path/to/dataset'))  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QComboBox, QFormLayout, QHBoxLayout,
                               QLabel, QListWidget, QListWidgetItem, QMainWindow,
                               QMessageBox, QPushButton, QVBoxLayout, QWidget)

from bivosia.dataset.analysis_inputs import analysis_inputs
from bivosia.dataset.layout import Dataset
from bivosia.processing.jobs import require_idle, submit
from bivosia.ui.icons import action_icon
from bivosia.ui.parameters import ParameterForm
from bivosia.ui.task import Task


class AnalysisWindow(QMainWindow):
    """Collect existing processing options and open the shared job console."""

    def __init__(self, dataset, windows, title, operation, options, defaults, *, keypoints=False,
                 input_loader=None, result_widget=None, required_positive=('n_animals',), help_text=None):
        super().__init__()
        self.dataset, self.windows = dataset, windows
        self.operation, self.title = operation, title
        self.with_keypoints = keypoints
        self.input_loader = input_loader or (lambda dataset, settings: analysis_inputs(dataset, keypoints=keypoints))
        self.required_positive = required_positive
        self.task = None
        self.console = None
        self._refresh_pending = False
        self._closed = False
        self._ready_to_run = False
        self.setWindowTitle(f'{title} · {dataset.directory.name}')
        size = (700, 580) if result_widget is not None else (520, 420)
        self.resize(*size)
        self.statusBar().messageChanged.connect(lambda message: self.statusBar().setToolTip(message))
        body = QWidget()
        layout = QVBoxLayout(body)
        path = QLabel(str(dataset.directory))
        path.setTextFormat(Qt.TextFormat.PlainText)
        path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(path)
        self.parameters = ParameterForm(options, defaults, help_text={
            'workers': 'Number of parallel CPU processes. Automatic uses available CPUs, limited by the number of blocks.',
            **(help_text or {}),
        })
        for name in required_positive:
            self.parameters.controls[name].setSpecialValueText('Select')
            self.parameters.controls[name].valueChanged.connect(self._update_run)
        if 'workers' in self.parameters.controls:
            self.parameters.controls['workers'].setSpecialValueText('Automatic')
        layout.addWidget(self.parameters)
        if keypoints:
            form = QFormLayout()
            self.annotation = QComboBox()
            self.annotation.currentIndexChanged.connect(self._update_run)
            form.addRow('Annotation', self.annotation)
            self.nodes = QListWidget()
            self.nodes.itemChanged.connect(self._update_run)
            form.addRow('Keypoints', self.nodes)
            layout.addLayout(form)
        if result_widget is not None:
            layout.addWidget(result_widget, 1)
        else:
            layout.addStretch()
        buttons = QHBoxLayout()
        self.refresh_button = self._button('Refresh', self.refresh, 'refresh')
        buttons.addWidget(self.refresh_button)
        buttons.addWidget(self._button('Console', self.open_console, 'console'))
        self.run_button = self._button('Run', self.run_analysis, 'run')
        self.run_button.setEnabled(False)
        buttons.addWidget(self.run_button)
        layout.addLayout(buttons)
        self.setCentralWidget(body)
        self.refresh()

    def _button(self, label, callback, icon):
        button = QPushButton(label)
        button.setIcon(action_icon(icon))
        button.clicked.connect(lambda: self._run(callback))
        return button

    def _run(self, callback):
        try:
            return callback()
        except Exception as error:
            self.statusBar().showMessage(str(error))
            QMessageBox.warning(self, self.title, str(error))

    def open_console(self):
        console = self.windows.open('console', self.dataset)
        if console is not self.console:
            console.completed.connect(self.refresh)
            self.console = console
        return console

    def refresh(self):
        if self._closed:
            return
        if self.task is not None:
            self._refresh_pending = True
            return
        self.dataset = Dataset(self.dataset.directory)
        self._ready_to_run = False
        self.run_button.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.statusBar().showMessage('Checking dataset files…')
        dataset, settings, loader = self.dataset, self.parameters.values, self.input_loader
        task = self.task = Task(lambda: loader(dataset, settings), QApplication.instance())
        task.ready.connect(self._ready)
        task.failed.connect(self.statusBar().showMessage)
        task.finished.connect(self._finished)
        task.finished.connect(task.deleteLater)
        QApplication.instance().aboutToQuit.connect(task.wait)
        self.destroyed.connect(task.requestInterruption)
        task.start()

    def _ready(self, info):
        if self._closed:
            return
        if 'n_animals' in self.parameters.controls and info['animals'] is not None:
            self.parameters.controls['n_animals'].setValue(info['animals'])
        if self.with_keypoints:
            self.annotation.clear()
            for path in info['annotations']:
                self.annotation.addItem(path, path)
            self.annotation.setCurrentIndex(self.annotation.findData(info['annotation']))
            self.nodes.clear()
            for name in info['node_names']:
                item = QListWidgetItem(name, self.nodes)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked)
        self._ready_to_run = True
        self.statusBar().showMessage(info.get('message', ''))
        self._update_run()

    def _finished(self):
        self.task = None
        self.refresh_button.setEnabled(True)
        if self._refresh_pending:
            self._refresh_pending = False
            self.refresh()

    def _update_run(self, *args):
        settings = self.parameters.values
        enabled = self._ready_to_run and all(settings[name] > 0 for name in self.required_positive)
        if self.with_keypoints:
            enabled = enabled and self.annotation.currentData() is not None
        self.run_button.setEnabled(enabled)

    def run_analysis(self):
        dataset = Dataset(self.dataset.directory)
        require_idle(dataset)
        settings = self.parameters.values
        if 'workers' in settings:
            settings['workers'] = settings['workers'] or None
        if self.with_keypoints:
            selected = self.annotation.currentData()
            if selected is None:
                raise ValueError('Select an annotation before aligning keypoints.')
            dataset.select_annotation(selected)
            settings['exclude_keypoints'] = [self.nodes.item(index).text()
                for index in range(self.nodes.count())
                if self.nodes.item(index).checkState() != Qt.CheckState.Checked]
        self.dataset = dataset
        job = submit(dataset, self.operation, settings)
        self.open_console().select_job(job.id)
        self.statusBar().showMessage(f'Started {self.title.lower()}.')

    def closeEvent(self, event):
        self._closed = True
        if self.task is not None:
            self.task.requestInterruption()
        event.accept()
