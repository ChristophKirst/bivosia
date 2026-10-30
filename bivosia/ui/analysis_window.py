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
from PySide6.QtWidgets import (QComboBox, QFormLayout, QHBoxLayout,
                               QLabel,
                               QMessageBox, QPushButton, QScrollArea, QVBoxLayout, QWidget)

from bivosia.dataset.analysis_inputs import analysis_inputs
from bivosia.dataset.layout import Dataset
from bivosia.ui.processing_request import ProcessingRequest
from bivosia.ui.icons import action_icon
from bivosia.ui.dock_panel import PanelHost, PanelSplitter
from bivosia.ui.processing_panel import ProcessingPanel
from bivosia.dataset.array_transport import unpack_arrays
from bivosia.ui.analysis_parameters import AnalysisParameters
from bivosia.ui.task import Task
from bivosia.dataset.locations import remote_locations
from bivosia.dataset.jsonio import read_json
from bivosia.processing.remote import Remote
from bivosia.ui.dataset_access import location_preferences
from bivosia.ui.window_manager import notify_dataset_changed


class AnalysisWindow(PanelHost):
    """Edit shared processing parameters and follow jobs in the same window."""

    def __init__(self, dataset, windows, title, operation, *, keypoints=False,
                 input_loader=None, result_widget=None, result_controls=None,
                 required_positive=('n_animals',), processing=False, parent=None):
        super().__init__(parent)
        self.dataset, self.windows = dataset, windows
        self.operation, self.title = operation, title
        self.with_keypoints = keypoints
        self.input_loader = input_loader or (lambda dataset, settings: analysis_inputs(dataset, keypoints=keypoints))
        self.required_positive = required_positive
        self.task = None
        self.request = None
        self._refresh_pending = False
        self._closed = False
        self._ready_to_run = False
        self.setWindowTitle(f'{title} · {dataset.directory.name}')
        size = (1100, 720) if result_widget is not None else (520, 420)
        self.resize(*size)
        self.statusBar().messageChanged.connect(lambda message: self.statusBar().setToolTip(message))
        body = QWidget()
        layout = QVBoxLayout(body)
        path = QLabel(str(dataset.directory))
        path.setTextFormat(Qt.TextFormat.PlainText)
        path.setWordWrap(result_widget is not None or parent is not None)
        path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(path)
        self.parameters = AnalysisParameters(dataset, operation)
        for name in required_positive:
            if self.parameters.controls[name].minimum() == 0:
                self.parameters.controls[name].setSpecialValueText('Select')
            self.parameters.controls[name].valueChanged.connect(self._update_run)
        if 'workers' in self.parameters.controls:
            self.parameters.controls['workers'].setSpecialValueText('Automatic')
        layout.addWidget(self.parameters)
        if result_controls is not None:
            layout.addWidget(result_controls)
        if keypoints:
            form = QFormLayout()
            self.annotation = QComboBox()
            self.annotation.currentIndexChanged.connect(self._update_run)
            self.annotation.activated.connect(lambda: self._run(self._select_annotation))
            form.addRow('Annotation', self.annotation)
            self.nodes = self.parameters.nodes
            layout.addLayout(form)
        layout.addStretch()
        buttons = QHBoxLayout()
        self.refresh_button = self._button('Refresh', self.refresh, 'refresh')
        buttons.addWidget(self.refresh_button)
        layout.addLayout(buttons)
        if result_widget is not None:
            sidebar = PanelHost()
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(body)
            sidebar.setCentralWidget(scroll)
            split = PanelSplitter(Qt.Orientation.Horizontal)
            split.addWidget(sidebar)
            split.addWidget(result_widget)
            split.setSizes([440, 660])
            self.setCentralWidget(split)
        else:
            sidebar = self
            self.setCentralWidget(body)
        self.processing = None
        if processing:
            self.processing = ProcessingPanel(dataset, operation=operation)
            self.run_button = self._button('Run', self.run_analysis, 'run')
            self.run_button.setEnabled(False)
            buttons.addWidget(self.run_button)
            self.processing.monitor.completed.connect(self.refresh)
            self.processing.monitor.updated.connect(self._job_updated)
            self.processing.changed.connect(self.refresh)
            self.processing.dock(sidebar)
        self.parameters.changed.connect(self._update_run)
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

    def _job_updated(self, result):
        self.statusBar().showMessage(self.processing.monitor.job_status.text())
        self._update_run()

    def refresh(self):
        if self._closed:
            return
        if self.task is not None:
            self._refresh_pending = True
            return
        self.dataset = Dataset(self.dataset.directory)
        self._ready_to_run = False
        if self.processing is not None:
            self.run_button.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.statusBar().showMessage('Checking dataset files…')
        dataset, settings, loader = self.dataset, self.parameters.values, self.input_loader
        try:
            remote = self.processing.remote() if self.processing is not None else None
        except Exception as error:
            self.statusBar().showMessage(str(error))
            self.refresh_button.setEnabled(True)
            return
        profiles, _ = location_preferences(dataset)
        def remote_inputs(source):
            options = ({'annotation': dataset.settings.get('annotation')}
                       if self.operation == 'shape_features' else {})
            result = source.call('analysis_inputs', operation=self.operation, settings=settings, **options)
            if result.get('coordinates') is not None:
                result.update(unpack_arrays(result['coordinates']))
            return result
        def inputs():
            try:
                return loader(dataset, settings) if remote is None else remote_inputs(remote)
            except FileNotFoundError:
                pass
            sources = [lambda: loader(dataset, settings)] if remote else []
            for location in remote_locations(dataset):
                profile = next((p for p in profiles if p['host'] == location['host']), None)
                if profile is not None:
                    source = Remote(**{key: profile[key] for key in ('host', 'python', 'code')}, dataset=location['path'])
                    if source != remote:
                        sources.append(lambda source=source: remote_inputs(source))
            for load in sources:
                try:
                    info = load()
                    info['message'] = 'Required inputs will be synchronized before running.'
                    return info
                except FileNotFoundError:
                    continue
            # Settings remain editable while files are distributed across locations.
            paths = dataset.annotation_files
            annotations = [dataset.relative(path) for path in paths]
            key_frames = dataset.file('key_frames')
            info = {'animals': read_json(key_frames).get('animals') if key_frames.is_file() else None,
                    'annotations': annotations, 'annotation': dataset.settings.get('annotation') or
                    (annotations[0] if len(annotations) == 1 else None), 'node_names': [],
                    'message': 'Required inputs will be synchronized before running.'}
            if self.operation == 'shape_features':
                info.update(degree=settings['degree'], radius=settings['radius'], coordinates=None)
            return info
        task = self.task = Task(inputs, self)
        task.connect_callbacks(ready=self._ready, failed=self.statusBar().showMessage, finished=self._finished)
        task.start()

    def _ready(self, info):
        if self._closed or self._refresh_pending:
            return
        if 'n_animals' in self.parameters.controls and info['animals'] is not None:
            self.parameters.store.suggest(self.operation, n_animals=info['animals'])
        if self.with_keypoints:
            self.annotation.clear()
            for path in info['annotations']:
                self.annotation.addItem(path, path)
            self.annotation.setCurrentIndex(self.annotation.findData(info['annotation']))
            self.parameters.set_nodes(info['node_names'])
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
        if self.processing is None:
            return
        settings = self.parameters.values
        enabled = self.request is None and not self.processing.active and self._ready_to_run and all(settings[name] > 0 for name in self.required_positive)
        if self.with_keypoints:
            enabled = enabled and self.annotation.currentData() is not None
        self.run_button.setEnabled(enabled)

    def _select_annotation(self):
        selected = self.annotation.currentData()
        if selected is not None:
            Dataset(self.dataset.directory).select_annotation(selected)
            notify_dataset_changed(self.dataset)

    def run_analysis(self):
        dataset = Dataset(self.dataset.directory)
        if self.task is not None or self.request is not None:
            return
        remote = self.processing.remote()
        settings = self.parameters.values
        if 'workers' in settings:
            settings['workers'] = settings['workers'] or None
        selected = dataset.settings.get('annotation') if self.operation == 'shape_features' else None
        if self.with_keypoints:
            selected = self.annotation.currentData()
            if selected is None:
                raise ValueError('Select an annotation before aligning keypoints.')
            if remote is None:
                dataset.select_annotation(selected)
                notify_dataset_changed(dataset)
            settings['exclude_keypoints'] = [self.nodes.item(index).text()
                for index in range(self.nodes.count())
                if self.nodes.item(index).checkState() != Qt.CheckState.Checked]
        self.dataset = dataset
        request = self.request = ProcessingRequest(self, dataset, self.operation, settings, remote=remote,
                                                    windows=self.windows, annotation=selected)
        request.started.connect(lambda result: self.processing.select_job(result, remote=remote))
        request.failed.connect(self.statusBar().showMessage)
        request.finished.connect(self._request_finished)
        self.run_button.setEnabled(False)
        self.processing.setEnabled(False)
        request.start()

    def _request_finished(self):
        self.request = None
        self.processing.setEnabled(True)
        self._update_run()

    def closeEvent(self, event):
        self._closed = True
        if self.processing is not None:
            self.processing.shutdown()
        if self.request:
            self.request.cancel()
        if self.task is not None:
            self.task.requestInterruption()
        event.accept()
