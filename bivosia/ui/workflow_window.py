"""One dataset's workflow, editor actions and background processing progress.

Examples
--------
>>> from PySide6.QtWidgets import QApplication
>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.ui.window_manager import DatasetWindows
>>> app = QApplication.instance() or QApplication([])  # doctest: +SKIP
>>> windows = DatasetWindows(app)  # doctest: +SKIP
>>> workflow = windows.open('workflow', Dataset('/path/to/dataset'))  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QAbstractItemView, QAbstractScrollArea, QCheckBox, QHeaderView, QHBoxLayout,
                               QLabel, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem,
                               QToolButton, QVBoxLayout, QWidget)

from bivosia.config import JOB_TERMINAL_STATES, WORKFLOW_STEPS
from bivosia.dataset.layout import Dataset
from bivosia.dataset.processing_settings import execution_settings
from bivosia.processing.workflow import required_products, step_availability, validate_step_settings
from bivosia.ui.job_monitor import JobMonitor
from bivosia.ui.processing_target import ProcessingTarget
from bivosia.ui.processing_request import ProcessingRequest
from bivosia.ui.processing_settings_dialog import ProcessingSettingsDialog
from bivosia.ui.step_settings import step_store
from bivosia.processing.remote_inventory import inspect_locations
from bivosia.ui.dataset_access import location_preferences
from bivosia.ui.colors import WORKFLOW_COLORS
from bivosia.ui.dataset_viewer import _duration
from bivosia.ui.dock_panel import DockPanel, PanelHost
from bivosia.ui.icons import action_icon
from bivosia.ui.legend_panel import LegendPanel, legend_groups_html
from bivosia.ui.plot_models import LegendGroup
from bivosia.ui.task import Task


STATE_LABELS = {'available': 'Available', 'missing': 'Needs calculation / annotation',
                'review': 'Needs review / validation', 'checking': 'Checking…'}


class _WorkflowTable(QTableWidget):
    """Keep step information and controls aligned in a fixed workflow order."""

    def __init__(self, count):
        super().__init__(count, 6)
        self.setHorizontalHeaderLabels(('', 'Step', 'Info', 'Action', 'Settings', 'Run'))
        self.verticalHeader().hide()
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.setShowGrid(False)
        self.setAlternatingRowColors(True)
        self.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

    def set_control(self, row, column, control):
        cell = QWidget()
        layout = QHBoxLayout(cell)
        layout.setContentsMargins(4, 3, 4, 3)
        alignment = (Qt.AlignmentFlag.AlignVCenter if column == 3 or self.columnSpan(row, column) > 1
                     else Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(control, 0, alignment)
        self.setCellWidget(row, column, cell)


class WorkflowWindow(PanelHost):
    """Guide a single dataset through the existing editors and processing steps."""

    def __init__(self, dataset, windows):
        super().__init__()
        self.status = self.statusBar()
        self.status.messageChanged.connect(lambda message: self.status.setToolTip(message))
        self.dataset, self.windows = dataset, windows
        self.info = None
        self.task = None
        self._refresh_pending = False
        self._pending_steps = None
        self.request = None
        self._job_active = False
        self.settings_dialogs = {}
        self.store = step_store(dataset)
        self._closed = False
        self.setWindowTitle(f'Workflow · {dataset.directory.name}')
        self.resize(850, 650)
        body = QWidget()
        layout = QVBoxLayout(body)
        top = QHBoxLayout()
        path = QLabel(str(dataset.directory))
        path.setTextFormat(Qt.TextFormat.PlainText)
        path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        top.addWidget(path, 1)
        top.addWidget(self._button('Viewer', lambda: self.open('viewer'), 'view'))
        top.addWidget(self._button('View files', lambda: self.open('files'), 'files'))
        self.refresh_button = self._button('Refresh', self.refresh, 'refresh')
        top.addWidget(self.refresh_button)
        layout.addLayout(top)
        self.labels = {}
        self.actions = {}
        self.indicators = {}
        self.included = {}
        self.gears = {}
        operations = {definition['output']: name for name, definition in WORKFLOW_STEPS.items()}
        definitions = (
            ('video', '1. Dataset and video', [('Open Video', lambda: self.open('video'), 'open')]),
            ('key_frames', '2. Key frames', [('Open Keyframes', lambda: self.open('key_frames'), 'edit')]),
            ('masks', '3. VOS inference', [('Open VOS', lambda: self.open('vos'), 'open')]),
            ('analysis', '4. Mask analysis', [('Open Masks', lambda: self.open('mask_analysis'), 'open')]),
            ('annotations', '5. ZoD detection and linking', [('Open ZoDs', lambda: self.open('zones'), 'edit')]),
            ('export', '6. Export and review', [('Export', lambda: self.select_steps(('export',)), 'export')]),
            ('features', 'Mask features', [('Open features', lambda: self.open('features'), 'open')]),
            ('alignment', 'Keypoint alignment', [('Open alignment', lambda: self.open('alignment'), 'open')]),
        )
        self.steps = _WorkflowTable(len(definitions) + 1)
        for row, (key, title, actions) in enumerate(definitions):
            box = self.indicators[key] = QLabel()
            box.setTextFormat(Qt.TextFormat.RichText)
            self._set_state(key, 'checking')
            self.steps.set_control(row, 0, box)
            self.steps.setItem(row, 1, QTableWidgetItem(title))
            label = self.labels[key] = QTableWidgetItem('Checking…')
            self.steps.setItem(row, 2, label)
            self.actions[key] = []
            for text, callback, icon in actions:
                button = self._button(text, callback, icon)
                self.actions[key].append(button)
                self.steps.set_control(row, 3, button)
            if key in operations:
                operation = operations[key]
                include = self.included[operation] = QCheckBox()
                include.setAccessibleName('Include ' + WORKFLOW_STEPS[operation]['label'])
                include.toggled.connect(self._update_plan)
                self.steps.set_control(row, 5, include)
            if key == 'video' or key in operations:
                operation = 'dataset' if key == 'video' else operations[key]
                gear = self.gears[operation] = QToolButton()
                gear.setIcon(action_icon('settings'))
                gear.setToolTip('Dataset settings' if operation == 'dataset' else
                                WORKFLOW_STEPS[operation]['label'] + ' settings')
                gear.setAccessibleName(gear.toolTip())
                gear.clicked.connect(lambda checked=False, name=operation: self.open_settings(name))
                self.steps.set_control(row, 4, gear)
        self.run_button = self._button('Run', self.run_selected, 'run')
        self.run_button.setEnabled(False)
        self.steps.setSpan(len(definitions), 4, 1, 2)
        self.steps.set_control(len(definitions), 4, self.run_button)
        layout.addWidget(self.steps, 1)
        self.legend = QLabel(legend_groups_html((LegendGroup(items=tuple(
            (STATE_LABELS[state], WORKFLOW_COLORS[state]) for state in ('available', 'missing', 'review'))),)))
        self.legend.setTextFormat(Qt.TextFormat.RichText)
        self.legend.setWordWrap(True)
        self.legend.setToolTip('Colors describe saved outputs. Editors and processing validate their inputs.')
        layout.addWidget(LegendPanel.containing(self.legend))
        processing = QWidget()
        processing_layout = QVBoxLayout(processing)
        processing_layout.setContentsMargins(0, 0, 0, 0)
        self.target = ProcessingTarget(dataset)
        self.target.changed.connect(self._target_changed)
        processing_layout.addWidget(self.target)
        self.monitor = JobMonitor(dataset)
        self.monitor.updated.connect(self._job_updated)
        processing_layout.addWidget(self.monitor, 1)
        self.store.changed.connect(lambda *args: self._update_plan())
        self.setCentralWidget(body)
        self.processing_dock = DockPanel('Processing', processing, self)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.processing_dock)
        self.resizeDocks([self.processing_dock], [self.processing_dock.minimumSizeHint().height()],
                         Qt.Orientation.Vertical)
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(10000)
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.timeout.connect(self.refresh)
        self.windows.dataset_changed.connect(self._dataset_changed)
        self.refresh()

    def _set_state(self, key, state, detail=''):
        box = self.indicators[key]
        box.setProperty('workflow_state', state)
        box.setText(legend_groups_html((LegendGroup(items=(('', WORKFLOW_COLORS[state]),)),)))
        box.setAccessibleName(STATE_LABELS[state])
        box.setToolTip(STATE_LABELS[state] + (f'\n{detail}' if detail else ''))

    def _button(self, text, callback, icon):
        button = QPushButton(text)
        button.setIcon(action_icon(icon))
        button.clicked.connect(lambda: self._run(callback))
        return button

    def _run(self, callback):
        try:
            return callback()
        except Exception as error:
            QMessageBox.warning(self, 'Workflow', str(error))

    def open(self, kind):
        return self.windows.open(kind, self.dataset)

    def open_settings(self, operation):
        dialog = self.settings_dialogs.get(operation)
        if dialog is None:
            if operation == 'dataset':
                from bivosia.ui.dataset_settings_dialog import DatasetSettingsDialog
                dialog = DatasetSettingsDialog(self.dataset, self)
                dialog.finished.connect(self.refresh)
            else:
                dialog = ProcessingSettingsDialog(self.dataset, operation, self.target.remote, self)
            self.settings_dialogs[operation] = dialog
            dialog.destroyed.connect(lambda: self.settings_dialogs.pop(operation, None))
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _available_products(self):
        if self.info is None:
            return set()
        info = self.info
        products = {name for name in ('key_frames', 'masks', 'analysis', 'features', 'alignment')
                    if self._available(info[name])}
        if not info['key_frame_count']:
            products.discard('key_frames')
        for name in ('video', 'keypoints'):
            if info[name]['exists'] and not info[name]['error']:
                products.add(name)
        if info['annotations'] and not info['annotation_error']:
            products.add('annotations')
        if info['export'] and info['export']['exists'] and not info['export_error'] and info['export_current']:
            products.add('export')
        return products

    def _selected_steps(self):
        return [{'operation': name, 'settings': execution_settings(name, self.store.get(name))}
                for name, checkbox in self.included.items() if checkbox.isChecked()]

    def _update_plan(self, *args):
        if not hasattr(self, 'target'):
            return
        selected = {name for name, checkbox in self.included.items() if checkbox.isChecked()}
        settings = {name: self.store.get(name) for name in self.included}
        available = self._available_products()
        enabled = step_availability(available, selected, settings=settings)
        feature_inputs = set(available)
        annotation = self.info['annotation'] if self.info else None
        if 'export' not in available and not (annotation and annotation['valid']):
            feature_inputs.discard('annotations')
        self.actions['features'][0].setEnabled(not step_availability(feature_inputs, (), settings=settings)['shape_features'])
        busy = self.request is not None or self._job_active
        for name, missing in enabled.items():
            box = self.included[name]
            box.blockSignals(True)
            if missing:
                box.setChecked(False)
            box.setEnabled(not busy and not missing)
            box.setToolTip('Missing inputs: ' + ', '.join(sorted(missing)) if missing else 'Include this step')
            box.blockSignals(False)
            self.gears[name].setEnabled(not busy)
        self.gears['dataset'].setEnabled(not busy)
        steps = self._selected_steps()
        error = ''
        if steps:
            try:
                validate_step_settings(steps)
            except ValueError as failure:
                error = str(failure)
        self.run_button.setEnabled(bool(steps) and not busy and not error)
        self.run_button.setToolTip(error)
        self.target.setEnabled(not busy)
        if error:
            self.statusBar().showMessage(error)

    def run_selected(self):
        if self.request is not None or self._job_active:
            return
        steps = self._selected_steps()
        validate_step_settings(steps)
        remote = self.target.remote()
        self.dataset = Dataset(self.dataset.directory)
        annotation = (self.dataset.settings.get('annotation')
                      if {'annotations', 'export'} & required_products(steps) else None)
        request = self.request = ProcessingRequest(self, self.dataset, 'workflow', {'steps': steps},
                                                    remote=remote, windows=self.windows, annotation=annotation)
        request.started.connect(lambda result: self._started(result, remote))
        request.failed.connect(self.statusBar().showMessage)
        request.finished.connect(self._request_finished)
        self._update_plan()
        request.start()

    def _started(self, result, remote):
        self.monitor.select_job(result['id'], remote=remote)
        self._job_updated(result)

    def _target_changed(self):
        try:
            self.monitor.set_source(self.dataset, self.target.remote())
        except Exception as error:
            self.monitor.job_status.setText(str(error))
        self.refresh()

    def _job_updated(self, result):
        self._job_active = result['status']['state'] not in JOB_TERMINAL_STATES
        self._update_plan()

    def _request_finished(self):
        self.request = None
        self._update_plan()

    def select_steps(self, operations):
        """Refresh saved outputs and select the steps requested by an editor."""
        if self.request is not None or self._job_active:
            raise ValueError('Wait for the current workflow job')
        self._pending_steps = tuple(operations)
        self.refresh()

    def _dataset_changed(self, directory):
        if directory == self.dataset.directory:
            self.refresh()

    def refresh(self):
        if self._closed:
            return
        self.refresh_timer.stop()
        if self.task is not None:
            self._refresh_pending = True
            return
        dataset = Dataset(self.dataset.directory) if self.dataset.description.is_file() else self.dataset
        self.dataset = dataset
        self.refresh_button.setEnabled(False)
        self.statusBar().showMessage('Checking dataset files…')
        profiles, preferred = location_preferences(dataset)
        task = self.task = Task(lambda: inspect_locations(dataset, profiles, preferred, include_workflow=True), self)
        task.connect_callbacks(ready=self._ready, failed=self.statusBar().showMessage, finished=self._finished)
        task.start()

    @staticmethod
    def _available(entries):
        return all(entry['exists'] and not entry['error'] for entry in entries.values())

    @staticmethod
    def _file_status(entries):
        errors = [item['error'] for item in entries.values() if item['error']]
        if errors:
            return '\n'.join(errors)
        count = sum(item['exists'] for item in entries.values())
        hosts = sorted({item['host'] for item in entries.values() if item.get('host')})
        return f'{count} / {len(entries)} files available' + (' · ' + ', '.join(hosts) if hosts else '')

    def _ready(self, info):
        if self._closed:
            return
        self.info = info
        if info['animals'] is not None:
            for operation in ('quantify', 'align'):
                self.store.suggest(operation, n_animals=info['animals'])
        video = info['video']
        metadata = video['metadata']
        self.labels['video'].setText(video['error'] or (
            f'{metadata.frames:,} frames; {_duration(metadata.duration)}' if metadata and metadata.frames is not None
            else 'Video available' if video['exists'] else 'Select a video'))
        for name in ('key_frames', 'masks', 'analysis', 'features', 'alignment'):
            self.labels[name].setText(self._file_status(info[name]))
        if self._available(info['key_frames']) and info['key_frame_count'] is not None:
            self.labels['key_frames'].setText(f"{info['key_frame_count']} saved key-frame masks")
        annotation = info['annotation']
        annotation_valid = bool(annotation and annotation['valid'] and not info['annotation_error'])
        annotation_text = f"{len(info['annotations'])} saved annotation files"
        if annotation is not None:
            validation = annotation['error'] or ('Ready to export' if annotation_valid else
                                                f"{len(annotation['issues'])} validation issues")
            annotation_text = f"{annotation['path'].name} · {validation}"
        self.labels['annotations'].setText(info['annotation_error'] or annotation_text)
        self.labels['annotations'].setToolTip(info['annotation_error'] or (
            annotation['error'] or '\n'.join(annotation['issues']) if annotation else ''))
        exported = info['export']
        export_text = 'No saved export for the selected annotation'
        if exported:
            export_text = exported['error'] or (f"Saved export: {exported['path'].name}" if exported['exists']
                                               else 'The saved export file is missing')
            if exported['exists'] and not exported['error'] and not info['export_current']:
                export_text = 'Annotations changed; export again'
        self.labels['export'].setText(info['export_error'] or export_text)
        has_video = video['exists'] and not video['error']
        self.actions['video'][0].setEnabled(bool(has_video))
        has_masks = self._available(info['masks'])
        has_analysis = self._available(info['analysis'])
        has_key_frames = self._available(info['key_frames']) and bool(info['key_frame_count'])
        has_export = bool(exported and exported['exists'] and not exported['error']
                          and not info['export_error'] and info['export_current'])
        for key, available in (('video', has_video), ('key_frames', has_key_frames),
                               ('masks', has_masks), ('analysis', has_analysis), ('export', has_export)):
            self._set_state(key, 'available' if available else 'missing', self.labels[key].text())
        annotation_state = ('available' if annotation_valid else
                            'review' if info['annotations'] or info['annotation_error'] else 'missing')
        self._set_state('annotations', annotation_state,
                        self.labels['annotations'].text())
        for key in ('key_frames', 'masks'):
            self.actions[key][0].setEnabled(bool(has_video))
        self.actions['analysis'][0].setEnabled(has_masks)
        for key in ('annotations', 'export'):
            for button in self.actions[key]:
                button.setEnabled(bool(has_video and has_masks and has_analysis))
        for key in ('features', 'alignment'):
            self._set_state(key, 'available' if self._available(info[key]) else 'missing', self.labels[key].text())
        self.actions['alignment'][0].setEnabled(bool(has_masks and info['keypoints']['exists']
                                                   and not info['keypoints']['error'] and info['annotations']))
        if not has_video:
            next_step = 'Select the experimental video in Dataset settings.'
        elif not has_masks:
            next_step = ('Open VOS to check key frames and run inference.'
                         if self._available(info['key_frames']) and info['key_frame_count']
                         else 'Annotate and save key frames.')
        elif not has_analysis:
            next_step = 'Open Mask analysis and run it.'
        elif not annotation_valid:
            next_step = 'Open ZoD detection and linking to review and validate.'
        elif not has_export:
            next_step = 'Export the validated annotation.'
        else:
            next_step = 'Workflow outputs are available.'
        self.statusBar().showMessage(' · '.join(info['remote_errors']) if info.get('remote_errors') and not has_masks
                                     else f'Suggested next step: {next_step}')
        if self._pending_steps is not None:
            for name, checkbox in self.included.items():
                checkbox.blockSignals(True)
                checkbox.setChecked(name in self._pending_steps)
                checkbox.blockSignals(False)
            self._pending_steps = None
        self._update_plan()

    def _finished(self):
        self.task = None
        if self._closed:
            return
        self.refresh_button.setEnabled(True)
        if self._refresh_pending:
            self._refresh_pending = False
            self.refresh()
        else:
            self.refresh_timer.start()

    def closeEvent(self, event):
        self._closed = True
        self.monitor.shutdown()
        self.processing_dock.hide()
        if self.request is not None:
            self.request.cancel()
        self.refresh_timer.stop()
        if self.task is not None:
            self.task.requestInterruption()
        event.accept()
