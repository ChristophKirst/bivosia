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

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QGridLayout, QHBoxLayout, QInputDialog,
                               QLabel, QMainWindow, QMessageBox, QPushButton, QVBoxLayout, QWidget)

from bivosia.dataset.layout import Dataset
from bivosia.dataset.workflow import inspect_workflow
from bivosia.processing.jobs import require_idle, submit
from bivosia.ui.colors import WORKFLOW_COLORS
from bivosia.ui.dataset_viewer import _duration
from bivosia.ui.icons import action_icon
from bivosia.ui.legend_panel import LegendPanel, legend_groups_html
from bivosia.ui.plot_models import LegendGroup
from bivosia.ui.task import Task


STATE_LABELS = {'available': 'Available', 'missing': 'Needs calculation / annotation',
                'review': 'Needs review / validation', 'checking': 'Checking…'}


class WorkflowWindow(QMainWindow):
    """Guide a single dataset through the existing editors and processing steps."""

    def __init__(self, dataset, windows):
        super().__init__()
        self.status = self.statusBar()
        self.status.messageChanged.connect(lambda message: self.status.setToolTip(message))
        self.dataset, self.windows = dataset, windows
        self.info = None
        self.task = None
        self.console = None
        self._closed = False
        self.setWindowTitle(f'Workflow · {dataset.directory.name}')
        self.resize(850, 370)
        body = QWidget()
        layout = QVBoxLayout(body)
        top = QHBoxLayout()
        path = QLabel(str(dataset.directory))
        path.setTextFormat(Qt.TextFormat.PlainText)
        path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        top.addWidget(path, 1)
        top.addWidget(self._button('View files', lambda: self.open('files'), 'files'))
        top.addWidget(self._button('Console', self.open_console, 'console'))
        self.refresh_button = self._button('Refresh', self.refresh, 'refresh')
        top.addWidget(self.refresh_button)
        layout.addLayout(top)
        steps = QGridLayout()
        self.labels = {}
        self.actions = {}
        self.indicators = {}
        definitions = (
            ('video', '1. Dataset and video', [('Select video', self.select_video, 'open')]),
            ('key_frames', '2. Key frames', [('Open editor', lambda: self.open('key_frames'), 'edit')]),
            ('masks', '3. VOS inference', [('Open VOS', lambda: self.open('vos'), 'open')]),
            ('analysis', '4. Mask analysis', [('Run', self.run_analysis, 'run'), ('View', lambda: self.open('viewer'), 'view')]),
            ('annotations', '5. ZoD annotation and linking', [('Open editor', lambda: self.open('zones'), 'edit')]),
            ('export', '6. Export and review', [('Export', lambda: self.revision_action('export_global'), 'export'),
                                                ('View result', lambda: self.revision_action('open_result'), 'view')]),
            ('features', 'Mask features', [('Open', lambda: self.open('features'), 'open')]),
            ('alignment', 'Keypoint alignment', [('Open', lambda: self.open('alignment'), 'open')]),
        )
        for row, (key, title, actions) in enumerate(definitions):
            box = self.indicators[key] = QLabel()
            box.setTextFormat(Qt.TextFormat.RichText)
            self._set_state(key, 'checking')
            steps.addWidget(box, row, 0, Qt.AlignmentFlag.AlignTop)
            steps.addWidget(QLabel(title), row, 1, Qt.AlignmentFlag.AlignTop)
            label = self.labels[key] = QLabel('Checking…')
            label.setWordWrap(True)
            label.setTextFormat(Qt.TextFormat.PlainText)
            steps.addWidget(label, row, 2, Qt.AlignmentFlag.AlignTop)
            buttons = QHBoxLayout()
            self.actions[key] = []
            for text, callback, icon in actions:
                button = self._button(text, callback, icon)
                self.actions[key].append(button)
                buttons.addWidget(button)
            steps.addLayout(buttons, row, 3)
        steps.setColumnStretch(2, 1)
        layout.addLayout(steps)
        self.legend = QLabel(legend_groups_html((LegendGroup(items=tuple(
            (STATE_LABELS[state], WORKFLOW_COLORS[state]) for state in ('available', 'missing', 'review'))),)))
        self.legend.setTextFormat(Qt.TextFormat.RichText)
        self.legend.setWordWrap(True)
        self.legend.setToolTip('Colors describe saved outputs. Editors and processing validate their inputs.')
        layout.addWidget(LegendPanel.containing(self.legend))
        layout.addStretch()
        self.setCentralWidget(body)
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

    def open_console(self):
        console = self.open('console')
        if console is not self.console:
            console.completed.connect(self.refresh)
            self.console = console
        return console

    def revision_action(self, action):
        window = self.open('zones')
        getattr(window, action)()

    def select_video(self):
        from bivosia.ui.dataset_dialogs import select_video_dialog
        if select_video_dialog(self, self.dataset):
            self.refresh()

    def refresh(self):
        if self.task is not None or self._closed:
            return
        dataset = Dataset(self.dataset.directory) if self.dataset.description.is_file() else self.dataset
        self.dataset = dataset
        self.refresh_button.setEnabled(False)
        self.statusBar().showMessage('Checking dataset files…')
        task = self.task = Task(lambda: inspect_workflow(dataset), QApplication.instance())
        task.ready.connect(self._ready)
        task.failed.connect(self.statusBar().showMessage)
        task.finished.connect(self._finished)
        task.finished.connect(task.deleteLater)
        QApplication.instance().aboutToQuit.connect(task.wait)
        self.destroyed.connect(task.requestInterruption)
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
        return f'{count} / {len(entries)} files available'

    def _ready(self, info):
        if self._closed:
            return
        self.info = info
        video = info['video']
        metadata = video['metadata']
        self.labels['video'].setText(video['error'] or (
            f'{metadata.frames:,} frames; {_duration(metadata.duration)}' if metadata and metadata.frames is not None
            else 'Video available' if video['exists'] else 'Select a video'))
        for name in ('key_frames', 'masks', 'analysis', 'features', 'alignment'):
            self.labels[name].setText(self._file_status(info[name]))
        if self._available(info['key_frames']) and info['key_frame_count'] is not None:
            self.labels['key_frames'].setText(f"{info['key_frame_count']} saved key-frame masks")
        self.labels['annotations'].setText(info['annotation_error'] or f"{len(info['annotations'])} saved annotation files")
        exported = info['export']
        export_text = 'No saved export for the selected annotation'
        if exported:
            export_text = exported['error'] or (f"Saved export: {exported['path'].name}" if exported['exists']
                                               else 'The saved export file is missing')
            if exported['exists'] and not exported['error'] and not info['export_current']:
                export_text = 'Annotations changed; export again'
        self.labels['export'].setText(info['export_error'] or export_text)
        has_video = video['exists'] and not video['error']
        has_masks = self._available(info['masks'])
        has_analysis = self._available(info['analysis'])
        has_key_frames = self._available(info['key_frames']) and bool(info['key_frame_count'])
        has_export = bool(exported and exported['exists'] and not exported['error']
                          and not info['export_error'] and info['export_current'])
        for key, available in (('video', has_video), ('key_frames', has_key_frames),
                               ('masks', has_masks), ('analysis', has_analysis), ('export', has_export)):
            self._set_state(key, 'available' if available else 'missing', self.labels[key].text())
        annotation_state = 'available' if has_export else 'review' if info['annotations'] else 'missing'
        self._set_state('annotations', annotation_state,
                        'Current export matches the saved annotation.' if has_export else self.labels['annotations'].text())
        for key in ('key_frames', 'masks'):
            self.actions[key][0].setEnabled(bool(has_video))
        self.actions['analysis'][0].setEnabled(has_masks)
        self.actions['analysis'][1].setEnabled(bool(has_video and has_masks and has_analysis))
        for key in ('annotations', 'export'):
            for button in self.actions[key]:
                button.setEnabled(bool(has_video and has_masks and has_analysis))
        for key in ('features', 'alignment'):
            self._set_state(key, 'available' if self._available(info[key]) else 'missing', self.labels[key].text())
        self.actions['features'][0].setEnabled(has_export)
        self.actions['alignment'][0].setEnabled(bool(has_masks and info['keypoints']['exists']
                                                   and not info['keypoints']['error'] and info['annotations']))
        if not has_video:
            next_step = 'Select the experimental video.'
        elif not has_masks:
            next_step = ('Open VOS to check key frames and run inference.'
                         if self._available(info['key_frames']) and info['key_frame_count']
                         else 'Annotate and save key frames.')
        elif not has_analysis:
            next_step = 'Run mask analysis.'
        else:
            next_step = 'Open ZoD annotation and linking to review or export.'
        self.statusBar().showMessage(f'Suggested next step: {next_step}')

    def _finished(self):
        self.task = None
        self.refresh_button.setEnabled(True)

    def run_analysis(self):
        if self.dataset.description.is_file():
            self.dataset = Dataset(self.dataset.directory)
        require_idle(self.dataset)
        default = int(self.info.get('animals') or 2) if self.info else 2
        animals, accepted = QInputDialog.getInt(self, 'Mask analysis', 'Animals', default, 2, 2147483647)
        if not accepted:
            return
        if self.info and any(self.dataset.array(name).exists() for name in self.info['analysis']):
            answer = QMessageBox.question(self, 'Mask analysis', 'Replace the existing mask analysis results?',
                                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                                          QMessageBox.StandardButton.Cancel)
            if answer != QMessageBox.StandardButton.Yes:
                return
        if not self.dataset.description.is_file():
            self.dataset.save()
        job = submit(self.dataset, 'quantify', {'n_animals': animals})
        self.open_console().select_job(job.id)

    def closeEvent(self, event):
        self._closed = True
        if self.task is not None:
            self.task.requestInterruption()
        event.accept()
