"""Run video-object segmentation locally or over SSH and inspect its masks.

Examples
--------
With an existing dataset:

>>> from PySide6.QtWidgets import QApplication
>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.ui.vos_window import VosWindow
>>> app = QApplication.instance() or QApplication([])  # doctest: +SKIP
>>> dataset = Dataset("/path/to/dataset")
>>> window = VosWindow(dataset)  # doctest: +SKIP
>>> window.show()  # doctest: +SKIP
>>> app.exec()  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path
import json

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QScrollArea,
    QVBoxLayout, QWidget,
)

from bivosia.ui.icons import action_icon
from bivosia.analysis.zones_of_disagreement import boolean_runs
from bivosia.config import JOB_TERMINAL_STATES
from bivosia.dataset.jobs import write_script
from bivosia.dataset.layout import Dataset
from bivosia.dataset.key_frame_recording import KeyFrameRecording
from bivosia.dataset.masks import open_masks
from bivosia.ui.processing_request import ProcessingRequest
from bivosia.processing.segmentation import check
from bivosia.ui.colors import bgr_colors, identity_colors
from bivosia.ui.dock_panel import PanelHost, PanelSplitter
from bivosia.ui.frame_previews import FramePreviews
from bivosia.ui.player import FramePlayer
from bivosia.ui.plot_models import IntervalSpec, IntervalTrack, RowSpec
from bivosia.ui.key_frame_view import configure_key_frame_view, key_frame_cursors, key_frame_row
from bivosia.ui.processing_panel import ProcessingPanel
from bivosia.ui.vos_parameters import VosParameters
from bivosia.ui.task import Task
from bivosia.ui.video_panels import (
    MaskDisplayControl, VideoViewGroup, ZoomableVideoLabel, image_from_bgr,
)
from bivosia.ui.zone_plot import ZonePanel, ZoneTimeline


class VosWindow(PanelHost):
    def __init__(self, dataset=None):
        super().__init__()
        self.statusBar()
        self.setWindowTitle('VOS')
        self.resize(1400, 900)
        self.dataset = self.videos = self.key_frames = self.task = self.key_frame_editor = None
        self.recording = None
        self.masks, self.job_id = {}, None
        self._job_active = False
        self.job_remote = None
        self.job_dataset = None
        self.request = None
        self.remote_reader = None
        self._pending_seek = None
        split = PanelSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(split)
        sidebar = QWidget()
        self.processing = ProcessingPanel(dataset or Dataset(Path.cwd()), operation='vos')
        self.settings = VosParameters(dataset or Dataset(Path.cwd()), self.processing.remote)
        self.processing.changed.connect(self.settings.target_changed)
        self.processing.monitor.updated.connect(self.update_job)
        self.processing.monitor.completed.connect(self._job_finished)
        side = QVBoxLayout(sidebar)
        form = QFormLayout()
        side.addLayout(form)
        form.setContentsMargins(0, 0, 0, 0)
        self.path = QLineEdit()
        self.path.setReadOnly(True)
        row = QHBoxLayout()
        row.addWidget(self.path, 1)
        button = QPushButton('Open')
        button.setIcon(action_icon('open'))
        button.clicked.connect(self.choose_dataset)
        row.addWidget(button)
        form.addRow('Dataset', row)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.settings)
        side.addWidget(scroll, 2)
        row = QHBoxLayout()
        for label, callback in (('Check', self.check), ('Export script', self.export_script)):
            button = QPushButton(label)
            button.clicked.connect(callback)
            row.addWidget(button)
        self.run_button = QPushButton('Run')
        self.run_button.setIcon(action_icon('run'))
        self.run_button.clicked.connect(self.start)
        row.addWidget(self.run_button)
        side.addLayout(row)
        self.previews = FramePreviews()
        self.previews.frameSelected.connect(lambda frame: self.player.seek(frame))
        side.addWidget(self.previews, 1)
        row = QHBoxLayout()
        for label, callback in (('Edit key frames', self.edit_key_frames), ('Reload', self.reload)):
            button = QPushButton(label)
            button.clicked.connect(callback)
            row.addWidget(button)
        side.addLayout(row)
        split.addWidget(sidebar)
        panel_host = PanelHost()
        split.addWidget(panel_host)
        right = PanelSplitter(Qt.Orientation.Vertical)
        panel_host.setCentralWidget(right)
        video = QWidget()
        layout = QVBoxLayout(video)
        self.player = FramePlayer('frame', '#ff3b4f', selectable=False)
        self.player.frameRequested.connect(self.seek)
        layout.addWidget(self.player)
        row = QHBoxLayout()
        row.addStretch()
        self.mask_display = MaskDisplayControl()
        self.mask_display.valueChanged.connect(lambda: self.seek(self.player.frame))
        row.addWidget(self.mask_display)
        reset = QPushButton('Reset zoom')
        reset.clicked.connect(lambda: self.navigation.reset())
        row.addWidget(reset)
        layout.addLayout(row)
        panes = QHBoxLayout()
        self.panes, self.titles = {}, {}
        for source in ('forward', 'reverse'):
            column = QVBoxLayout()
            self.titles[source] = QLabel(source.title())
            self.panes[source] = ZoomableVideoLabel()
            column.addWidget(self.titles[source])
            column.addWidget(self.panes[source], 1)
            panes.addLayout(column)
        self.navigation = VideoViewGroup(list(self.panes.values()))
        layout.addLayout(panes, 1)
        right.addWidget(video)
        self.timeline = ZoneTimeline(2, 'Frames')
        self.timeline.frameClicked.connect(self.player.seek)
        self.timeline.cursorMoved.connect(lambda key, frame: self.player.seek(frame) if key == 'playhead' else None)
        right.addWidget(ZonePanel(self.timeline))
        self.processing.dock(panel_host)
        self.status = self.processing.monitor.job_status
        split.setSizes([360, 1040])
        right.setSizes([580, 160])
        self.settings.setEnabled(dataset is not None)
        if dataset:
            self.load_dataset(dataset)

    def error(self, message):
        self.status.setText(str(message))
        QMessageBox.warning(self, 'VOS', str(message))

    def background(self, function, ready, *, quiet=False):
        if self.task is not None:
            return False
        self.task = Task(function, self)
        self.task.connect_callbacks(ready=ready, finished=self.task_finished,
            failed=(lambda message: self.status.setText('Status unavailable · ' + message)) if quiet else self.error)
        self.task.start()
        return True

    def task_finished(self):
        self.task = None
        if self._pending_seek is not None:
            frame, self._pending_seek = self._pending_seek, None
            QTimer.singleShot(0, self, lambda: self.seek(frame))
        QTimer.singleShot(0, self.next_thumbnail)

    def options(self):
        return self.settings.options()

    def choose_dataset(self):
        path = QFileDialog.getExistingDirectory(self, 'Dataset', self.path.text())
        if path:
            self.load_dataset(Dataset(path))

    def load_dataset(self, dataset):
        if self._job_active:
            self.status.setText('Wait for the current job before switching datasets')
            return
        if self.task is not None or self.request is not None:
            self.status.setText('Wait for the current operation')
            return
        from bivosia.dataset.remote_reader import RemoteDatasetReader
        from bivosia.processing.remote_inventory import inspect_locations
        from bivosia.ui.dataset_access import location_preferences
        profiles, preferred = location_preferences(dataset)
        def load():
            recording = KeyFrameRecording(dataset)
            reader, masks = None, {}
            try:
                info = inspect_locations(dataset, profiles, preferred)
                if info['mask_remote'] is not None:
                    reader = recording.manage(RemoteDatasetReader(info['mask_remote']))
                elif info['remote_errors']:
                    raise ValueError('\n'.join(info['remote_errors']))
                for source in ('forward', 'reverse'):
                    if reader:
                        masks[source] = reader.open_masks(source)
                    elif dataset.mask(source).is_file():
                        masks[source] = recording.manage(open_masks(dataset.mask(source), require_valid=False, direction=source))
                return recording, reader, masks
            except BaseException:
                recording.close()
                raise
        def loaded(result):
            recording, reader, masks = result
            self.close_video()
            self.recording = recording
            self.dataset, self.videos, self.key_frames = dataset, recording.videos, recording.session
            self.remote_reader, self.masks = reader, masks
            self.processing.set_dataset(dataset)
            self.settings.set_dataset(dataset)
            self.settings.setEnabled(True)
            self.path.setText(str(dataset.directory))
            self.setWindowTitle('VOS · ' + dataset.directory.name)
            configure_key_frame_view(recording, self.player, self.timeline, self.previews)
            self.timeline_rows()
            self._pending_seek = self.key_frames.crop[0]
            self.status.setText('Masks: ' + reader.remote.host if reader else 'Dataset loaded')
        self.background(load, loaded)

    def next_thumbnail(self):
        if self.task is not None or self.videos is None:
            return
        read = self.previews.next_thumbnail()
        if read is not None:
            self.background(read, self.previews.accept_thumbnail)

    def timeline_rows(self, current=None):
        if self.key_frames is None:
            return
        rows = [key_frame_row('Key frames')]
        for source, color in (('forward', '#66b3ff'), ('reverse', '#ff9e66')):
            intervals = ()
            if source in self.masks:
                intervals = tuple(IntervalSpec(f'{source}:{start}', int(start), int(end))
                                  for start, end in boolean_runs(self.masks[source].valid))
            elif current and current.get('direction') == source and 'interval' in current:
                first, last = current['interval']
                frame = current.get('frame', first)
                intervals = (IntervalSpec(source, first if source == 'forward' else frame,
                                          frame if source == 'forward' else last),)
            rows.append(RowSpec(source, source.title(), (IntervalTrack(source, '', intervals, color),)))
        self.timeline.set_rows(rows)
        self.timeline.set_cursors(key_frame_cursors(self.key_frames, self.player.frame))

    def seek(self, frame, playback=False):
        if self.videos is None:
            return
        if self.remote_reader is not None:
            if self.task is not None:
                self._pending_seek = frame
                return
            self.background(lambda: self._read_frame(frame, playback),
                            lambda result: self._show_frame(frame, *result))
            return
        try:
            self._show_frame(frame, *self._read_frame(frame, playback))
        except Exception as error:
            self.player.stop()
            self.statusBar().showMessage(str(error))

    def _read_frame(self, frame, playback):
        if playback:
            self.videos.prefetch('original', range(frame, min(frame + 12, self.videos.frame_count)))
        image = self.videos.read('original', frame)
        outputs = {}
        for source in self.panes:
            mask, label = None, source.title()
            if source in self.masks:
                handle = self.masks[source]
                if frame < handle.shape[0] and handle.valid[frame]:
                    mask = handle[frame]
                else:
                    label += ' · Outside processed range'
            elif frame in self.key_frames.masks:
                mask, label = self.key_frames.masks[frame], label + ' · Key frame'
            else:
                label += ' · No output'
            outputs[source] = mask, label
        return image, outputs

    def _show_frame(self, frame, image, outputs):
        colors = bgr_colors(identity_colors(self.key_frames.animals, .27))
        for source, pane in self.panes.items():
            mask, label = outputs[source]
            rendered = (self.mask_display.background(image) if mask is None else
                        self.mask_display.composite(image, mask, None, colors))
            pane.set_image(image_from_bgr(rendered))
            self.titles[source].setText(label)
        self.player.set_frame(frame)
        self.timeline.set_current_frame(frame)
        self.previews.select(frame)

    def check(self):
        if self.task is not None:
            return
        try:
            options = self.options()
            remote = self.processing.remote()
            if self.dataset is None:
                raise ValueError('Open a dataset')
            self.background(lambda: remote.call('check', settings=options) if remote else check(self.dataset, **options),
                            lambda result: self.status.setText('Ready · ' + json.dumps(result)))
        except Exception as error:
            self.error(error)

    def start(self):
        if self.task is not None or self.request is not None:
            self.status.setText('Wait for the current operation')
            return
        if self.job_id and self._job_active:
            self.error('A job is already being monitored; stop it or wait for completion')
            return
        try:
            options = self.options()
            remote = self.processing.remote()
            if self.dataset is None:
                raise ValueError('Open a dataset')
            self.job_remote = remote
            self.job_dataset = self.dataset
            request = self.request = ProcessingRequest(self, self.dataset, 'vos', options, remote=remote,
                                                        windows=getattr(self, 'dataset_windows', None))
            request.started.connect(self.started)
            request.failed.connect(self.error)
            request.finished.connect(self._request_finished)
            self.settings.setEnabled(False)
            request.start()
        except Exception as error:
            self.error(error)

    def _request_finished(self):
        self.request = None
        self.settings.setEnabled(not self._job_active)

    def started(self, result):
        self.job_id = result['id']
        self.processing.select_job(result, remote=self.job_remote)
        self.update_job(result)

    def update_job(self, response):
        self.job_id = response['id']
        status = response['status']
        self._job_active = status['state'] not in JOB_TERMINAL_STATES
        self.settings.setEnabled(not self._job_active)
        self.processing.target.setEnabled(not self._job_active)
        self.run_button.setEnabled(not self._job_active)
        self.timeline_rows(status)

    def _job_finished(self):
        self._job_active = False
        self.settings.setEnabled(True)
        self.processing.target.setEnabled(True)
        self.run_button.setEnabled(True)
        QTimer.singleShot(100, self.reload)

    def export_script(self):
        try:
            remote = self.processing.remote()
            if remote is None and self.dataset is None:
                raise ValueError('Open a dataset')
            path, _ = QFileDialog.getSaveFileName(self, 'Export script', 'run_vos.py', 'Python (*.py)')
            if not path:
                return
            write_script(path, remote.dataset if remote else self.dataset.directory, self.options(),
                         remote.code if remote else Path(__file__).resolve().parents[1])
            self.status.setText('Exported · ' + path)
        except Exception as error:
            self.error(error)

    def edit_key_frames(self):
        if self.dataset is not None:
            if getattr(self, 'dataset_windows', None) is not None:
                self.dataset_windows.open('key_frames', self.dataset)
                return
            from bivosia.ui.key_frame_editor import KeyFrameWindow
            if self.key_frame_editor is not None and self.key_frame_editor.dataset.directory != self.dataset.directory:
                if not self.key_frame_editor.close():
                    return
                self.key_frame_editor = None
            if self.key_frame_editor is None:
                self.key_frame_editor = KeyFrameWindow(self.dataset)
            self.key_frame_editor.showMaximized()
            self.key_frame_editor.raise_()

    def reload(self):
        if self.dataset is not None and self.task is None:
            frame = self.player.frame
            self.load_dataset(self.dataset)
            self._pending_seek = frame

    def close_video(self):
        self.player.stop()
        self.previews.set_video(None)
        if self.recording is not None:
            self.recording.close()
            self.recording = None
        self.masks.clear()
        self.remote_reader = None
        self.videos = None

    def closeEvent(self, event):
        if self.task is not None or self.request is not None:
            self.status.setText('Wait for the current operation before closing')
            event.ignore()
            return
        self.processing.target.remote_panel.remember()
        self.processing.shutdown()
        self.close_video()
        event.accept()
