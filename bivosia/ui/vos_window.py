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
import threading
import uuid

import cv2
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
    QVBoxLayout, QWidget,
)

from bivosia.ui.icons import action_icon
from bivosia.analysis.zone_detection import boolean_runs
from bivosia.config import JOB_TERMINAL_STATES, VOS_BACKENDS
from bivosia.dataset.jobs import Job, list_jobs, write_script
from bivosia.dataset.layout import Dataset
from bivosia.dataset.key_frames import KeyFrameSession
from bivosia.dataset.masks import open_masks
from bivosia.dataset.video import VideoSet
from bivosia.processing.jobs import inspect, submit
from bivosia.processing.segmentation import check
from bivosia.ui.colors import KEY_FRAME_COLORS, bgr_colors, identity_colors
from bivosia.ui.dock_panel import PanelHost, PanelSplitter
from bivosia.ui.frame_previews import FramePreviews
from bivosia.ui.model_install_dialog import ModelInstallDialog
from bivosia.ui.parameters import ParameterForm
from bivosia.ui.player import FramePlayer
from bivosia.ui.plot_models import CursorSpec, IntervalSpec, IntervalTrack, LegendGroup, RowSpec
from bivosia.ui.remote_target import RemoteTarget
from bivosia.ui.task import Task
from bivosia.ui.video_panels import (
    MaskDisplayControl, VideoViewGroup, ZoomableVideoLabel, image_from_bgr,
)
from bivosia.ui.zone_plot import ZonePanel, ZoneTimeline


class VosWindow(PanelHost):
    transfer_progress = Signal(object)

    def __init__(self, dataset=None):
        super().__init__()
        self.statusBar()
        self.setWindowTitle('VOS')
        self.resize(1400, 900)
        self.dataset = self.videos = self.key_frames = self.task = self.key_frame_editor = None
        self.masks, self.job_id, self.offset = {}, None, 0
        self.job_remote = None
        self.job_dataset = None
        self.key_frame_queue = []
        self._checkpoints, self._checkpoint_target = {}, None
        self._preparing = False
        self._transfer_cancel = threading.Event()
        self._transfer_stage = None
        self.transfer_progress.connect(self.show_transfer_progress)
        split = PanelSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(split)
        sidebar, self.settings = QWidget(), QWidget()
        side = QVBoxLayout(sidebar)
        form = QFormLayout(self.settings)
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
        self.target = QComboBox()
        self.target.addItems(['Local', 'SSH'])
        self.target.currentIndexChanged.connect(self.target_changed)
        form.addRow('Run on', self.target)
        self.remote_panel = RemoteTarget()
        self.remote_panel.changed.connect(self.update_checkpoint)
        form.addRow(self.remote_panel)
        self.model = QComboBox()
        for name in VOS_BACKENDS:
            self.model.addItem(VOS_BACKENDS[name]['label'], name)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model, 1)
        install = QPushButton('Install')
        install.clicked.connect(self.install_model)
        model_row.addWidget(install)
        form.addRow('Model', model_row)
        backend = VOS_BACKENDS[self.model.currentData()]
        self.checkpoint = QLineEdit(str(backend['checkpoint']))
        checkpoint_row = QHBoxLayout()
        checkpoint_row.addWidget(self.checkpoint, 1)
        self.browse_checkpoint = QPushButton('Open')
        self.browse_checkpoint.setIcon(action_icon('open'))
        self.browse_checkpoint.clicked.connect(self.choose_checkpoint)
        checkpoint_row.addWidget(self.browse_checkpoint)
        form.addRow('Checkpoint', checkpoint_row)
        self.device = QComboBox()
        self.device.setEditable(True)
        self.device.addItems(['auto', 'cuda', 'cpu', 'mps'])
        form.addRow('Device', self.device)
        self.direction = QComboBox()
        for label, value in (('Both', 'both'), ('Forward', 'forward'), ('Reverse', 'reverse')):
            self.direction.addItem(label, value)
        form.addRow('Direction', self.direction)
        self.references = QComboBox()
        for label, value in (('Start key frame', 'start'), ('Both ends', 'ends'), ('All key frames', 'all')):
            self.references.addItem(label, value)
        self.references.setToolTip('Both ends / All key frames require consistent animal IDs across references')
        for index in range(self.references.count()):
            self.references.model().item(index).setEnabled(self.references.itemData(index) in backend['reference_modes'])
        form.addRow('References', self.references)
        self.parameter_layout = QVBoxLayout()
        self.parameter_layout.setContentsMargins(0, 0, 0, 0)
        self.parameters = ParameterForm(backend.get('options', ()), backend['defaults'], backend.get('help'))
        self.parameter_layout.addWidget(self.parameters)
        form.addRow(self.parameter_layout)
        self.model.currentIndexChanged.connect(self.model_changed)
        self.scale = QComboBox()
        for label, value in (('Native', 1.), ('Half', .5), ('Quarter', .25)):
            self.scale.addItem(label, value)
        form.addRow('Output size', self.scale)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.settings)
        side.addWidget(scroll, 2)
        row = QHBoxLayout()
        for label, callback in (('Check', self.check), ('Run', self.start), ('Export script', self.export_script)):
            button = QPushButton(label)
            button.clicked.connect(callback)
            row.addWidget(button)
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
        row = QHBoxLayout()
        self.jobs = QComboBox()
        self.jobs.activated.connect(self.select_job)
        row.addWidget(self.jobs, 1)
        refresh = QPushButton('Jobs')
        refresh.clicked.connect(self.refresh_jobs)
        row.addWidget(refresh)
        side.addLayout(row)
        row = QHBoxLayout()
        self.stop_button = QPushButton('Stop')
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.cancel)
        row.addWidget(self.stop_button)
        self.fetch_button = QPushButton('Fetch results')
        self.fetch_button.setEnabled(False)
        self.fetch_button.clicked.connect(self.fetch)
        row.addWidget(self.fetch_button)
        side.addLayout(row)
        split.addWidget(sidebar)
        right = PanelSplitter(Qt.Orientation.Vertical)
        split.addWidget(right)
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
        status = QWidget()
        status_layout = QVBoxLayout(status)
        self.progress = QProgressBar()
        self.status = QLabel('Ready')
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        status_layout.addWidget(self.progress)
        status_layout.addWidget(self.status)
        status_layout.addWidget(self.log)
        right.addWidget(status)
        split.setSizes([360, 1040])
        right.setSizes([580, 160, 130])
        self.timer = QTimer(self)
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self.poll)
        self.target_changed()
        if dataset:
            self.load_dataset(dataset)

    def error(self, message):
        self.status.setText(str(message))
        QMessageBox.warning(self, 'VOS', str(message))

    def background(self, function, ready, *, quiet=False):
        if self.task is not None:
            return False
        self.task = Task(function, self)
        self.task.ready.connect(ready)
        self.task.failed.connect((lambda message: self.status.setText('Status unavailable · ' + message))
                                 if quiet else self.error)
        self.task.finished.connect(self.task_finished)
        self.task.start()
        return True

    def task_finished(self):
        self.task.deleteLater()
        self.task = None
        if self._preparing:
            self._preparing = False
            if not self.timer.isActive():
                self.settings.setEnabled(True)
                self.stop_button.setEnabled(False)
        QTimer.singleShot(0, self.next_thumbnail)

    def prepare_transfer(self):
        self._preparing = True
        self._transfer_stage = None
        self._transfer_cancel.clear()
        self.settings.setEnabled(False)
        self.stop_button.setEnabled(True)

    def show_transfer_progress(self, progress):
        stage, done, total = progress['stage'], progress['completed'], progress['total']
        self.progress.setRange(0, 1000 if total else 0)
        if total:
            self.progress.setValue(round(1000 * done / total))
        self.status.setText(stage + (f' · {100 * done / total:.0f}%' if total else '…'))
        if stage != self._transfer_stage:
            self.log.appendPlainText(stage)
            self._transfer_stage = stage

    def target_changed(self):
        if not hasattr(self, 'remote_panel'):
            return
        remote = self.target.currentText() == 'SSH'
        self.remote_panel.setVisible(remote)
        if hasattr(self, 'browse_checkpoint'):
            self.browse_checkpoint.setEnabled(not remote)
        if hasattr(self, 'fetch_button'):
            self.fetch_button.setEnabled(remote and self.job_id is not None)
        self.update_checkpoint()

    def remote(self):
        return self.remote_panel.remote()

    def update_checkpoint(self):
        if not hasattr(self, 'checkpoint'):
            return
        model = self.model.currentData()
        profile = self.remote_panel.profile if self.target.currentText() == 'SSH' else None
        target = ((profile['host'], profile['python'], profile['code']) if profile else
                  (self.target.currentText(),), model)
        if target == self._checkpoint_target:
            return
        if self._checkpoint_target is not None:
            self._checkpoints[self._checkpoint_target] = self.checkpoint.text()
        if profile:
            default = profile['checkpoints'][model]
        elif self.target.currentText() == 'SSH':
            default = ''
        else:
            default = str(VOS_BACKENDS[model]['checkpoint'])
        self.checkpoint.setText(self._checkpoints.get(target, default))
        self._checkpoint_target = target

    def install_model(self):
        try:
            remote = self.remote_panel.remote(require_dataset=False) if self.target.currentText() == 'SSH' else None
            dialog = ModelInstallDialog(VOS_BACKENDS, self.model.currentData(), remote=remote, parent=self)
            dialog.exec()
        except Exception as error:
            self.error(error)

    def options(self):
        if not self.checkpoint.text().strip():
            raise ValueError('Specify a checkpoint')
        return dict(model=self.model.currentData(), checkpoint=self.checkpoint.text().strip(),
                    device=self.device.currentText(), parameters=self.parameters.values,
                    direction=self.direction.currentData(), references=self.references.currentData(),
                    output_scale=self.scale.currentData())

    def model_changed(self):
        backend = VOS_BACKENDS[self.model.currentData()]
        self.update_checkpoint()
        self.parameter_layout.removeWidget(self.parameters)
        self.parameters.deleteLater()
        self.parameters = ParameterForm(backend.get('options', ()), backend['defaults'], backend.get('help'))
        self.parameter_layout.addWidget(self.parameters)
        for index in range(self.references.count()):
            self.references.model().item(index).setEnabled(self.references.itemData(index) in backend['reference_modes'])
        self.references.setCurrentIndex(0)

    def choose_dataset(self):
        path = QFileDialog.getExistingDirectory(self, 'Dataset', self.path.text())
        if path:
            self.load_dataset(Dataset(path))

    def choose_checkpoint(self):
        suffixes = VOS_BACKENDS[self.model.currentData()]['checkpoint_suffixes']
        filters = 'Weights (' + ' '.join('*' + suffix for suffix in suffixes) + ')'
        path, _ = QFileDialog.getOpenFileName(self, 'Checkpoint', self.checkpoint.text(), filters)
        if path:
            self.checkpoint.setText(path)

    def load_dataset(self, dataset):
        if self.timer.isActive():
            self.status.setText('Wait for the current job before switching datasets')
            return
        if self.task is not None:
            self.status.setText('Wait for the current operation')
            return
        videos = None
        try:
            videos = VideoSet({'original': dataset.video})
            image = videos.read('original', 0)
            key_frames = KeyFrameSession(dataset, videos.frame_count, image.shape[:2])
            self.close_video()
            self.dataset, self.videos, self.key_frames = dataset, videos, key_frames
            self.remote_panel.set_dataset(dataset.directory)
            self.videos.prefetch('original', [])
            self.path.setText(str(dataset.directory))
            self.setWindowTitle('VOS · ' + dataset.directory.name)
            for source in self.panes:
                path = dataset.mask(source)
                if path.exists():
                    self.masks[source] = open_masks(path, require_valid=False, direction=source)
            self.player.configure(videos.frame_count, videos.source_fps)
            self.player.set_loop(*key_frames.crop)
            self.timeline.frame_count = videos.frame_count
            self.timeline.reset_zoom()
            self.previews.images.clear()
            self.previews.set_frames(key_frames.frames, key_frames.masks, key_frames.crop,
                                     bgr_colors(identity_colors(key_frames.animals, .27)))
            self.key_frame_queue = list(key_frames.frames)
            self.timeline_rows()
            self.seek(key_frames.crop[0])
            self.next_thumbnail()
        except Exception as error:
            if videos is not None and videos is not self.videos:
                videos.close()
            self.error(error)

    def next_thumbnail(self):
        if not self.key_frame_queue or self.task is not None or self.videos is None:
            return
        frame = self.key_frame_queue.pop(0)

        def read():
            image = self.videos.read('original', frame)
            ratio = min(160 / image.shape[1], 100 / image.shape[0])
            return frame, cv2.resize(image, (max(1, round(image.shape[1] * ratio)),
                                             max(1, round(image.shape[0] * ratio))))
        self.background(read, lambda result: self.previews.set_image(*result))

    def timeline_rows(self, current=None):
        if self.key_frames is None:
            return
        legend = LegendGroup(items=tuple((label, KEY_FRAME_COLORS[key]) for key, label in (
            ('proposed', 'Unlabeled'), ('labeled', 'Labeled'), ('boundary', 'Boundary'))))
        rows = [RowSpec('key_frames', 'Key frames', legend=(legend,))]
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
        cursors = [CursorSpec('playhead', self.player.frame)]
        for frame in self.key_frames.frames:
            color = KEY_FRAME_COLORS['labeled' if frame in self.key_frames.masks else 'proposed']
            cursors.append(CursorSpec(f'key_frame:{frame}', frame, color, line=False, level=1, draggable=False,
                                      outline=KEY_FRAME_COLORS['boundary'] if frame in self.key_frames.crop else None))
        self.timeline.set_cursors(cursors)

    def seek(self, frame, playback=False):
        if self.videos is None:
            return
        try:
            if playback:
                self.videos.prefetch('original', range(frame, min(frame + 12, self.videos.frame_count)))
            image = self.videos.read('original', frame)
            colors = bgr_colors(identity_colors(self.key_frames.animals, .27))
            for source, pane in self.panes.items():
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
                rendered = (
                    self.mask_display.background(image)
                    if mask is None else self.mask_display.composite(
                        image, mask, None, colors
                    )
                )
                pane.set_image(image_from_bgr(rendered))
                self.titles[source].setText(label)
            self.player.set_frame(frame)
            self.timeline.set_current_frame(frame)
            self.previews.select(frame)
        except Exception as error:
            self.player.stop()
            self.statusBar().showMessage(str(error))

    def check(self):
        if self.task is not None:
            return
        try:
            options = self.options()
            remote = self.remote() if self.target.currentText() == 'SSH' else None
            if self.dataset is None:
                raise ValueError('Open a dataset')
            if remote:
                self.prepare_transfer()

                def preflight():
                    plan = remote.sync(self.dataset, self.transfer_progress.emit,
                                       self._transfer_cancel.is_set, dry_run=True)
                    model = remote.call('check_model', settings=options)
                    return {**model, 'files_to_upload': len(plan['missing'])}

                self.background(preflight, lambda result: self.status.setText(
                    f"{result['device']} · {result['files_to_upload']} files to upload when Run starts"))
            else:
                self.background(lambda: check(self.dataset, **options),
                                lambda result: self.status.setText('Ready · ' + json.dumps(result)))
        except Exception as error:
            self.error(error)

    def start(self):
        if self.task is not None:
            self.status.setText('Wait for the current operation')
            return
        if self.job_id and self.timer.isActive():
            self.error('A job is already being monitored; stop it or wait for completion')
            return
        try:
            options = self.options()
            remote = self.remote() if self.target.currentText() == 'SSH' else None
            if self.dataset is None:
                raise ValueError('Open a dataset')
            job_id = uuid.uuid4().hex
            self.job_remote, self.job_id, self.offset = remote, job_id, 0
            self.job_dataset = self.dataset
            self.jobs.clear()
            self.jobs.addItem(job_id)
            self.log.clear()
            if remote:
                self.prepare_transfer()

            def launch():
                if remote:
                    remote.sync(self.job_dataset, self.transfer_progress.emit, self._transfer_cancel.is_set)
                    if self._transfer_cancel.is_set():
                        raise InterruptedError('Remote run cancelled before submission')
                    remote.call('check', settings=options)
                    if self._transfer_cancel.is_set():
                        raise InterruptedError('Remote run cancelled before submission')
                    return remote.call('submit', operation='vos', settings=options, job_id=job_id)
                check(self.job_dataset, **options)
                return inspect(submit(self.job_dataset, 'vos', options, job_id=job_id))

            self.background(launch, self.started)
        except Exception as error:
            self.error(error)

    def started(self, result):
        self.settings.setEnabled(False)
        self.timer.start()
        self.update_job(result)

    def poll(self):
        if self.job_id and self.task is None:
            remote, job_id, offset = self.job_remote, self.job_id, self.offset
            self.background(lambda: remote.call('inspect', job_id=job_id, offset=offset) if remote else
                            inspect(Job(self.job_dataset, job_id), offset), self.update_job, quiet=True)

    def update_job(self, response):
        self.job_id, self.offset = response['id'], response['offset']
        status = response['status']
        if response['log']:
            self.log.appendPlainText(response['log'].rstrip())
        done, total = status.get('completed', 0), status.get('total', 0)
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        elapsed = status.get('elapsed', 0)
        eta = elapsed * (total - done) / done if done else None
        text = f"{status['state'].title()} · {done:,}/{total:,} frames · {elapsed / 60:.1f} min"
        if status.get('direction'):
            text += f" · {status['direction']} · frame {status.get('frame', 0):,}"
        if eta is not None and status['state'] == 'running':
            text += f' · ~{eta / 60:.1f} min left'
        if status.get('error'):
            text += ' · ' + status['error']
        self.status.setText(text)
        self.status.setWordWrap(True)
        self.timeline_rows(status)
        ended = status['state'] in JOB_TERMINAL_STATES
        self.settings.setEnabled(ended)
        self.stop_button.setEnabled(not ended)
        self.fetch_button.setEnabled(status['state'] == 'complete' and self.job_remote is not None)
        if ended:
            self.timer.stop()
            self.settings.setEnabled(True)
            if status['state'] == 'complete' and self.job_remote is None:
                QTimer.singleShot(100, self.reload)

    def cancel(self):
        if self._preparing:
            self._transfer_cancel.set()
            self.status.setText('Stopping input preparation…')
            return
        if self.job_id is None:
            return
        remote, job_id = self.job_remote, self.job_id
        self.background(lambda: remote.call('cancel', job_id=job_id) if remote else Job(self.job_dataset, job_id).cancel(),
                        lambda _: self.status.setText('Cancellation requested'))

    def refresh_jobs(self):
        remote = self.remote() if self.target.currentText() == 'SSH' else None
        if remote is None and self.dataset is None:
            return

        def loaded(response):
            self.jobs.clear()
            self.jobs.addItems(response['jobs'])
        self.background(lambda: remote.call('list', operation='vos') if remote else
                        {'jobs': list_jobs(self.dataset, operation='vos')}, loaded)

    def select_job(self, index):
        self.job_remote = self.remote() if self.target.currentText() == 'SSH' else None
        self.job_dataset = self.dataset
        self.job_id, self.offset = self.jobs.itemText(index), 0
        self.log.clear()
        self.timer.start()
        self.poll()

    def fetch(self):
        if not self.job_remote or not self.job_id or not self.dataset:
            self.error('Open the local destination dataset and select a completed remote job')
            return
        self.status.setText('Fetching completed masks; existing local files will not be replaced')
        remote, job_id, destination = self.job_remote, self.job_id, self.dataset
        self.background(lambda: remote.fetch(job_id, destination),
                        lambda _: QTimer.singleShot(100, self.reload))

    def export_script(self):
        try:
            remote = self.remote() if self.target.currentText() == 'SSH' else None
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
            self.seek(frame)

    def close_video(self):
        self.player.stop()
        for handle in self.masks.values():
            handle.close()
        self.masks.clear()
        if self.videos is not None:
            self.videos.close()
        self.videos = None

    def closeEvent(self, event):
        if self.task is not None:
            self.status.setText('Wait for the current operation before closing')
            event.ignore()
            return
        self.remote_panel.remember()
        self.timer.stop()
        self.key_frame_queue.clear()
        self.close_video()
        event.accept()
