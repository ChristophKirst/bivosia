"""Select video key frames and open their mask editor.

Examples
--------
With an existing dataset:

>>> from PySide6.QtWidgets import QApplication
>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.ui.key_frame_editor import KeyFrameWindow
>>> app = QApplication.instance() or QApplication([])  # doctest: +SKIP
>>> dataset = Dataset("/path/to/dataset")
>>> window = KeyFrameWindow(dataset)  # doctest: +SKIP
>>> window.show()  # doctest: +SKIP
>>> app.exec()  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFormLayout, QHBoxLayout, QLabel,
    QMenu, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from bivosia.config import KEY_FRAME_BACKENDS
from bivosia.dataset.annotations.keyframes import KeyFrameRecording
from bivosia.segmentation.image import create_segmenter
from bivosia.ui.colors import bgr_colors, identity_colors
from bivosia.ui.dock_panel import PanelHost, PanelSplitter
from bivosia.ui.frame_previews import FramePreviews
from bivosia.ui.mask_editor import MaskEditor
from bivosia.ui.player import FramePlayer
from bivosia.ui.key_frame_model import KeyFrameModelSelector
from bivosia.ui.key_frame_view import configure_key_frame_view, key_frame_cursors, key_frame_row
from bivosia.ui.processing_panel import ProcessingPanel
from bivosia.ui.task import Task
from bivosia.ui.task_monitor import TaskMonitor
from bivosia.ui.video_panels import (
    MaskDisplayControl, VideoViewGroup, ZoomableVideoLabel, image_from_bgr,
)
from bivosia.ui.zone_plot import ZonePanel, ZoneTimeline
from bivosia.ui.window_manager import notify_dataset_changed


class KeyFrameWindow(PanelHost):
    def __init__(self, dataset, *, model=None, checkpoint=None, device="auto", parameters=None, animals=2):
        super().__init__()
        self.statusBar()
        self.setWindowTitle("Key frames")
        self.resize(1350, 850)
        self.session = self.videos = self.dataset = None
        self.recording = None
        self.selected = None
        self.preview_task = None
        model = model or next(iter(KEY_FRAME_BACKENDS))
        self.mask_editor = MaskEditor(create_segmenter(model, checkpoint, device, parameters))
        self.mask_editor.saved.connect(self._labeled)
        self.mask_editor.navigate.connect(self.advance)
        split = PanelSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(split)
        sidebar = QWidget()
        left = QVBoxLayout(sidebar)
        self.processing = ProcessingPanel(dataset, monitor=TaskMonitor(), require_dataset=False)
        self.mask_editor.predictionStarted.connect(self.processing.monitor.follow)
        self.model_selector = KeyFrameModelSelector(self.processing.target, model, checkpoint, device, parameters)
        left.addWidget(self.model_selector)
        self.mask_editor.busyChanged.connect(self.model_selector.set_busy)
        self.model_selector.changed.connect(self._sync_predictor)
        self._sync_predictor()
        self.settings = QWidget()
        options = QFormLayout(self.settings)
        options.setContentsMargins(0, 0, 0, 0)
        left.addWidget(self.settings)
        self.animals = QSpinBox()
        self.animals.setRange(1, 255)
        self.animals.setValue(animals)
        self.animals.editingFinished.connect(self._animals_changed)
        options.addRow("Animals", self.animals)
        self.crop_start, self.crop_end = QSpinBox(), QSpinBox()
        row = QHBoxLayout()
        row.addWidget(self.crop_start)
        row.addWidget(self.crop_end)
        crop = QPushButton("Apply")
        crop.clicked.connect(self.apply_crop)
        row.addWidget(crop)
        options.addRow("Crop", row)
        self.crop_start.setToolTip("Inclusive first source frame; source video is unchanged")
        self.crop_end.setToolTip("Inclusive last source frame; source video is unchanged")
        self.count = QSpinBox()
        row = QHBoxLayout()
        row.addWidget(self.count)
        propose = QPushButton("Propose")
        propose.clicked.connect(self.propose)
        row.addWidget(propose)
        self.add_button = QPushButton("Add")
        self.add_button.setToolTip("Add the current video frame to the key-frame selection")
        self.add_button.clicked.connect(lambda: self.add_frame(self.player.frame))
        row.addWidget(self.add_button)
        options.addRow("Frames", row)
        self.position = QSpinBox()
        self.position.editingFinished.connect(self.move_selected)
        options.addRow("Selected", self.position)
        self.progress = QLabel()
        left.addWidget(self.progress)
        self.previews = FramePreviews()
        self.previews.frameSelected.connect(lambda frame: self.select_frame(frame, edit=True))
        left.addWidget(self.previews, 1)
        row = QHBoxLayout()
        for text, callback in (("Edit", self.edit), ("Next", lambda: self.advance(1)), ("Save", self.save)):
            button = QPushButton(text)
            button.clicked.connect(callback)
            row.addWidget(button)
        left.addLayout(row)
        sidebar_host = PanelHost()
        sidebar_host.setCentralWidget(sidebar)
        self.processing.dock(sidebar_host)
        split.addWidget(sidebar_host)
        right = PanelSplitter(Qt.Orientation.Vertical)
        split.addWidget(right)
        video = QWidget()
        video_layout = QVBoxLayout(video)
        self.player = FramePlayer("frame", "#ff3b4f", selectable=False)
        self.player.frameRequested.connect(self.seek)
        video_layout.addWidget(self.player)
        row = QHBoxLayout()
        row.addStretch()
        self.mask_display = MaskDisplayControl()
        self.mask_display.valueChanged.connect(lambda: self.seek(self.player.frame))
        row.addWidget(self.mask_display)
        reset = QPushButton("Reset zoom")
        reset.clicked.connect(lambda: self.navigation.reset())
        row.addWidget(reset)
        video_layout.addLayout(row)
        self.pane = ZoomableVideoLabel()
        self.navigation = VideoViewGroup([self.pane])
        video_layout.addWidget(self.pane, 1)
        right.addWidget(video)
        self.timeline = ZoneTimeline(2, "Frames")
        self.timeline.frameClicked.connect(self.player.seek)
        self.timeline.cursorSelected.connect(self._cursor_selected)
        self.timeline.cursorMoved.connect(self._cursor_moved)
        self.timeline.cursorReleased.connect(self._cursor_released)
        self.timeline.frameContextRequested.connect(self._frame_menu)
        right.addWidget(ZonePanel(self.timeline))
        split.setSizes([340, 1000])
        right.setSizes([650, 130])
        self.settings.setEnabled(False)
        self.load_dataset(dataset)

    def error(self, error):
        QMessageBox.warning(self, "Key frames", str(error))

    def _sync_predictor(self):
        parameters = self.model_selector.parameters
        if parameters is not None:
            model, path, device = parameters
            predictor = self.mask_editor.predictor
            remote = self.model_selector.remote
            if ((predictor.name, str(predictor.checkpoint), predictor.device) != (model, str(path), device)
                    or getattr(predictor, 'remote', None) != remote):
                self.mask_editor.set_predictor(create_segmenter(model, path, device, self.model_selector.overrides.get(model),
                                                                remote=remote))
        elif not self.mask_editor.busy:
            self.mask_editor.predictor.close()
        self.mask_editor.set_availability(self.model_selector.reason)

    def load_dataset(self, dataset):
        if not self.can_leave():
            return False
        try:
            recording = KeyFrameRecording(dataset, animals=self.animals.value())
        except Exception as error:
            self.error(error)
            return False
        self.mask_editor.hide()
        self.mask_editor.session = self.mask_editor.draft = None
        self.player.stop()
        if self.recording is not None:
            self.recording.close()
        self.recording = recording
        session = recording.session
        self.dataset, self.videos, self.session = dataset, recording.videos, session
        self.processing.set_dataset(dataset)
        self.setWindowTitle(f"Key frames · {dataset.directory.name}")
        configure_key_frame_view(recording, self.player, self.timeline, self.previews)
        for spin in (self.crop_start, self.crop_end, self.position):
            spin.setRange(0, session.frame_count - 1)
        self.selected = session.frames[0]
        self.settings.setEnabled(True)
        self.refresh()
        self.seek(self.selected)
        return True

    def refresh(self):
        if self.session is None:
            return
        session = self.session
        self.crop_start.setValue(session.crop[0])
        self.crop_end.setValue(session.crop[1])
        self.animals.setValue(session.animals)
        fixed = session.fixed_frames
        self.count.setRange(len(fixed), session.crop[1] - session.crop[0] + 1)
        self.count.setValue(len(session.frames))
        self.player.set_loop(*session.crop)
        if self.selected not in session.frames:
            self.selected = session.frames[0]
        if self.mask_editor.frame not in session.frames:
            self.mask_editor.hide()
            self.mask_editor.draft = self.mask_editor.session = None
        self.position.setValue(self.selected)
        self.position.setEnabled(self.selected not in fixed)
        self.position.setToolTip("Labeled frames and crop boundaries are fixed" if self.selected in fixed
                                 else "Move this unlabeled proposal")
        self.progress.setText(f"{sum(f in session.masks for f in session.frames)} / {len(session.frames)} labeled")
        self.previews.set_session(session)
        self.previews.select(self.selected)
        self.timeline.set_rows([key_frame_row('Frames')])
        self._cursors()
        self.add_button.setEnabled(self._can_add(self.player.frame))
        self._next_preview()

    def _cursors(self):
        self.timeline.set_cursors(key_frame_cursors(self.session, self.player.frame, editable=True))
        self.timeline.set_active_cursor(f"key_frame:{self.selected}")

    def _next_preview(self):
        if self.preview_task is not None:
            return
        read_thumbnail = self.previews.next_thumbnail()
        if read_thumbnail is None:
            return
        self.preview_task = Task(read_thumbnail, self)
        self.preview_task.connect_callbacks(ready=self.previews.accept_thumbnail,
                                            failed=self._preview_error, finished=self._preview_finished)
        self.preview_task.start()

    def _preview_error(self, message):
        self.previews.clear_pending()
        self.statusBar().showMessage(message)

    def _preview_finished(self):
        self.preview_task = None
        QTimer.singleShot(0, self._next_preview)

    def seek(self, frame, playback=False):
        if self.videos is None:
            return
        try:
            if playback:
                self.videos.prefetch("original", range(frame, min(frame + 12, self.videos.frame_count)))
            image = self.videos.read("original", frame)
            if frame in self.session.masks:
                image = self.mask_display.composite(
                    image, self.session.masks[frame], None,
                    bgr_colors(identity_colors(self.session.animals, .27)),
                )
            else:
                image = self.mask_display.background(image)
            self.pane.set_image(image_from_bgr(image))
            self.player.set_frame(frame)
            self.timeline.set_current_frame(frame)
            self.add_button.setEnabled(self._can_add(frame))
        except Exception as error:
            self.player.stop()
            self.statusBar().showMessage(str(error))

    def select_frame(self, frame, *, edit=False):
        if self.session is None or frame not in self.session.frames:
            return
        if frame != self.selected and not self.mask_editor.can_leave():
            self.previews.select(self.selected)
            return
        self.selected = frame
        self.player.seek(frame)
        self.position.setValue(frame)
        self.position.setEnabled(frame not in self.session.crop and frame not in self.session.masks)
        self.previews.select(frame)
        self._cursors()
        if edit:
            self.edit()

    def _cursor_selected(self, key):
        if key.startswith("key_frame:"):
            self.select_frame(int(key.split(":")[1]))

    def _cursor_moved(self, key, frame):
        if key.startswith("key_frame:"):
            self.player.seek(frame)

    def _cursor_released(self, key, frame):
        if key.startswith("key_frame:"):
            self._move(int(key.split(":")[1]), frame)

    def move_selected(self):
        if self.session is not None and self.position.value() != self.selected:
            self._move(self.selected, self.position.value())

    def _move(self, source, target):
        if source != target:
            if not self.mask_editor.can_leave():
                self.refresh()
                return
            try:
                self.session.move(source, target)
                self.selected = target
            except ValueError as error:
                self.error(error)
        self.refresh()
        self.player.seek(self.selected)

    def propose(self):
        if self.session is None or not self.mask_editor.can_leave():
            return
        try:
            self.session.propose(self.count.value())
        except ValueError as error:
            self.error(error)
        self.refresh()

    def _can_add(self, frame):
        return (self.session is not None and self.session.crop[0] <= frame <= self.session.crop[1]
                and frame not in self.session.frames)

    def _frame_menu(self, frame, position):
        menu = QMenu(self)
        action = menu.addAction("Add frame here")
        action.setEnabled(self._can_add(frame))
        action.triggered.connect(lambda: self.add_frame(frame))
        menu.exec(position)

    def add_frame(self, frame):
        if not self._can_add(frame) or not self.mask_editor.can_leave():
            return
        self.player.stop()
        self.session.add(frame)
        self.selected = frame
        self.refresh()
        self.player.seek(frame)

    def apply_crop(self):
        if self.session is None or not self.mask_editor.can_leave():
            return
        try:
            self.session.set_crop(self.crop_start.value(), self.crop_end.value())
        except ValueError as error:
            self.error(error)
        self.refresh()
        self.player.seek(self.selected)

    def _animals_changed(self):
        if self.session is None or self.animals.value() == self.session.animals:
            return
        if not self.mask_editor.can_leave():
            self.refresh()
            return
        try:
            self.session.set_animals(self.animals.value())
            self.mask_editor.hide()
            self.mask_editor.draft = self.mask_editor.session = None
        except ValueError as error:
            self.error(error)
        self.refresh()

    def edit(self):
        if self.session is None or not self.mask_editor.can_leave():
            return
        self.player.stop()
        try:
            self._sync_predictor()
            image = self.videos.read("original", self.selected)
            if self.mask_editor.set_frame(self.session, self.selected, image):
                self.mask_editor.show()
                self.mask_editor.raise_()
                self.mask_editor.activateWindow()
        except Exception as error:
            self.error(error)

    def advance(self, step):
        if self.session is None:
            return
        frame = self.mask_editor.frame if self.mask_editor.isVisible() else self.selected
        if frame not in self.session.frames:
            frame = self.selected
        index = self.session.frames.index(frame) + step
        if 0 <= index < len(self.session.frames):
            self.select_frame(self.session.frames[index], edit=True)

    def _labeled(self, frame):
        self.refresh()
        self.seek(self.player.frame)

    def save(self):
        if self.session is None:
            return False
        if self.mask_editor.preview_mask is not None or (self.mask_editor.draft is not None and self.mask_editor.draft.dirty):
            return self.mask_editor.save()
        try:
            self.session.save()
        except Exception as error:
            self.error(error)
            return False
        self.statusBar().showMessage("Saved", 3000)
        notify_dataset_changed(self.session.dataset)
        return True

    def can_leave(self):
        if self.preview_task is not None:
            QMessageBox.information(self, "Previews", "Wait for the current preview to finish")
            return False
        if not self.mask_editor.can_leave():
            return False
        if self.session is not None and self.session.dirty:
            answer = QMessageBox.question(self, "Save", "Save changes to the key-frame selection?",
                                          QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard |
                                          QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Save)
            if answer == QMessageBox.StandardButton.Save:
                return self.save()
            return answer == QMessageBox.StandardButton.Discard
        return True

    def closeEvent(self, event):
        if not self.can_leave():
            event.ignore()
            return
        self.player.stop()
        self.previews.set_video(None)
        self.model_selector.shutdown()
        self.processing.shutdown()
        self.mask_editor.predictor.close()
        self.mask_editor.close()
        if self.recording is not None:
            self.recording.close()
            self.recording = None
            self.videos = None
        event.accept()
