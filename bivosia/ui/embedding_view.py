"""Linked embedding, recording and annotation views for mask-feature inspection."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QComboBox, QFormLayout, QPushButton, QVBoxLayout, QWidget

from bivosia.analysis.zones_of_disagreement import RevisionZone
from bivosia.dataset.shape_features import open_feature_video
from bivosia.ui.colors import MASK_COLORS_BGR, identity_colors
from bivosia.ui.dock_panel import PanelSplitter
from bivosia.ui.embedding_plot import EmbeddingPlot
from bivosia.ui.embedding_timeline import EmbeddingTimeline
from bivosia.ui.mask_selection import MaskSelectionLabel
from bivosia.ui.player import PlayerCursorController
from bivosia.ui.video_panels import MaskDisplayControl
from bivosia.ui.video_preview import VideoPreview
from bivosia.ui.zone_plot import ZonePanel


class _MaskPreview(VideoPreview):
    """Use the selected consensus-mask source with the shared video controls."""

    maskSelected = Signal(int, int)

    def __init__(self):
        super().__init__(show_slider=False, pane=MaskSelectionLabel())
        self._mask_frame, self._labels = None, None
        self._selection = (None, None)
        self.pane.imageClicked.connect(self._select_mask)
        self.pane.setToolTip('Click: select mask · left-drag: pan · right-drag: rectangle zoom · wheel: zoom')
        self.display = MaskDisplayControl()
        self.layout().insertWidget(1, self.display)
        self.display.valueChanged.connect(lambda: self.show_frame(self.player.frame))
        self.sample_colors = lambda frame: None

    def render_frame(self, image, frame):
        masks = self.videos.masks.get('preferred')
        self._mask_frame = frame
        self._labels = masks[frame] if masks is not None else None
        self._update_selection()
        if self._labels is None:
            return image
        return self.display.composite(image, self._labels, MASK_COLORS_BGR,
                                      label_colors=self.sample_colors(frame))

    def set_selected(self, frame, identity):
        self._selection = (frame, identity)
        self._update_selection()

    def _update_selection(self):
        frame, identity = self._selection
        self.pane.set_selection(self._labels, identity if frame == self._mask_frame else None)

    def _select_mask(self, x, y):
        if self.videos is None or self._labels is None or self._mask_frame != self.player.frame:
            return
        height, width = self._labels.shape
        identity = int(self._labels[min(height - 1, int(y * height)), min(width - 1, int(x * width))])
        if identity:
            self.maskSelected.emit(self._mask_frame, identity)


class EmbeddingView(QWidget):
    """Keep the selected frame and samples synchronized across three displays."""

    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.frames = self.identities = np.empty(0, dtype=int)
        self.coordinates = None
        self.animal_colors = None
        self._video_source = None
        self._selecting = False
        self._analysis = None
        self.mask_windows = set()
        self.plot = EmbeddingPlot()
        self.preview = _MaskPreview()
        self.timeline = EmbeddingTimeline()
        top = PanelSplitter(Qt.Orientation.Horizontal)
        top.addWidget(self.plot)
        top.addWidget(self.preview)
        top.setSizes([500, 500])
        split = PanelSplitter(Qt.Orientation.Vertical)
        split.addWidget(top)
        split.addWidget(ZonePanel(self.timeline))
        split.setSizes([500, 180])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(split)
        self.controls = QWidget()
        form = QFormLayout(self.controls)
        form.setContentsMargins(0, 0, 0, 0)
        self.color_mode = QComboBox()
        self.color_mode.addItem('Sample', 'sample')
        self.color_mode.addItem('Animal', 'animal')
        form.addRow('Color by', self.color_mode)
        self.masks_button = QPushButton('Show masks')
        self.masks_button.setEnabled(False)
        self.masks_button.clicked.connect(self.show_masks)
        form.addRow(self.masks_button)
        self.color_mode.currentIndexChanged.connect(self._recolor)
        self.preview.sample_colors = self._mask_colors
        self.preview.failed.connect(self.failed)
        self.preview.maskSelected.connect(self.select_pair)
        self.plot.pointSelected.connect(self.select_sample)
        self.timeline.pointSelected.connect(self.select_sample)
        self.timeline.frameClicked.connect(self.preview.player.seek)
        self.preview.player.frameRequested.connect(lambda frame, playback: self.select_frame(frame))
        self.cursors = PlayerCursorController([self.preview.player])
        self.cursors.set_plots(brackets=(self.timeline,))

    def clear_samples(self):
        self.masks_button.setEnabled(False)
        self.coordinates = None
        self.frames = self.identities = np.empty(0, dtype=int)
        self._recolor()

    def set_data(self, info, dataset, settings):
        source = settings['mask_source']
        self._analysis = (dataset, info, dict(settings))
        self.coordinates = info.get('coordinates')
        self.frames = np.asarray(info['frames']) if info.get('frames') is not None else np.empty(0, dtype=int)
        self.identities = np.asarray(info['identities']) if info.get('identities') is not None else np.empty(0, dtype=int)
        self.animal_colors = (np.array([color.getRgb()[:3][::-1] for color in
                              identity_colors(int(self.identities.max()))], dtype=np.uint8)
                              if len(self.identities) else None)
        self.plot.set_points(self.coordinates, self.identities, self.color_mode.currentData())
        self.masks_button.setEnabled(self.coordinates is not None and self.coordinates.shape[1] == 2)
        zones = [RevisionZone(**zone) for zone in info.get('zones', ())]
        count = info.get('frame_count', 1)
        self.timeline.set_data(count, zones, self.frames, self.identities, self.plot.colors)
        key = (str(dataset.directory), source, info.get('source'))
        if key != self._video_source:
            try:
                self.preview.set_videos(open_feature_video(dataset, source=source))
                self._video_source = key
            except Exception as error:
                self._video_source = None
                self.preview.player.stop()
                if self.preview.videos is not None:
                    self.preview.videos.close()
                    self.preview.videos = None
                self.preview.pane.clear()
                self.preview.pane.setText(str(error))
                # Keep embedding/timeline navigation available without a local video.
                self.preview.player.configure(count, 0)
                self.failed.emit(str(error))
        self.select_frame(self.preview.player.frame)
        self.preview.show_frame(self.preview.player.frame)

    def _recolor(self):
        self.plot.set_points(self.coordinates, self.identities, self.color_mode.currentData())
        self.timeline.frames, self.timeline.identities = self.frames, self.identities
        self.timeline.animals = int(self.identities.max()) if len(self.identities) else 1
        self.timeline.colors = self.plot.colors
        self.timeline.invalidate()
        self.select_frame(self.preview.player.frame)
        self.preview.show_frame(self.preview.player.frame)

    def _indices(self, frame):
        start = np.searchsorted(self.frames, frame, side='left')
        end = np.searchsorted(self.frames, frame, side='right')
        return np.arange(start, end)

    def _mask_colors(self, frame):
        if not len(self.identities):
            return None
        colors = self.animal_colors.copy()
        for index in self._indices(frame):
            colors[self.identities[index] - 1] = self.plot.colors[index, :3][::-1]
        return colors

    def select_frame(self, frame):
        self.timeline.set_current_frame(frame)
        if not self._selecting:
            self._highlight(self._indices(frame))

    def select_sample(self, index):
        if not 0 <= index < len(self.frames):
            return
        self._selecting = True
        try:
            self.preview.player.seek(int(self.frames[index]))
            self._highlight([index])
        finally:
            self._selecting = False

    def _highlight(self, indices):
        self.plot.set_selected(indices)
        self.timeline.set_selected(indices)
        if len(indices) == 1:
            index = indices[0]
            self.preview.set_selected(int(self.frames[index]), int(self.identities[index]))
        else:
            self.preview.set_selected(None, None)

    def show_masks(self):
        if self._analysis is None or self.coordinates is None or self.coordinates.shape[1] != 2:
            return
        from bivosia.ui.embedding_masks_window import EmbeddingMasksWindow
        window = EmbeddingMasksWindow(*self._analysis, self.plot.two, self.window())
        self.mask_windows.add(window)
        window.destroyed.connect(lambda: self.mask_windows.discard(window))
        window.sampleSelected.connect(self.select_pair)
        window.show()

    def select_pair(self, frame, identity):
        indices = self._indices(frame)
        indices = indices[self.identities[indices] == identity]
        if len(indices):
            self.select_sample(int(indices[0]))
        else:
            self.preview.player.seek(frame)
            self._highlight([])
            self.preview.set_selected(frame, identity)

    def closeEvent(self, event):
        for window in tuple(self.mask_windows):
            window.close()
        self.preview.close()
        event.accept()
