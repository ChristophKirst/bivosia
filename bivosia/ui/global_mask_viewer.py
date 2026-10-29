"""Read-only inspection of exported masks with one global identity palette.

Examples
--------
With an existing video and exported linked masks:

>>> from PySide6.QtWidgets import QApplication
>>> from bivosia.ui.global_mask_viewer import GlobalMaskViewer
>>> app = QApplication.instance() or QApplication([])  # doctest: +SKIP
>>> viewer = GlobalMaskViewer(  # doctest: +SKIP
...     "/path/to/recording.mp4", "/path/to/review.linked.h5", 6)
>>> viewer.showMaximized()  # doctest: +SKIP
>>> app.exec()  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import ExitStack
from pathlib import Path

from PySide6.QtWidgets import QHBoxLayout, QLabel, QMainWindow, QVBoxLayout, QWidget

from bivosia.config import LINKED_MASK_STREAMS
from bivosia.dataset.masks import open_linked_masks
from bivosia.dataset.video import VideoSet
from bivosia.ui.colors import bgr_colors, identity_colors
from bivosia.ui.player import FramePlayer
from bivosia.ui.video_panels import (
    MaskDisplayControl, VideoViewGroup, ZoomableVideoLabel, image_from_bgr,
)


class GlobalMaskViewer(QMainWindow):
    def __init__(self, video, masks, n):
        super().__init__()
        self.statusBar()
        self.setWindowTitle(f"Global identities · {Path(masks).name}")
        with ExitStack() as resources:
            self.videos = VideoSet({"original": Path(video)})
            resources.callback(self.videos.close)
            self.masks = resources.enter_context(open_linked_masks(masks))
            self.count = self.masks['preferred'].shape[0]
            if self.videos.frame_count != self.count:
                raise ValueError("Export and video frame counts differ")
            self.colors = bgr_colors(identity_colors(n, .27))
            self._build_ui()
            self._resources = resources.pop_all()
        self.destroyed.connect(self._resources.close)

    def _build_ui(self):
        root = QWidget()
        layout = QVBoxLayout(root)
        self.player = FramePlayer("frame", "#ff3b4f", selectable=False)
        self.player.configure(self.count, self.videos.source_fps or 50.)
        self.player.loop.setChecked(False)
        self.player.frameRequested.connect(
            lambda frame, _playback: self.show_frame(frame)
        )
        controls = QHBoxLayout()
        controls.addWidget(self.player)
        controls.addStretch()
        self.mask_display = MaskDisplayControl()
        self.mask_display.valueChanged.connect(
            lambda: self.show_frame(self.player.frame)
        )
        controls.addWidget(self.mask_display)
        layout.addLayout(controls)
        row = QHBoxLayout()
        self.panes, self.titles = {}, {}
        for name in LINKED_MASK_STREAMS:
            panel = QVBoxLayout()
            self.titles[name] = QLabel(name.title())
            self.panes[name] = ZoomableVideoLabel()
            panel.addWidget(self.titles[name])
            panel.addWidget(self.panes[name], 1)
            row.addLayout(panel)
        self.navigation = VideoViewGroup(self.panes.values())
        layout.addLayout(row, 1)
        self.setCentralWidget(root)
        self.resize(1400, 800)
        self._initial_video_zoom = True
        self.show_frame(0)

    def showEvent(self, event):
        super().showEvent(event)
        if self._initial_video_zoom:
            self._initial_video_zoom = False
            frame = self.player.frame
            original = self.videos.read('original', frame)
            self.navigation.fit_masks([self.masks[name][frame] for name in self.panes], original.shape)

    def show_frame(self, frame):
        try:
            original = self.videos.read("original", frame)
            for name, pane in self.panes.items():
                valid = bool(self.masks[name].valid[frame])
                self.titles[name].setText(f"{name.title()} · frame {frame:,}" + ("" if valid else " · INVALID / UNRESOLVED"))
                pane.set_image(image_from_bgr(self.mask_display.composite(
                    original, self.masks[name][frame], None, self.colors,
                )))
            self.player.set_frame(frame)
        except Exception as error:
            self.player.stop()
            self.statusBar().showMessage(str(error))

    def closeEvent(self, event):
        self.player.stop()
        self._resources.close()
        event.accept()
