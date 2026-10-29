"""Read-only video preview with shared playback and image navigation.

Examples
--------
>>> from PySide6.QtWidgets import QApplication
>>> from bivosia.ui.video_preview import VideoPreview
>>> app = QApplication.instance() or QApplication([])  # doctest: +SKIP
>>> viewer = VideoPreview('/path/to/recording.mp4')  # doctest: +SKIP
>>> viewer.show()  # doctest: +SKIP
>>> app.exec()  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path
import logging

from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtWidgets import QSlider, QVBoxLayout, QWidget

from bivosia.dataset.video import VideoSet
from bivosia.ui.player import FramePlayer
from bivosia.ui.video_panels import VideoViewGroup, ZoomableVideoLabel, image_from_bgr


class VideoPreview(QWidget):
    """Display one video, releasing its reader when the preview closes."""

    failed = Signal(str)

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = Path(path)
        self.videos = None
        self._closed = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.player = FramePlayer('frame', '#ff3b4f', selectable=False)
        self.player.frameRequested.connect(self.show_frame)
        layout.addWidget(self.player)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setEnabled(False)
        self.slider.sliderPressed.connect(self.player.stop)
        self.slider.valueChanged.connect(self.player.seek)
        self.player.frameChanged.connect(self._sync_slider)
        layout.addWidget(self.slider)
        self.pane = ZoomableVideoLabel()
        self.navigation = VideoViewGroup([self.pane])
        layout.addWidget(self.pane, 1)
        QTimer.singleShot(0, self._load)

    def _load(self):
        if self._closed:
            return
        videos = None
        try:
            videos = VideoSet({'original': self.path})
            image = videos.read('original', 0)
            self.player.configure(videos.frame_count, videos.source_fps)
            self.slider.setRange(0, videos.frame_count - 1)
            self.slider.setEnabled(True)
            self.pane.set_image(image_from_bgr(image))
        except Exception as error:
            if videos is not None:
                videos.close()
            self._error(error)
            return
        self.videos = videos
        self.destroyed.connect(videos.close)

    def _sync_slider(self, frame):
        with QSignalBlocker(self.slider):
            self.slider.setValue(frame)

    def show_frame(self, frame, playback=False):
        if self.videos is None or self._closed:
            return
        try:
            image = self.videos.read('original', frame)
            self.pane.set_image(image_from_bgr(image))
        except Exception as error:
            self._error(error)

    def _error(self, error):
        logging.getLogger(__name__).exception('Video preview failed')
        self.player.stop()
        self.pane.clear()
        self.pane.setText(str(error))
        self.failed.emit(str(error))

    def closeEvent(self, event):
        self._closed = True
        self.player.stop()
        if self.videos is not None:
            self.videos.close()
            self.videos = None
        event.accept()
