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

from bivosia.dataset.video import RecordingStreams
from bivosia.ui.player import FramePlayer
from bivosia.ui.video_panels import VideoViewGroup, ZoomableVideoLabel, image_from_bgr


class VideoPreview(QWidget):
    """Display one video, releasing its reader when the preview closes."""

    failed = Signal(str)

    def __init__(self, path=None, parent=None, *, show_slider=True, pane=None):
        super().__init__(parent)
        self.path = Path(path) if path is not None else None
        self.videos = None
        self._closed = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.player = FramePlayer('frame', '#ff3b4f', selectable=False)
        self.player.frameRequested.connect(self.show_frame)
        layout.addWidget(self.player)
        self.slider = None
        if show_slider:
            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setRange(0, 0)
            self.slider.setEnabled(False)
            self.slider.sliderPressed.connect(self.player.stop)
            self.slider.valueChanged.connect(self.player.seek)
            self.player.frameChanged.connect(self._sync_slider)
            layout.addWidget(self.slider)
        self.pane = pane if pane is not None else ZoomableVideoLabel()
        self.navigation = VideoViewGroup([self.pane])
        layout.addWidget(self.pane, 1)
        if self.path is not None:
            QTimer.singleShot(0, self._load)

    def set_videos(self, videos):
        """Replace owned readers after the new recording's first frame is readable."""
        try:
            image = videos.read('original', 0)
        except BaseException:
            videos.close()
            raise
        previous, self.videos = self.videos, videos
        if previous is not None:
            previous.close()
        self.player.configure(videos.frame_count, videos.source_fps)
        if self.slider is not None:
            self.slider.setRange(0, videos.frame_count - 1)
            self.slider.setEnabled(True)
        self.pane.set_image(image_from_bgr(self.render_frame(image, 0)))

    def render_frame(self, image, frame):
        """Provide the displayed image, optionally specialized by a mask preview."""
        return image

    def _load(self):
        if self._closed:
            return
        videos = None
        try:
            videos = RecordingStreams({'original': self.path})
            self.set_videos(videos)
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
            self.pane.set_image(image_from_bgr(self.render_frame(image, frame)))
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
