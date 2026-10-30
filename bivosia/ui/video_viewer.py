"""View a dataset's experimental recording with shared playback controls.

Examples
--------
>>> from PySide6.QtWidgets import QApplication
>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.ui.video_viewer import VideoViewer
>>> app = QApplication.instance() or QApplication([])  # doctest: +SKIP
>>> viewer = VideoViewer(Dataset('/path/to/dataset'))  # doctest: +SKIP
>>> viewer.show()  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from PySide6.QtWidgets import QMainWindow

from bivosia.ui.video_preview import VideoPreview


class VideoViewer(QMainWindow):
    """Play the selected video without loading masks or analysis results."""

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.statusBar()
        self.setWindowTitle(f'Video · {dataset.directory.name}')
        self.resize(1000, 700)
        self.preview = VideoPreview(dataset.video, show_slider=False)
        self.preview.failed.connect(self.statusBar().showMessage)
        self.setCentralWidget(self.preview)

    def closeEvent(self, event):
        self.preview.close()
        event.accept()
