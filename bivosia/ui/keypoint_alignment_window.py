"""Calculate keypoint alignment within the recording viewer.

Examples
--------
>>> from PySide6.QtWidgets import QApplication
>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.ui.window_manager import DatasetWindows
>>> app = QApplication.instance() or QApplication([])  # doctest: +SKIP
>>> windows = DatasetWindows(app)  # doctest: +SKIP
>>> window = windows.open('alignment', Dataset('/path/to/dataset'))  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from bivosia.dataset.layout import Dataset
from bivosia.ui.analysis_window import AnalysisWindow
from bivosia.ui.dock_panel import PanelHost
from bivosia.ui.recording_viewer import create_recording_viewer


class _AlignmentControls(AnalysisWindow):
    """Refresh the viewer after loading settings or finishing an alignment job."""

    def __init__(self, dataset, windows, owner):
        self.owner = owner
        super().__init__(dataset, windows, 'Keypoint alignment', 'align', keypoints=True,
                         processing=True, parent=owner)

    def statusBar(self):
        return self.owner.statusBar()

    def _ready(self, info):
        if self._closed or self._refresh_pending:
            return
        super()._ready(info)
        self.owner.refresh_viewer()

    def _select_annotation(self):
        super()._select_annotation()
        self.refresh()


class KeypointAlignmentWindow(PanelHost):
    """Run alignment and inspect saved assignments in the shared viewer."""

    def __init__(self, dataset, windows):
        super().__init__()
        self.dataset, self.windows = dataset, windows
        self.viewer = None
        self.controls = _AlignmentControls(dataset, windows, self)
        self.setCentralWidget(self.controls)
        self.setWindowTitle(f'Keypoint alignment · {dataset.directory.name}')
        self.resize(1400, 900)

    def refresh_viewer(self):
        previous = self.viewer
        if previous is None:
            self.takeCentralWidget()
        self.dataset = self.controls.dataset
        try:
            viewer = create_recording_viewer(self.dataset, parent=self,
                open_dataset=self._open_dataset, controls=('Keypoint alignment', self.controls))
        except Exception as error:
            if previous is None:
                self.setCentralWidget(self.controls)
            else:
                previous.panel_docks['Keypoint alignment'].setWidget(self.controls)
            self.controls.statusBar().showMessage(str(error))
            return
        aligned = viewer.keypoints is not None and viewer.keypoints.alignment is not None
        mode = 'both' if aligned else 'original'
        if previous is not None:
            if previous.keypoints is not None and previous.keypoints.alignment is not None:
                mode = previous.mask_display.keypoint_mode.currentData()
            viewer.show_frame(min(previous.current_frame, viewer.videos.frame_count - 1))
        if viewer.mask_display.keypoint_mode is not None:
            viewer.mask_display.keypoint_mode.setCurrentIndex(viewer.mask_display.keypoint_mode.findData(mode))
        self.viewer = viewer
        self.setCentralWidget(viewer)
        if previous is not None:
            previous.close()
        self.controls.statusBar().showMessage('Alignment loaded.' if aligned else 'No saved alignment. Select settings and press Run.')

    def _open_dataset(self, path):
        dataset = Dataset(path)
        replacement = self.windows.open('alignment', dataset)
        self.viewer.file_menu.remember(dataset)
        if replacement is not self:
            self.close()

    def refresh_font_geometry(self):
        if self.viewer is not None:
            self.viewer.refresh_font_geometry()

    def closeEvent(self, event):
        self.controls.close()
        if self.viewer is not None:
            self.viewer.close()
        event.accept()
