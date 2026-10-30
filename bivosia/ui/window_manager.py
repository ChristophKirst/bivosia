"""Keep dataset windows together and reuse editors that are already open.

Examples
--------
>>> from PySide6.QtWidgets import QApplication
>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.ui.window_manager import DatasetWindows
>>> app = QApplication.instance() or QApplication([])  # doctest: +SKIP
>>> windows = DatasetWindows(app)  # doctest: +SKIP
>>> windows.open('files', Dataset('/path/to/dataset'))  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import QApplication, QDialog, QMenu, QWidget

from bivosia.dataset.layout import Dataset


def window_dataset(window):
    """Find the dataset belonging to a window, its session or its parent."""
    while window is not None:
        dataset = getattr(window, 'dataset', None)
        if not isinstance(dataset, Dataset):
            dataset = getattr(getattr(window, 'session', None), 'dataset', None)
        if isinstance(dataset, Dataset):
            return dataset
        window = window.parentWidget()
    return None


def show_window(window):
    """Bring a window forward without changing an existing maximized layout."""
    if window.isMinimized():
        window.showNormal()
    else:
        window.show()
    window.raise_()
    window.activateWindow()


def dataset_windows():
    """Reuse the application's dataset-window manager, including standalone viewers."""
    app = QApplication.instance()
    windows = app.findChild(DatasetWindows)
    return windows if windows is not None else DatasetWindows(app)


def notify_dataset_changed(dataset):
    """Notify open workflow windows after dataset files or settings are saved."""
    dataset_windows().dataset_changed.emit(dataset.directory)


class DatasetWindows(QObject):
    """Track visible dataset windows and own windows opened by the application."""

    changed = Signal()
    dataset_changed = Signal(object)

    def __init__(self, app):
        super().__init__(app)
        self.datasets = {}
        self.windows = {}
        self.main_window = None
        self._pending = False
        app.installEventFilter(self)

    def add_dataset(self, dataset):
        if not dataset.directory.is_dir():
            raise FileNotFoundError(f'Dataset folder does not exist: {dataset.directory}')
        self.datasets[dataset.directory] = dataset
        self._schedule()
        return dataset

    def _schedule(self):
        if not self._pending:
            self._pending = True
            QTimer.singleShot(0, self._refresh)

    def eventFilter(self, watched, event):
        if (event.type() in (QEvent.Type.Show, QEvent.Type.Hide, QEvent.Type.WindowTitleChange)
                and isinstance(watched, QWidget) and watched.isWindow()
                and not isinstance(watched, (QDialog, QMenu))):
            self._schedule()
        return False

    def _refresh(self):
        self._pending = False
        for window in QApplication.topLevelWidgets():
            if window.isVisible() and not isinstance(window, (QDialog, QMenu)):
                dataset = window_dataset(window)
                if dataset is not None:
                    self.datasets.setdefault(dataset.directory, dataset)
                    self.track(window)
        self.changed.emit()

    def track(self, window):
        identity = id(window)
        if identity not in self.windows:
            self.windows[identity] = window
            window.dataset_windows = self
            window.destroyed.connect(lambda: self._forget(identity))

    def _forget(self, identity):
        self.windows.pop(identity, None)
        self._schedule()

    def visible(self, dataset):
        return [window for window in self.windows.values()
                if window.isVisible() and (owner := window_dataset(window)) is not None
                and owner.directory == dataset.directory]

    def open_main(self, dataset=None):
        """Show the main BiVOSIA window and select the supplied dataset."""
        if self.main_window is None:
            from bivosia.ui.main_window import MainWindow
            self.main_window = MainWindow(self)
        if dataset is not None:
            self.main_window.open_dataset(dataset.directory)
        show_window(self.main_window)
        return self.main_window

    def open(self, kind, dataset):
        """Open one workflow, file browser or editor for this dataset."""
        dataset = self.add_dataset(dataset)
        # Read the saved description again after another editor may have updated it.
        current = Dataset(dataset.directory) if dataset.description.is_file() else dataset
        for window in self.visible(dataset):
            if window.property('dataset_window_kind') == kind:
                if kind == 'video' and window.preview.path != current.video:
                    window.close()
                    continue
                show_window(window)
                return window
        if kind == 'workflow':
            from bivosia.ui.workflow_window import WorkflowWindow
            window = WorkflowWindow(current, self)
        elif kind == 'files':
            from bivosia.ui.dataset_viewer import DatasetWidget
            window = DatasetWidget(current)
        elif kind == 'key_frames':
            from bivosia.ui.key_frame_editor import KeyFrameWindow
            window = KeyFrameWindow(current)
        elif kind == 'vos':
            from bivosia.ui.vos_window import VosWindow
            window = VosWindow(current)
        elif kind == 'features':
            from bivosia.ui.mask_features_window import MaskFeaturesWindow
            window = MaskFeaturesWindow(current, self)
        elif kind == 'mask_analysis':
            from bivosia.ui.mask_analysis_window import MaskAnalysisWindow
            window = MaskAnalysisWindow(current, self)
        elif kind == 'alignment':
            from bivosia.ui.keypoint_alignment_window import KeypointAlignmentWindow
            window = KeypointAlignmentWindow(current, self)
        elif kind == 'zones':
            from bivosia.ui.revision_linking_editor import RevisionLinkingWindow
            window = RevisionLinkingWindow(current)
        elif kind == 'viewer':
            from bivosia.ui.recording_viewer import create_recording_viewer
            window = create_recording_viewer(current)
        elif kind == 'video':
            from bivosia.ui.video_viewer import VideoViewer
            window = VideoViewer(current)
        else:
            raise ValueError(f'Unknown dataset window: {kind}')
        window.setProperty('dataset_window_kind', kind)
        window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.track(window)
        if kind in ('key_frames', 'vos', 'zones', 'viewer'):
            window.showMaximized()
        else:
            window.show()
        self._schedule()
        return window

    def close_all(self):
        """Respect each editor's save, cancellation and active-operation checks."""
        self._refresh()
        for window in reversed(tuple(self.windows.values())):
            if window.isVisible() and not window.close():
                show_window(window)
                return False
        return True
