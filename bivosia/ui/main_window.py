"""Application window listing open datasets and their editor windows.

Examples
--------
>>> from PySide6.QtWidgets import QApplication
>>> from bivosia.ui.main_window import MainWindow
>>> app = QApplication.instance() or QApplication([])  # doctest: +SKIP
>>> window = MainWindow()  # doctest: +SKIP
>>> window.show()  # doctest: +SKIP
>>> app.exec()  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path
import os

from PySide6.QtCore import QPoint, QRect, QSettings, Qt
from PySide6.QtGui import QKeySequence, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QMainWindow,
                               QMessageBox, QPushButton, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget)

from bivosia.config import APPLICATION_SETTINGS, RESOURCE_DIRECTORY
from bivosia.dataset.layout import Dataset
from bivosia.dataset.text import read_text
from bivosia.ui.about_dialog import show_about
from bivosia.ui.icons import action_icon
from bivosia.ui.remote_installations_dialog import RemoteInstallationsDialog
from bivosia.ui.text_preview import TextPreview
from bivosia.ui.window_manager import DatasetWindows, show_window


class _DatasetTree(QTreeWidget):
    """Keep the dataset list readable over a faint, centered application graphic."""

    def __init__(self):
        super().__init__()
        self._background = QPixmap(str(Path(__file__).with_name('assets') / 'bivosia-graphic.png'))
        self.viewport().setAutoFillBackground(False)

    def paintEvent(self, event):
        painter = QPainter(self.viewport())
        painter.fillRect(self.viewport().rect(), self.palette().brush(QPalette.ColorRole.Base))
        size = self._background.size()
        size.scale(self.viewport().size(), Qt.AspectRatioMode.KeepAspectRatio)
        target = QRect(QPoint(), size)
        target.moveCenter(self.viewport().rect().center())
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setOpacity(.18)
        painter.drawPixmap(target, self._background)
        painter.end()
        super().paintEvent(event)


class MainWindow(QMainWindow):
    """Open datasets and bring their existing windows forward."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle('BiVOSIA')
        self.resize(560, 420)
        self.windows = DatasetWindows(QApplication.instance())
        self.preferences = QSettings(*APPLICATION_SETTINGS)
        self.help_windows = {}
        self._items = {}
        body = QWidget()
        layout = QVBoxLayout(body)
        row = QHBoxLayout()
        open_button = QPushButton('Open dataset')
        open_button.setIcon(action_icon('open'))
        open_button.clicked.connect(self.choose_dataset)
        row.addWidget(open_button)
        self.workflow_button = QPushButton('Workflow')
        self.workflow_button.setIcon(action_icon('workflow'))
        self.workflow_button.clicked.connect(lambda: self.open_selected('workflow'))
        row.addWidget(self.workflow_button)
        self.files_button = QPushButton('View files')
        self.files_button.setIcon(action_icon('files'))
        self.files_button.clicked.connect(lambda: self.open_selected('files'))
        row.addWidget(self.files_button)
        layout.addLayout(row)
        self.tree = _DatasetTree()
        self.tree.setHeaderLabels(('Datasets and windows',))
        self.tree.itemClicked.connect(self._activate)
        self.tree.currentItemChanged.connect(self._selection_changed)
        layout.addWidget(self.tree)
        self.setCentralWidget(body)
        files = self.menuBar().addMenu('File')
        files.addAction(action_icon('new'), 'New dataset…', self.new_dataset)
        files.addAction(action_icon('open'), 'Open dataset…', self.choose_dataset).setShortcut(QKeySequence.StandardKey.Open)
        self.recent_menu = files.addMenu('Recent datasets')
        self.recent_menu.setIcon(action_icon('recent'))
        self._recent_menu()
        files.addAction(action_icon('workflow'), 'Workflow', lambda: self.open_selected('workflow'))
        files.addSeparator()
        files.addAction(action_icon('quit'), 'Quit', self.close).setShortcut(QKeySequence.StandardKey.Quit)
        settings = self.menuBar().addMenu('Settings')
        settings.addAction('Remote installations…', self.remote_installations)
        help_menu = self.menuBar().addMenu('Help')
        help_menu.addAction(action_icon('help'), 'Documentation',
                            lambda: self.show_help('docs/DOCUMENTATION.md'))
        help_menu.addAction(action_icon('help'), 'Remote setup',
                            lambda: self.show_help('docs/REMOTE_SETUP.md'))
        help_menu.addAction(action_icon('about'), 'About BiVOSIA', lambda: show_about(self))
        self.windows.changed.connect(self._refresh)
        self._selection_changed()

    def _run(self, function):
        try:
            return function()
        except Exception as error:
            QMessageBox.warning(self, 'BiVOSIA', str(error))

    def open_dataset(self, path):
        dataset = self.windows.add_dataset(Dataset(path))
        self._remember(dataset)
        self._refresh()
        self.tree.setCurrentItem(self._items[dataset.directory])
        return dataset

    def choose_dataset(self):
        path = QFileDialog.getExistingDirectory(self, 'Open dataset')
        if path:
            self._run(lambda: self.open_dataset(path))

    def new_dataset(self):
        from bivosia.ui.dataset_dialogs import create_dataset_dialog
        dataset = self._run(lambda: create_dataset_dialog(self))
        if dataset is not None:
            self.windows.add_dataset(dataset)
            self._remember(dataset)
            self._refresh()
            self.tree.setCurrentItem(self._items[dataset.directory])
            self._run(lambda: self.windows.open('workflow', dataset))

    def _remember(self, dataset):
        path = str(dataset.directory)
        recent = self.preferences.value('recent_datasets', [], type=list)
        self.preferences.setValue('recent_datasets', [path, *[item for item in recent if item != path]])
        self._recent_menu()

    def _recent_menu(self):
        self.recent_menu.clear()
        paths = self.preferences.value('recent_datasets', [], type=list)
        self.recent_menu.setEnabled(bool(paths))
        for path in paths:
            action = self.recent_menu.addAction(action_icon('open'), path)
            action.triggered.connect(lambda checked=False, path=path: self._run(lambda: self.open_dataset(path)))

    def selected_dataset(self):
        item = self.tree.currentItem()
        if item is None:
            return None
        return self.windows.datasets.get(item.data(0, Qt.ItemDataRole.UserRole))

    def open_selected(self, kind):
        dataset = self.selected_dataset()
        if dataset is not None:
            self._run(lambda: self.windows.open(kind, dataset))

    def _activate(self, item, column):
        window = item.data(0, Qt.ItemDataRole.UserRole + 1)
        if window is not None:
            show_window(window)

    def _selection_changed(self, *args):
        enabled = self.selected_dataset() is not None
        self.workflow_button.setEnabled(enabled)
        self.files_button.setEnabled(enabled)

    def _refresh(self):
        selected = self.tree.currentItem()
        selection = (selected.data(0, Qt.ItemDataRole.UserRole),
                     selected.data(0, Qt.ItemDataRole.UserRole + 1)) if selected else None
        expanded = {path: item.isExpanded() for path, item in self._items.items()}
        self.tree.clear()
        self._items.clear()
        for path, dataset in self.windows.datasets.items():
            root = QTreeWidgetItem(self.tree, [path.name])
            root.setToolTip(0, str(path))
            root.setData(0, Qt.ItemDataRole.UserRole, path)
            self._items[path] = root
            root.setExpanded(expanded.get(path, True))
            if selection == (path, None):
                self.tree.setCurrentItem(root)
            for window in self.windows.visible(dataset):
                item = QTreeWidgetItem(root, [window.windowTitle()])
                item.setData(0, Qt.ItemDataRole.UserRole, path)
                item.setData(0, Qt.ItemDataRole.UserRole + 1, window)
                if selection == (path, window):
                    self.tree.setCurrentItem(item)
        self._selection_changed()

    def show_help(self, filename, anchor=''):
        path = (RESOURCE_DIRECTORY / filename).resolve()
        if path not in self.help_windows:
            syntax = 'markdown' if path.suffix.lower() == '.md' else 'plain'
            assets = Path(__file__).with_name('assets')
            image_links = RESOURCE_DIRECTORY / 'bivosia/ui/assets'
            images = {Path(os.path.relpath(image_links / name, path.parent)).as_posix(): assets / name for name in
                      ('bivosia-icon.svg', 'bivosia-graphic.png', 'about.gif')}
            window = TextPreview(path.name, read_text(path), syntax, self, images=images)
            titles = {'DOCUMENTATION.md': 'BiVOSIA help', 'REMOTE_SETUP.md': 'Remote setup'}
            window.setWindowTitle(titles.get(path.name, path.name))
            if syntax == 'markdown':
                window.text.anchorClicked.connect(lambda url: self._open_help_link(path, url))
            window.destroyed.connect(lambda: self.help_windows.pop(path, None))
            self.help_windows[path] = window
        window = self.help_windows[path]
        show_window(window)
        if anchor and path.suffix.lower() == '.md':
            window.text.scrollToAnchor(anchor)

    def _open_help_link(self, source, url):
        if url.isRelative() and url.path():
            self._run(lambda: self.show_help(source.parent / url.path(), url.fragment()))

    def remote_installations(self):
        RemoteInstallationsDialog(self).exec()

    def closeEvent(self, event):
        if not self.windows.close_all():
            event.ignore()
            return
        for window in list(self.help_windows.values()):
            window.close()
        event.accept()
