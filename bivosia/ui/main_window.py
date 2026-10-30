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

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QKeySequence, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import (QHBoxLayout, QMainWindow,
                               QMessageBox, QPushButton, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget)

from bivosia.dataset.layout import Dataset
from bivosia.ui.application_menus import DatasetMenu, HelpMenu, add_settings_menu
from bivosia.ui.icons import action_icon
from bivosia.ui.window_manager import dataset_windows, show_window


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

    def __init__(self, windows=None):
        super().__init__()
        self.setWindowTitle('BiVOSIA')
        self.resize(560, 420)
        self.windows = windows if windows is not None else dataset_windows()
        self.windows.main_window = self
        self._items = {}
        body = QWidget()
        layout = QVBoxLayout(body)
        row = QHBoxLayout()
        open_button = QPushButton('Open dataset')
        open_button.setIcon(action_icon('open'))
        open_button.clicked.connect(lambda: self.file_menu.choose_dataset())
        row.addWidget(open_button)
        self.workflow_button = QPushButton('Workflow')
        self.workflow_button.setIcon(action_icon('workflow'))
        self.workflow_button.clicked.connect(lambda: self.file_menu.open_window('workflow'))
        row.addWidget(self.workflow_button)
        self.viewer_button = QPushButton('View analysis')
        self.viewer_button.setIcon(action_icon('view'))
        self.viewer_button.clicked.connect(lambda: self.file_menu.open_window('viewer'))
        row.addWidget(self.viewer_button)
        self.files_button = QPushButton('View files')
        self.files_button.setIcon(action_icon('files'))
        self.files_button.clicked.connect(lambda: self.file_menu.open_window('files'))
        row.addWidget(self.files_button)
        layout.addLayout(row)
        self.tree = _DatasetTree()
        self.tree.setHeaderLabels(('Datasets and windows',))
        self.tree.itemClicked.connect(self._activate)
        self.tree.currentItemChanged.connect(self._selection_changed)
        layout.addWidget(self.tree)
        self.setCentralWidget(body)
        files = self.file_menu = DatasetMenu(self, self.windows, self.selected_dataset, self.open_dataset)
        self.menuBar().addMenu(files)
        new_action = files.addAction(action_icon('new'), 'New dataset…', self.new_dataset)
        files.insertAction(files.open_action, new_action)
        self.viewer_action = files.addAction(action_icon('view'), 'View analysis', lambda: files.open_window('viewer'))
        files.insertAction(files.files_action, self.viewer_action)
        files.addSeparator()
        files.addAction(action_icon('quit'), 'Quit', self.close).setShortcut(QKeySequence.StandardKey.Quit)
        add_settings_menu(self)
        self.help_menu = HelpMenu(self)
        self.menuBar().addMenu(self.help_menu)
        self.windows.changed.connect(self._refresh)
        self._refresh()

    def _run(self, function):
        try:
            return function()
        except Exception as error:
            QMessageBox.warning(self, 'BiVOSIA', str(error))

    def open_dataset(self, path):
        dataset = self.windows.add_dataset(Dataset(path))
        self.file_menu.remember(dataset)
        self._refresh()
        self.tree.setCurrentItem(self._items[dataset.directory])
        return dataset

    def new_dataset(self):
        from bivosia.ui.dataset_dialogs import create_dataset_dialog
        dataset = self._run(lambda: create_dataset_dialog(self))
        if dataset is not None:
            self.windows.add_dataset(dataset)
            self.file_menu.remember(dataset)
            self._refresh()
            self.tree.setCurrentItem(self._items[dataset.directory])
            self._run(lambda: self.windows.open('workflow', dataset))

    def selected_dataset(self):
        item = self.tree.currentItem()
        if item is None:
            return None
        return self.windows.datasets.get(item.data(0, Qt.ItemDataRole.UserRole))

    def _activate(self, item, column):
        window = item.data(0, Qt.ItemDataRole.UserRole + 1)
        if window is not None:
            show_window(window)

    def _selection_changed(self, *args):
        enabled = self.selected_dataset() is not None
        self.workflow_button.setEnabled(enabled)
        self.files_button.setEnabled(enabled)
        self.viewer_button.setEnabled(enabled)
        self.viewer_action.setEnabled(enabled)
        self.file_menu.set_dataset_enabled(enabled)

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

    def closeEvent(self, event):
        if not self.windows.close_all():
            event.ignore()
            return
        self.help_menu.close_windows()
        event.accept()
