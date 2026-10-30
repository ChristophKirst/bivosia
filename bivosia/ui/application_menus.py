"""Shared dataset navigation, settings and documentation menus."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path
import os

from PySide6.QtCore import QSettings
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QFileDialog, QMenu, QMessageBox

from bivosia.config import APPLICATION_SETTINGS, RESOURCE_DIRECTORY
from bivosia.dataset.text import read_text
from bivosia.ui.about_dialog import show_about
from bivosia.ui.icons import action_icon
from bivosia.ui.remote_installations_dialog import RemoteInstallationsDialog
from bivosia.ui.text_preview import TextPreview
from bivosia.ui.window_manager import show_window


class DatasetMenu(QMenu):
    """Open datasets and their windows using the owning window's selection."""

    def __init__(self, window, windows, selected_dataset, open_dataset):
        super().__init__('File', window)
        self.window = window
        self.windows = windows
        self.selected_dataset = selected_dataset
        self.open_dataset = open_dataset
        self.preferences = QSettings(*APPLICATION_SETTINGS)
        self.open_action = self.addAction(action_icon('open'), 'Open dataset…', self.choose_dataset)
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.recent_menu = self.addMenu('Recent datasets')
        self.recent_menu.setIcon(action_icon('recent'))
        self.workflow_action = self.addAction(action_icon('workflow'), 'Workflow',
                                             lambda: self.open_window('workflow'))
        self.files_action = self.addAction(action_icon('files'), 'View files',
                                          lambda: self.open_window('files'))
        self.aboutToShow.connect(self.refresh)
        self.refresh()

    def _run(self, callback):
        try:
            return callback()
        except Exception as error:
            QMessageBox.warning(self.window, 'BiVOSIA', str(error))

    def choose_dataset(self):
        dataset = self.selected_dataset()
        directory = str(dataset.directory.parent) if dataset is not None else ''
        path = QFileDialog.getExistingDirectory(self.window, 'Open dataset', directory)
        if path:
            self._run(lambda: self.open_dataset(path))

    def remember(self, dataset):
        path = str(dataset.directory)
        recent = self.preferences.value('recent_datasets', [], type=list)
        self.preferences.setValue('recent_datasets', [path, *[item for item in recent if item != path]])

    def refresh(self):
        self.recent_menu.clear()
        paths = self.preferences.value('recent_datasets', [], type=list)
        self.recent_menu.setEnabled(bool(paths))
        for path in paths:
            action = self.recent_menu.addAction(action_icon('open'), path)
            action.triggered.connect(lambda checked=False, path=path: self._run(lambda: self.open_dataset(path)))
        self.set_dataset_enabled(self.selected_dataset() is not None)

    def set_dataset_enabled(self, enabled):
        self.workflow_action.setEnabled(enabled)
        self.files_action.setEnabled(enabled)

    def open_window(self, kind):
        dataset = self.selected_dataset()
        if dataset is not None:
            return self._run(lambda: self.windows.open(kind, dataset))

    def open_main(self):
        return self._run(lambda: self.windows.open_main(self.selected_dataset()))


class HelpMenu(QMenu):
    """Display the same documentation and About dialog in application windows."""

    def __init__(self, window):
        super().__init__('Help', window)
        self.window = window
        self.help_windows = {}
        self.addAction(action_icon('help'), 'Documentation', lambda: self.show_document('docs/DOCUMENTATION.md'))
        self.addAction(action_icon('help'), 'Remote setup', lambda: self.show_document('docs/REMOTE_SETUP.md'))
        self.addAction(action_icon('about'), 'About BiVOSIA', lambda: show_about(window))

    def show_document(self, filename, anchor=''):
        try:
            path = (RESOURCE_DIRECTORY / filename).resolve()
            if path not in self.help_windows:
                syntax = 'markdown' if path.suffix.lower() == '.md' else 'plain'
                assets = Path(__file__).with_name('assets')
                image_links = RESOURCE_DIRECTORY / 'bivosia/ui/assets'
                images = {Path(os.path.relpath(image_links / name, path.parent)).as_posix(): assets / name for name in
                          ('bivosia-icon.svg', 'bivosia-graphic.png', 'about.gif')}
                window = TextPreview(path.name, read_text(path), syntax, self.window, images=images)
                titles = {'DOCUMENTATION.md': 'BiVOSIA help', 'REMOTE_SETUP.md': 'Remote setup'}
                window.setWindowTitle(titles.get(path.name, path.name))
                if syntax == 'markdown':
                    window.text.anchorClicked.connect(lambda url: self._open_link(path, url))
                window.destroyed.connect(lambda: self.help_windows.pop(path, None))
                self.help_windows[path] = window
            window = self.help_windows[path]
            show_window(window)
            if anchor and path.suffix.lower() == '.md':
                window.text.scrollToAnchor(anchor)
        except Exception as error:
            QMessageBox.warning(self.window, 'BiVOSIA', str(error))

    def _open_link(self, source, url):
        if url.isRelative() and url.path():
            self.show_document(source.parent / url.path(), url.fragment())

    def close_windows(self):
        for window in list(self.help_windows.values()):
            window.close()


def add_settings_menu(window):
    """Add the shared remote-installation settings entry."""
    menu = window.menuBar().addMenu('Settings')
    menu.addAction('Remote installations…', lambda: RemoteInstallationsDialog(window).exec())
    return menu


def add_dataset_window_menus(window, windows, selected_dataset, open_dataset):
    """Add dataset navigation and shared Help around a window's existing menus."""
    files = DatasetMenu(window, windows, selected_dataset, open_dataset)
    existing = window.menuBar().actions()
    if existing:
        window.menuBar().insertMenu(existing[0], files)
    else:
        window.menuBar().addMenu(files)
    files.addSeparator()
    files.addAction(action_icon('window'), 'Open main BiVOSIA window', files.open_main)
    add_settings_menu(window)
    help_menu = HelpMenu(window)
    window.menuBar().addMenu(help_menu)
    return files, help_menu
