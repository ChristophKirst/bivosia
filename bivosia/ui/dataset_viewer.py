"""Browse dataset files and metadata with read-only text and video previews.

Examples
--------
>>> from PySide6.QtWidgets import QApplication
>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.ui.dataset_viewer import DatasetWidget
>>> app = QApplication.instance() or QApplication([])  # doctest: +SKIP
>>> widget = DatasetWidget(Dataset('/path/to/dataset'))  # doctest: +SKIP
>>> widget.show()  # doctest: +SKIP
>>> app.exec()  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from collections import deque
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QFileDialog, QHeaderView, QHBoxLayout, QLabel, QMenu,
                               QMessageBox, QPushButton, QStatusBar,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from bivosia.ui.icons import action_icon
from bivosia.dataset.formats import file_inspector, format_handler, format_spec, format_suffixes
from bivosia.dataset.inventory import FileMetadata
from bivosia.processing.file_operations import (copy_selected_file, delete_selected_file,
                                               file_locations, move_selected_file)
from bivosia.processing.remote_inventory import scan_locations
from bivosia.ui.dataset_locations_dialog import DatasetLocationsDialog
from bivosia.ui.remote_settings import remote_settings
from bivosia.ui.colors import identity_colors
from bivosia.ui.dock_panel import PanelSplitter
from bivosia.ui.legend_panel import LegendPanel, legend_groups_html
from bivosia.ui.plot_models import LegendGroup
from bivosia.ui.task import Task
from bivosia.ui.text_preview import TextPreview, create_text_view
from bivosia.ui.video_preview import VideoPreview


def _file_size(size):
    if size is None:
        return ''
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if size < 1024 or unit == 'TiB':
            return f'{size:,} B' if unit == 'B' else f'{size:.1f} {unit}'
        size /= 1024


def _duration(seconds):
    """Format video duration as hours, minutes, seconds and milliseconds.

    >>> from bivosia.ui.dataset_viewer import _duration
    >>> _duration(3661.25)
    '01:01:01.250'
    >>> _duration(None)
    'Unavailable'
    """
    if seconds is None:
        return 'Unavailable'
    seconds, milliseconds = divmod(round(seconds * 1000), 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}'


class DatasetWidget(QWidget):
    """Dataset file tree with previews and actions for selected files.

    File statistics appear first; background reads fill the Info column.
    Select text or video to preview it below the tree. Text can open in a separate window.
    """

    def __init__(self, dataset, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.scan_task = None
        self.info_task = None
        self.text_task = None
        self.file_task = None
        self._pending_file = None
        self._file_error = ''
        self._refresh_message = ''
        self._generation = 0
        self._headers = {}
        self._items = {}
        self._inspectors = {}
        self._header_queue = deque()
        self._text_queue = deque()
        self._text_windows = {}
        self._preview_view = None
        self._preview = None
        self._selected = None
        self._closed = False
        self.setWindowTitle(f'Dataset — {dataset.directory.name}')
        self.resize(1100, 700)

        layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        self.locations_label = QLabel()
        self.locations_label.setTextFormat(Qt.TextFormat.RichText)
        self.locations_label.setWordWrap(True)
        self.locations_label.setMinimumWidth(0)
        self.locations_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        toolbar.addWidget(LegendPanel.containing(self.locations_label), 1)
        self.locations_button = QPushButton('Locations…')
        self.locations_button.setIcon(action_icon('files'))
        self.locations_button.clicked.connect(self.edit_locations)
        toolbar.addWidget(self.locations_button)
        self.refresh_button = QPushButton('Refresh')
        self.refresh_button.setIcon(action_icon('refresh'))
        self.refresh_button.clicked.connect(self.refresh)
        toolbar.addWidget(self.refresh_button)
        layout.addLayout(toolbar)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(('Name', 'Info', 'Location', 'Modified', 'Size'))
        self.tree.header().setStretchLastSection(False)
        for column in range(self.tree.columnCount()):
            self.tree.header().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.tree.currentItemChanged.connect(self._selection_changed)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._file_menu)

        self.preview_panel = QWidget()
        self.preview_layout = QVBoxLayout(self.preview_panel)
        self.preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_toolbar = QHBoxLayout()
        self.preview_name = QLabel()
        self.preview_name.setTextFormat(Qt.TextFormat.PlainText)
        preview_toolbar.addWidget(self.preview_name, 1)
        self.open_text_button = QPushButton('Open in window')
        self.open_text_button.setIcon(action_icon('window'))
        self.open_text_button.setEnabled(False)
        self.open_text_button.clicked.connect(self._open_text)
        preview_toolbar.addWidget(self.open_text_button)
        self.preview_layout.addLayout(preview_toolbar)

        self.splitter = PanelSplitter(Qt.Orientation.Vertical)
        self.splitter.addWidget(self.tree)
        self.splitter.addWidget(self.preview_panel)
        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 1)
        self.preview_panel.hide()
        layout.addWidget(self.splitter, 1)
        self.status = QStatusBar(self)
        self.status.messageChanged.connect(lambda message: self.status.setToolTip(message))
        layout.addWidget(self.status)
        self.refresh()

    def refresh(self):
        """Rescan the supplied Dataset and discard cached file headers."""
        if self.scan_task is not None or self.file_task is not None or self._pending_file is not None:
            return
        self.dataset.reload()
        self._generation += 1
        self._headers.clear()
        self._items.clear()
        self._inspectors.clear()
        self._header_queue.clear()
        self._text_queue.clear()
        self._selected = None
        self._clear_preview()
        self.tree.clear()
        self.status.showMessage('Reading file information…')
        self.refresh_button.setEnabled(False)
        profiles = tuple(remote_settings().installations.values())
        self.scan_task = self._task(lambda: scan_locations(self.dataset, profiles), self._show_files,
                                    self.status.showMessage, self._scan_finished)
        self.scan_task.start()

    def edit_locations(self):
        DatasetLocationsDialog(self.dataset, self).exec()
        self.refresh()

    def _file_menu(self, position):
        if self.scan_task is not None or self.file_task is not None or self._pending_file is not None:
            return
        item = self.tree.itemAt(position)
        entry = item.data(0, Qt.ItemDataRole.UserRole) if item is not None else None
        if entry is None or entry.is_directory or entry.error:
            return
        menu = QMenu(self)
        self._destination_menu(menu, entry, 'Move', move_selected_file)
        self._destination_menu(menu, entry, 'Copy', copy_selected_file)
        menu.addAction('Delete…').triggered.connect(lambda: self._delete_file(entry))
        menu.exec(self.tree.viewport().mapToGlobal(position))
        menu.deleteLater()

    def _destination_menu(self, menu, entry, label, operation):
        destinations = menu.addMenu(label)
        for location in file_locations(self.dataset):
            host, path = location['host'], location['path']
            action = destinations.addAction(f'{host}:{path}' if host else path)
            action.setEnabled(host != entry.host or Path(path) / entry.relative != entry.path)
            action.triggered.connect(lambda checked=False, target=location:
                                     self._transfer_file(entry, target, label, operation))
        destinations.addSeparator()
        destinations.addAction('Choose folder…').triggered.connect(
            lambda: self._choose_file_folder(entry, label, operation))

    def _choose_file_folder(self, entry, label, operation):
        path = QFileDialog.getExistingDirectory(self, f'{label} to dataset folder', str(self.dataset.directory))
        if path:
            self._transfer_file(entry, {'host': '', 'path': path}, label, operation)

    def _transfer_file(self, entry, location, label, operation):
        profiles = tuple(remote_settings().installations.values())
        progress = 'Moving' if label == 'Move' else 'Copying'
        self._start_file_operation(entry, lambda: operation(self.dataset, entry, location, profiles),
                                   f'{progress} {entry.relative.name}…')

    def _delete_file(self, entry):
        path = f'{entry.host}:{entry.path}' if entry.host else str(entry.path)
        dialog = QMessageBox(QMessageBox.Icon.Warning, 'Delete file',
                             f'Permanently delete this file?\n\n{path}', parent=self)
        dialog.setTextFormat(Qt.TextFormat.PlainText)
        delete = dialog.addButton('Delete', QMessageBox.ButtonRole.DestructiveRole)
        cancel = dialog.addButton(QMessageBox.StandardButton.Cancel)
        dialog.setDefaultButton(cancel)
        dialog.exec()
        confirmed = dialog.clickedButton() is delete
        dialog.deleteLater()
        if confirmed:
            profiles = tuple(remote_settings().installations.values())
            self._start_file_operation(entry, lambda: delete_selected_file(self.dataset, entry, profiles),
                                       f'Deleting {entry.relative.name}…')

    def _start_file_operation(self, entry, function, message):
        self._generation += 1
        self._selected = None
        self._header_queue.clear()
        self._text_queue.clear()
        self._clear_preview()
        window = self._text_windows.get(entry.path)
        if window is not None:
            window.close()
        self.tree.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.locations_button.setEnabled(False)
        self.status.showMessage(message)
        self._pending_file = function
        self._file_error = ''
        self._start_pending_file()

    def _start_pending_file(self):
        if self._pending_file is None or self.info_task is not None or self.text_task is not None:
            return
        self.file_task = self._task(self._pending_file, lambda result: None,
                                    self._file_failed, self._file_finished)
        self._pending_file = None
        self.file_task.start()

    def _file_failed(self, message):
        self._file_error = message

    def _file_finished(self):
        self.file_task = None
        self.tree.setEnabled(True)
        self.locations_button.setEnabled(True)
        self._refresh_message = self._file_error
        self.refresh()

    def _task(self, function, ready, failed, finished):
        # Application ownership lets background file work finish after a window closes.
        application = QApplication.instance()
        task = Task(function, application)
        self.destroyed.connect(task.requestInterruption)
        application.aboutToQuit.connect(task.wait)
        task.ready.connect(ready)
        task.failed.connect(failed)
        task.finished.connect(finished)
        task.finished.connect(task.deleteLater)
        return task

    def _show_files(self, entries):
        roots = tuple(dict.fromkeys((*self.dataset.roots, *(entry.source for entry in entries if entry.source))))
        colors = dict(zip(roots, identity_colors(len(roots), .58)))
        self.locations_label.setText('<br>'.join(
            legend_groups_html((LegendGroup(items=((str(root), color.name()),)),))
            for root, color in colors.items()))

        def row(parent, name):
            item = QTreeWidgetItem(parent, [name])
            for column in range(self.tree.columnCount()):
                horizontal = Qt.AlignmentFlag.AlignRight if column == 4 else Qt.AlignmentFlag.AlignLeft
                item.setTextAlignment(column, horizontal | Qt.AlignmentFlag.AlignTop)
            return item

        root = row(self.tree, self.dataset.directory.name)
        folders = {Path(): root}
        locations = {}

        def folder(relative):
            if relative not in folders:
                folders[relative] = row(folder(relative.parent), relative.name)
            return folders[relative]

        for entry in entries:
            if entry.is_directory:
                item = folder(entry.relative)
                paths = locations.setdefault(entry.relative, [])
                paths.append(f'{entry.host}:{entry.path}' if entry.host else str(entry.path))
                item.setToolTip(0, '\n'.join(paths))
            else:
                item = row(folder(entry.relative.parent), entry.relative.name)
                if not entry.host:
                    self._items[entry.path] = item
                item.setData(0, Qt.ItemDataRole.UserRole, entry)
                root = entry.source
                item.setBackground(2, colors[root])
                item.setToolTip(2, str(root))
                item.setText(4, _file_size(entry.size))
                if entry.size is not None:
                    item.setToolTip(4, f'{entry.size:,} bytes')
                if entry.modified is not None:
                    item.setText(3, datetime.fromtimestamp(entry.modified).strftime('%Y-%m-%d %H:%M:%S'))
                item.setToolTip(0, f'{entry.host}:{entry.path}' if entry.host else str(entry.path))
                if not entry.error and not entry.host:
                    try:
                        inspector = file_inspector(entry.path)
                        if inspector is not None:
                            self._inspectors[entry.path] = inspector
                            self._header_queue.append(entry.path)
                    except Exception as error:
                        self._headers[entry.path] = FileMetadata(), str(error)
                        self._show_headers(entry.path)
            if entry.error:
                item.setText(4, 'Unavailable')
                item.setToolTip(4, item.toolTip(4) + '\n' + entry.error)
        self.tree.expandToDepth(1)
        self.status.showMessage(self._refresh_message)
        self._refresh_message = ''
        self._next_header()

    def _scan_finished(self):
        self.scan_task = None
        self.refresh_button.setEnabled(True)

    def _selection_changed(self, current, previous):
        self._selected = None
        self._text_queue.clear()
        self._clear_preview()
        self.status.clearMessage()
        if current is None:
            return
        entry = current.data(0, Qt.ItemDataRole.UserRole)
        if entry is None:
            self.status.showMessage(current.toolTip(4).strip())
            return
        if entry.error:
            self.status.showMessage(entry.error)
            return
        if entry.host:
            self.status.showMessage(f'Remote file · {entry.host}:{entry.path}')
            return
        self._selected = entry.path
        if entry.path in self._headers:
            self.status.showMessage(self._headers[entry.path][1])
        elif entry.path in self._header_queue:
            self._header_queue.remove(entry.path)
            self._header_queue.appendleft(entry.path)
        self._next_header()
        video = entry.path.suffix.lower() in format_suffixes('video')
        if video or entry.path.suffix.lower() in format_suffixes('text'):
            self.preview_name.setText(entry.path.name)
            self.preview_panel.show()
            if self.splitter.sizes()[1] == 0:
                self.splitter.setSizes([400, 240])
            self.open_text_button.setVisible(not video)
            if video:
                self._preview_view = VideoPreview(entry.path)
                self._preview_view.failed.connect(self.status.showMessage)
                self.preview_layout.addWidget(self._preview_view, 1)
                return
            active = self.text_task.request if self.text_task is not None else None
            if (self._generation, entry.path) != active:
                self._text_queue.append(entry.path)
            self._next_text()

    def _next_header(self):
        if self.info_task is not None or not self._header_queue or self._closed:
            return
        path = self._header_queue.popleft()
        inspector = self._inspectors[path]
        self._items[path].setText(1, 'Reading…')
        request = self._generation, path
        self.info_task = self._task(lambda: inspector(path), self._headers_ready,
                                    self._headers_failed, self._info_finished)
        self.info_task.request = request
        self.info_task.start()

    def _headers_ready(self, metadata):
        generation, path = self.info_task.request
        if generation == self._generation:
            self._headers[path] = metadata, ''
            self._show_headers(path)

    def _headers_failed(self, message):
        generation, path = self.info_task.request
        if generation == self._generation:
            self._headers[path] = FileMetadata(), message
            self._show_headers(path)

    def _show_headers(self, path):
        metadata, error = self._headers[path]
        item = self._items[path]
        lines = [f'{name}: {shape}, {dtype}' for name, shape, dtype in metadata.arrays]
        if metadata.frames is not None:
            lines.append(f'Frames: {metadata.frames:,}; Duration: {_duration(metadata.duration)}')
        text = '\n'.join(lines)
        tooltip = text
        if metadata.frames is not None:
            tooltip += '\nFrame count reported by the video file. '
            tooltip += ('Duration estimated from frame count and frame rate (HH:MM:SS.mmm).'
                        if metadata.duration is not None else 'Frame rate is unavailable.')
        item.setText(1, 'Unavailable' if error else text)
        item.setToolTip(1, error or tooltip)
        if path == self._selected:
            self.status.showMessage(error)

    def _info_finished(self):
        self.info_task = None
        self._start_pending_file()
        self._next_header()

    def _clear_preview(self):
        self._preview = None
        if self._preview_view is not None:
            self.preview_layout.removeWidget(self._preview_view)
            self._preview_view.close()
            self._preview_view.deleteLater()
            self._preview_view = None
        self.open_text_button.setEnabled(False)
        self.preview_name.clear()
        self.preview_panel.hide()

    def _open_text(self):
        if self._preview is None:
            return
        path, text, syntax = self._preview
        window = self._text_windows.get(path)
        if window is not None:
            window.showNormal()
            window.raise_()
            window.activateWindow()
            return
        window = TextPreview(path.name, text, syntax, self)
        self._text_windows[path] = window
        window.destroyed.connect(lambda: self._text_windows.pop(path, None))
        window.show()

    def _next_text(self):
        if self.text_task is not None or not self._text_queue or self._closed:
            return
        path = self._text_queue.popleft()
        self.status.showMessage(f'Reading {path.name}…')
        self.text_task = self._task(lambda: format_handler('text', path)(path),
                                    self._text_ready, self._text_failed, self._text_finished)
        self.text_task.request = self._generation, path
        self.text_task.start()

    def _text_ready(self, text):
        generation, path = self.text_task.request
        if generation != self._generation or path != self._selected or self._closed:
            return
        syntax = format_spec('text', path)['syntax']
        self._preview = path, text, syntax
        self._preview_view = create_text_view(text, syntax)
        self.preview_layout.addWidget(self._preview_view, 1)
        self.open_text_button.setEnabled(True)
        self.status.clearMessage()

    def _text_failed(self, message):
        generation, path = self.text_task.request
        if generation == self._generation and path == self._selected and not self._closed:
            self.status.showMessage(message)

    def _text_finished(self):
        self.text_task = None
        self._start_pending_file()
        self._next_text()

    def showEvent(self, event):
        super().showEvent(event)
        self._closed = False
        self._next_header()

    def closeEvent(self, event):
        self._closed = True
        self._selected = None
        self._text_queue.clear()
        self._clear_preview()
        for window in tuple(self._text_windows.values()):
            window.close()
        event.accept()
