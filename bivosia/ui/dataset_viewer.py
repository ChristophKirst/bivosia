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

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (QDialog, QFileDialog, QHeaderView, QHBoxLayout, QLabel, QMenu,
                               QMessageBox, QPushButton, QStatusBar,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from bivosia.ui.icons import action_icon
from bivosia.dataset.io.registry import file_inspector, format_handler, format_spec, format_suffixes
from bivosia.dataset.io.interfaces import FileMetadata
from bivosia.dataset.workflow.jobs import is_job_path
from bivosia.dataset.paths import folder_path
from bivosia.processing.storage.file_operations import (copy_selected_file, delete_selected_file,
                                               file_locations, move_selected_file)
from bivosia.processing.storage.folder_operations import (completed_job_entries, delete_folder_entries, folder_entries,
                                                 plan_folder_transfer, transfer_folder)
from bivosia.dataset.storage.inventory import scan_locations
from bivosia.ui.dataset_locations_dialog import DatasetLocationsDialog
from bivosia.ui.consolidation_dialog import ConsolidationDialog
from bivosia.processing.storage.consolidation import plan_consolidation, plan_input_sync, consolidate
from bivosia.ui.remote_settings import remote_settings
from bivosia.ui.colors import identity_colors
from bivosia.ui.dock_panel import PanelSplitter
from bivosia.ui.legend_panel import LegendPanel, legend_groups_html
from bivosia.ui.plot_models import LegendGroup
from bivosia.ui.task import Task
from bivosia.ui.text_preview import TextPreview, create_text_view
from bivosia.ui.video_preview import VideoPreview
from bivosia.ui.window_manager import notify_dataset_changed


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

    transfer_status = Signal(str)

    def __init__(self, dataset, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.scan_task = None
        self.info_task = None
        self.text_task = None
        self.file_task = None
        self._sync_callback = None
        self._pending_sync = None
        self._pending_file = None
        self._file_error = ''
        self._entries = ()
        self._folder_transfer = None
        self._job_cleanup = None
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
        self.consolidate_button = QPushButton('Consolidate…')
        self.consolidate_button.setIcon(action_icon('files'))
        self.consolidate_button.clicked.connect(self._consolidate_menu)
        toolbar.addWidget(self.consolidate_button)
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
        self.transfer_status.connect(self.status.showMessage)
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

    def prepare_processing_sync(self, operation, location, callback, *, annotation=None, settings=None):
        if self.file_task is not None or self._pending_file is not None or self._sync_callback is not None:
            raise ValueError('Finish the current dataset file operation before preparing another job')
        profiles = tuple(remote_settings().installations.values())
        self._sync_callback = callback
        def plan():
            self._consolidation_plan = plan_input_sync(self.dataset, location, profiles, operation,
                annotation=annotation, settings=settings, progress=self.transfer_status.emit)
        self._pending_sync = plan
        self._start_sync()

    def _start_sync(self):
        if self._pending_sync is not None and self.scan_task is None:
            function, self._pending_sync = self._pending_sync, None
            self._start_file_operation(None, function, 'Checking required processing inputs…')

    def _sync_finished(self, success, message=''):
        callback, self._sync_callback = self._sync_callback, None
        if callback is not None:
            callback(success, message)

    def _consolidate_menu(self):
        if self.scan_task is not None or self.file_task is not None or self._pending_file is not None:
            return
        menu = QMenu(self)
        for location in file_locations(self.dataset):
            label = (location['host'] + ':' if location['host'] else '') + location['path']
            menu.addAction(label).triggered.connect(lambda checked=False, target=location: self._plan_consolidation(target))
        menu.exec(self.consolidate_button.mapToGlobal(self.consolidate_button.rect().bottomLeft()))
        menu.deleteLater()

    def _plan_consolidation(self, location):
        profiles = tuple(remote_settings().installations.values())
        self._consolidation_plan = None
        def plan():
            result = plan_consolidation(self.dataset, location, profiles, self.transfer_status.emit)
            self._consolidation_plan = result
        self._start_file_operation(None, plan, 'Checking files before consolidation…')

    def _confirm_consolidation(self):
        plan, self._consolidation_plan = self._consolidation_plan, None
        folder, self._folder_transfer = self._folder_transfer, None
        if self._closed or plan is None:
            return
        choices = {}
        if plan.conflicts:
            options = {'action': folder[2], 'scope': f'contents of {folder[0]}'} if folder else {}
            dialog = ConsolidationDialog(plan, self, **options)
            accepted = dialog.exec() == QDialog.DialogCode.Accepted
            choices = dialog.choices()
            dialog.deleteLater()
            if not accepted:
                self._sync_finished(False)
                self.refresh()
                return
        profiles = tuple(remote_settings().installations.values())
        if folder:
            relative, entries, action = folder
            self._start_file_operation(None, lambda: transfer_folder(
                self.dataset, entries, plan, choices, profiles, move=action == 'Move',
                progress=self.transfer_status.emit), f'{action}: {relative}…')
            return
        self._start_file_operation(None, lambda: consolidate(self.dataset, plan, choices, profiles,
                                                           self.transfer_status.emit),
                                   'Consolidating dataset…' if plan.complete else 'Synchronizing required inputs…')

    def _file_menu(self, position):
        if self.scan_task is not None or self.file_task is not None or self._pending_file is not None:
            return
        item = self.tree.itemAt(position)
        relative = item.data(0, Qt.ItemDataRole.UserRole + 1) if item is not None else None
        if relative is not None:
            self._folder_menu(position, Path(relative))
            return
        entry = item.data(0, Qt.ItemDataRole.UserRole) if item is not None else None
        if entry is None or entry.is_directory or entry.error:
            return
        menu = QMenu(self)
        if not is_job_path(entry.relative):
            self._destination_menu(menu, entry, 'Move', move_selected_file)
            self._destination_menu(menu, entry, 'Copy', copy_selected_file)
        menu.addAction('Delete…').triggered.connect(lambda: self._delete_file(entry))
        menu.exec(self.tree.viewport().mapToGlobal(position))
        menu.deleteLater()

    def _folder_menu(self, position, relative):
        try:
            entries = folder_entries(self._entries, relative)
        except ValueError as error:
            self.status.showMessage(str(error))
            return
        if not entries:
            return
        menu = QMenu(self)
        for label in (() if is_job_path(relative) else ('Move', 'Copy')):
            destinations = menu.addMenu(label)
            for location in file_locations(self.dataset):
                host, path = location['host'], location['path']
                action = destinations.addAction(f'{host}:{path}' if host else path)
                action.setEnabled(any(entry.host != host or entry.path != Path(path) / entry.relative
                                      for entry in entries))
                action.triggered.connect(lambda checked=False, target=location, operation=label:
                                         self._transfer_folder(relative, entries, target, operation))
        if relative == folder_path('jobs'):
            menu.addAction('Clean up completed jobs…').triggered.connect(lambda: self._plan_job_cleanup(entries))
        menu.addAction('Delete…').triggered.connect(lambda: self._delete_folder(relative, entries))
        menu.exec(self.tree.viewport().mapToGlobal(position))
        menu.deleteLater()

    def _transfer_folder(self, relative, entries, location, action):
        profiles = tuple(remote_settings().installations.values())
        self._consolidation_plan = None
        self._folder_transfer = relative, entries, action
        def plan():
            self._consolidation_plan = plan_folder_transfer(self.dataset, entries, location, profiles,
                                                           self.transfer_status.emit)
        self._start_file_operation(None, plan, f'Checking {relative} before {action.lower()}…')

    def _delete_folder(self, relative, entries):
        paths = sorted({(entry.host + ':' if entry.host else '') + str(entry.path)
                        for entry in entries if entry.relative == relative})
        # Synthetic parents can represent externally referenced files without a folder entry.
        if not paths:
            paths = sorted({str(entry.source) for entry in entries})
        count = sum(not entry.is_directory for entry in entries)
        self._confirm_delete_entries(entries, 'Delete folder contents',
                                     f'Permanently delete {count} files and their folders under {relative}?\n\n'
                                     + '\n'.join(paths))

    def _plan_job_cleanup(self, entries):
        profiles = tuple(remote_settings().installations.values())
        def plan():
            self._job_cleanup = completed_job_entries(self.dataset, entries, profiles)
        self._start_file_operation(None, plan, 'Checking completed jobs…')

    def _confirm_job_cleanup(self):
        entries, self._job_cleanup = self._job_cleanup, None
        if self._closed:
            return
        if not entries:
            self._refresh_message = 'No completed jobs to clean up'
            self.refresh()
            return
        jobs = [entry for entry in entries if entry.is_directory and
                entry.relative.parent == folder_path('jobs')]
        paths = sorted((entry.host + ':' if entry.host else '') + str(entry.path) for entry in jobs)
        count = sum(not entry.is_directory for entry in entries)
        self._confirm_delete_entries(entries, 'Clean up completed jobs',
                                     f'Permanently delete {len(jobs)} completed jobs and their logs ({count} files)?\n'
                                     'Analysis results are kept.\n\n' + '\n'.join(paths))

    def _confirm_delete_entries(self, entries, title, message):
        dialog = QMessageBox(QMessageBox.Icon.Warning, title, message, parent=self)
        dialog.setTextFormat(Qt.TextFormat.PlainText)
        dialog.setDetailedText('\n'.join((entry.host + ':' if entry.host else '') + str(entry.path)
                                        for entry in entries))
        delete = dialog.addButton('Delete', QMessageBox.ButtonRole.DestructiveRole)
        cancel = dialog.addButton(QMessageBox.StandardButton.Cancel)
        dialog.setDefaultButton(cancel)
        dialog.exec()
        confirmed = dialog.clickedButton() is delete
        dialog.deleteLater()
        if confirmed:
            profiles = tuple(remote_settings().installations.values())
            self._start_file_operation(None, lambda: delete_folder_entries(
                self.dataset, entries, profiles, self.transfer_status.emit), 'Deleting selected contents…')
        else:
            self.refresh_button.setEnabled(True)
            self.status.clearMessage()

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
        windows = ([self._text_windows[entry.path]] if entry is not None and entry.path in self._text_windows else
                   list(self._text_windows.values()) if entry is None else [])
        for window in windows:
            window.close()
        self.tree.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.locations_button.setEnabled(False)
        self.consolidate_button.setEnabled(False)
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
        self.consolidate_button.setEnabled(True)
        self._refresh_message = self._file_error
        if self._job_cleanup is not None and not self._file_error:
            QTimer.singleShot(0, self._confirm_job_cleanup)
        elif getattr(self, '_consolidation_plan', None) is not None and not self._file_error:
            QTimer.singleShot(0, self._confirm_consolidation)
        else:
            self._folder_transfer = None
            self._job_cleanup = None
            self._sync_finished(not self._file_error, self._file_error)
            notify_dataset_changed(self.dataset)
            self.refresh()

    def _task(self, function, ready, failed, finished):
        task = Task(function, self)
        task.connect_callbacks(ready=ready, failed=failed, finished=finished)
        return task

    def _show_files(self, entries):
        self._entries = tuple(entries)
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
                folders[relative].setData(0, Qt.ItemDataRole.UserRole + 1, str(relative))
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
        self._start_sync()

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
        self._sync_finished(False)
        self._pending_sync = None
        self._closed = True
        self._selected = None
        self._text_queue.clear()
        self._clear_preview()
        for window in tuple(self._text_windows.values()):
            window.close()
        event.accept()
