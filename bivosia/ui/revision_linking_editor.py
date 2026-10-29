"""Combined zone revision and mask-linking editor.

Examples
--------
>>> from bivosia.ui.revision_linking_editor import RevisionLinkingWindow
>>> from PySide6.QtWidgets import QApplication
>>> app = QApplication.instance() or QApplication([])  # doctest: +SKIP
>>> from bivosia.dataset.layout import Dataset
>>> editor = RevisionLinkingWindow(Dataset("experiment"))  # doctest: +SKIP
>>> editor.showMaximized()  # doctest: +SKIP
>>> app.exec()  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMenu, QMessageBox, QProgressDialog, QPushButton, QSpinBox,
    QScrollArea, QTabWidget, QVBoxLayout, QWidget,
)

from bivosia.ui.icons import action_icon
from bivosia.config import ANALYSIS_DEFAULTS
from bivosia.ui.dock_panel import DockPanel, PanelHost, PanelSplitter
from bivosia.ui.link_editor import LinkDiagram
from bivosia.ui.link_colors import LinkColors, boundary_status
from bivosia.ui.plot_models import PlotInteractions, ReferenceLine, SeriesGroup, SeriesSpec
from bivosia.ui.plot_styles import measure_labels
from bivosia.ui.player import FrameKeyNavigation, PlayerCursorController
from bivosia.ui.quality_plot import GraphTimeline, GraphPanel
from bivosia.ui.revision_plots import editor_rows
from bivosia.ui.review_video_viewer import ReviewVideoViewer
from bivosia.ui.viewer_plot_config import FeaturePlotController
from bivosia.ui.zone_plot import ZoneTimeline, TimelineOverview, ZonePanel
from bivosia.processing.export import export_masks
from bivosia.analysis.linking import inverse, swap_endpoint, validation_errors
from bivosia.ui.playback import zone_loop, link_loops, upcoming_frames
from bivosia.processing.revision import RevisionData
from bivosia.dataset.session import RevisionSession
from bivosia.dataset.layout import Dataset
from bivosia.dataset.video import VideoBuffer


class ExportWorker(QThread):
    progress = Signal(int, int)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, session, data, path):
        super().__init__()
        self.session, self.path = deepcopy(session), path
        self.dataset = deepcopy(data.dataset)
        self.quality, self.switches = data.quality, data.evidence.switch_frames

    def run(self):
        try:
            self.completed.emit(export_masks(self.session, self.dataset, self.path,
                                              quality=self.quality, switches=self.switches, progress=self.progress.emit,
                                              cancelled=self.isInterruptionRequested))
        except Exception as error:
            self.failed.emit(str(error))


class RevisionLinkingWindow(QMainWindow):
    def __init__(self, dataset=None, annotation=None):
        super().__init__()
        self.statusBar()
        self.data = self.session = self.detected = None
        self.zone_id = self.job_id = None
        self.draft = None
        self.active_cursor = "zone"
        self.active_viewer = "Zone"
        self._updating = False
        self._drag = None
        self._coloring = None
        self.warnings = []
        self.dataset = dataset
        self.result_windows = []
        self.export_worker = None
        self.autosave = QTimer(self)
        self.autosave.setSingleShot(True)
        self.autosave.timeout.connect(lambda: self._run(self.save))
        self.setWindowTitle("Zone editor")
        self.resize(1650, 1050)
        self._build_ui()
        self.frame_keys = FrameKeyNavigation(
            self, self._step_frame, lambda: self.active_viewer == "Link"
        )
        if dataset is not None:
            self.directory_edit.setText(str(dataset.directory))
            self.storage_folder_edit.setText(next(iter(dataset.settings.get("storage_folders", [])), ""))
            self.load_data(annotation)

    def _button(self, text, callback, layout, tooltip=""):
        button = QPushButton(text)
        button.setToolTip(tooltip)
        button.clicked.connect(lambda: self._run(callback))
        layout.addWidget(button)
        return button

    def _run(self, callback):
        try:
            callback()
        except Exception as error:
            self._stop_players()
            QMessageBox.warning(self, "Revision editor", str(error))

    def _build_ui(self):
        body = PanelSplitter(Qt.Orientation.Horizontal)
        sidebar = QWidget()
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(6, 6, 6, 6)
        files = QGridLayout()
        self.directory_edit = QLineEdit()
        self.storage_folder_edit = QLineEdit()
        self.annotation_edit = QLineEdit()
        for row, (label, control) in enumerate((("Dataset", self.directory_edit),
                                               ("Storage", self.storage_folder_edit),
                                               ("Annotation", self.annotation_edit))):
            files.addWidget(QLabel(label), row, 0)
            files.addWidget(control, row, 1)
            control.textChanged.connect(control.setToolTip)
        choose = QPushButton("…")
        choose.setToolTip("Choose dataset folder")
        choose.setFixedWidth(25)
        choose.clicked.connect(self._choose_dataset)
        files.addWidget(choose, 0, 2)
        load = QPushButton("Open")
        load.setIcon(action_icon('open'))
        load.setToolTip("Load the dataset")
        load.clicked.connect(lambda: self._run(self.load_data))
        files.addWidget(load, 0, 3)
        self.directory_edit.returnPressed.connect(lambda: self._run(self.load_data))
        choose_storage = QPushButton("Open")
        choose_storage.setIcon(action_icon('open'))
        choose_storage.setToolTip("Choose storage folder")
        choose_storage.clicked.connect(self._choose_storage_folder)
        files.addWidget(choose_storage, 1, 2, 1, 2)
        open_file = QPushButton("Open")
        open_file.setIcon(action_icon('open'))
        open_file.clicked.connect(lambda: self._run(self.open_annotations))
        files.addWidget(open_file, 2, 2, 1, 2)
        files.setColumnStretch(1, 1)
        side.addLayout(files)
        row = QHBoxLayout()
        self._button("New", self.new_annotations, row, "New annotation file")
        self.save_button = self._button("Save", self.save, row)
        self._button("Save as…", self.save_as, row)
        side.addLayout(row)
        settings = QGridLayout()
        self.seed, self.recover = QDoubleSpinBox(), QDoubleSpinBox()
        for column, (label, spin, value) in enumerate((
            ("Seed", self.seed, ANALYSIS_DEFAULTS['zone_detection']['theta_quality']),
            ("Recovery", self.recover, ANALYSIS_DEFAULTS['zone_detection']['theta_recovery']),
        )):
            spin.setRange(-1, 1)
            spin.setDecimals(2)
            spin.setSingleStep(.01)
            spin.setValue(value)
            settings.addWidget(QLabel(label), 0, column)
            settings.addWidget(spin, 1, column)
        self.gap = QSpinBox()
        self.gap.setRange(0, 100000)
        self.gap.setValue(ANALYSIS_DEFAULTS['zone_detection']['theta_gap'])
        settings.addWidget(QLabel("Gap"), 0, 2)
        settings.addWidget(self.gap, 1, 2)
        side.addLayout(settings)
        row = QHBoxLayout()
        self.preview_button = self._button("Preview", self.preview_detection, row)
        self.start_button = self._button("Start", self.start_revision, row, "Start or resume annotation")
        self.undo_button = self._button("Undo", lambda: self._history(False), row)
        self.redo_button = self._button("Redo", lambda: self._history(True), row)
        side.addLayout(row)
        self.tabs = QTabWidget()
        self.zone_list, self.boundary_list = QListWidget(), QListWidget()
        self.tabs.setMinimumHeight(160)
        self.tabs.addTab(self.zone_list, "Zones")
        self.tabs.addTab(self.boundary_list, "Boundaries")
        self.tabs.currentChanged.connect(self._tab_changed)
        self.zone_list.currentItemChanged.connect(self._zone_item)
        self.zone_list.itemDoubleClicked.connect(self._zone_item)
        self.boundary_list.currentItemChanged.connect(self._boundary_item)
        side.addWidget(self.tabs, 1)
        form = QFormLayout()
        self.start_frame, self.end_frame = QSpinBox(), QSpinBox()
        self.source = QComboBox()
        self.source.addItems(["forward", "reverse", "invalid"])
        for title, field in (("Start", self.start_frame), ("End", self.end_frame), ("Masks", self.source)):
            form.addRow(title, field)
        side.addLayout(form)
        self.source.currentTextChanged.connect(lambda value: self._run(lambda: self._change_zone_source(value)))
        row = QHBoxLayout()
        self._button("Apply", lambda: self._edit("bounds", start=self.start_frame.value(), end=self.end_frame.value()), row)
        self._button("Add", lambda: self._edit("add", start=self.start_frame.value(), end=self.end_frame.value()), row)
        side.addLayout(row)
        row = QHBoxLayout()
        self._button("Split", lambda: self._edit("split", frame=self.players["zone"].frame), row, "Split at the zone cursor")
        self._button("Join", self._join, row, "Join with the next zone")
        self._button("Delete", lambda: self._edit("delete"), row)
        side.addLayout(row)
        row = QHBoxLayout()
        self._button("Review →", self.review_zone, row, "Mark this zone reviewed and select the next zone")
        self._button("Next link →", self.next_boundary, row, "Next boundary needing review")
        side.addLayout(row)
        self.diagram_holder = QVBoxLayout()
        side.addLayout(self.diagram_holder)
        self.diagram = None
        row = QHBoxLayout()
        self._button("Hungarian", self.repropose, row, "Recompute the Hungarian linking proposal")
        self._button("Approve →", self.approve_link, row, "Approve this link and select the next boundary")
        side.addLayout(row)
        row = QHBoxLayout()
        self._button("Validate", self.validate, row)
        self._button("Export…", self.export_global, row, "Export globally relabeled masks")
        self.result_button = self._button("Result", self.open_result, row, "Open the global mask viewer")
        side.addLayout(row)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(sidebar)
        body.addWidget(scroll)
        panel_host = PanelHost()
        body.addWidget(panel_host)
        self.viewers = {
            "Zone": ReviewVideoViewer((("zone", "#ff6576"),)),
            "Link": ReviewVideoViewer((("left", "#62db8d"), ("right", "#ff9754"))),
        }
        self.players = {name: player for viewer in self.viewers.values() for name, player in viewer.players.items()}
        self.player_cursors = PlayerCursorController(
            self.players.values(),
            on_cursor_moved=lambda key, frame: self._run(
                lambda: self._anchor_changed(key, frame)
            ),
        )
        for title, viewer in self.viewers.items():
            viewer.activated.connect(self._select_cursor)
            viewer.displayChanged.connect(lambda: self._run(self._render_frames))
        for name, player in self.players.items():
            player.frameRequested.connect(lambda frame, playback, key=name: self._run(lambda: self._anchor_changed(key, frame, playback)))
            player.playingChanged.connect(lambda playing, key=name: self._run(lambda: self._playing_changed(key, playing)))
            player.rangeChanged.connect(lambda key=name: self._range_changed(key))
            player.sourceChanged.connect(lambda value, key=name: self._run(lambda: self._direction_changed(key, value)))
        body.setStretchFactor(1, 1)
        body.setSizes((380, 1260))
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(4, 4, 6, 6)
        self.plots_holder = PanelSplitter(Qt.Orientation.Vertical)
        self.plots_holder.setChildrenCollapsible(False)
        plot_layout.addWidget(self.plots_holder, 1)
        self.message = QLabel("Open a dataset to begin.")
        self.message.setWordWrap(True)
        plot_layout.addWidget(self.message)
        self.panel_docks = {}
        for title, widget in (*self.viewers.items(), ("Plots", plot_panel)):
            dock = DockPanel("" if title == "Plots" else title, widget, panel_host)
            panel_host.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, dock)
            self.panel_docks[title] = dock
        panel_host.splitDockWidget(self.panel_docks["Zone"], self.panel_docks["Plots"], Qt.Orientation.Vertical)
        panel_host.tabifyDockWidget(self.panel_docks["Zone"], self.panel_docks["Link"])
        panel_host.setTabPosition(Qt.DockWidgetArea.AllDockWidgetAreas, QTabWidget.TabPosition.North)
        panel_host.resizeDocks([self.panel_docks["Zone"], self.panel_docks["Plots"]], [500, 400], Qt.Orientation.Vertical)
        for title in self.viewers:
            self.panel_docks[title].visibilityChanged.connect(lambda visible, key=title: self._viewer_visible(key, visible))
        self.panel_docks["Zone"].raise_()
        self.setCentralWidget(body)
        for keys, callback in ((QKeySequence.StandardKey.Undo, lambda: self._history(False)),
                               (QKeySequence.StandardKey.Redo, lambda: self._history(True)),
                               (QKeySequence.StandardKey.Save, self.save)):
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.activated.connect(lambda cb=callback: self._run(cb))
        for pane in self.viewers["Link"].grid.panes.values():
            self._connect_link_drag(pane)

    def _choose_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "Dataset folder", self.directory_edit.text())
        if path:
            self.directory_edit.setText(path)
            self.storage_folder_edit.clear()

    def _choose_storage_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Storage folder", self.storage_folder_edit.text())
        if path:
            self.storage_folder_edit.setText(path)

    def _leave_session(self):
        if self.session and self.session.dirty:
            self.save()
        self.autosave.stop()
        self._stop_players()

    def load_data(self, annotation=None):
        self._leave_session()
        self.message.setText("Loading cached evidence and video masks…")
        QApplication.processEvents()
        directory = Path(self.directory_edit.text()).expanduser().resolve()
        dataset = self.dataset
        if dataset is None or dataset.directory != directory:
            dataset = Dataset(directory)
        storage = self.storage_folder_edit.text()
        folders = dataset.settings.get("storage_folders", [])
        if storage != next(iter(folders), ""):
            dataset.settings["storage_folders"] = [storage, *folders[1:]] if storage else []
        data = RevisionData(dataset)
        self.dataset = dataset
        if self.data:
            self.data.close()
        self.data, self.session, self.zone_id, self.job_id = data, None, None, None
        self.link_colors = LinkColors(data)
        self.setWindowTitle(f"Zone editor · {data.paths['original'].stem}")
        self.draft = None
        for player in self.players.values():
            player.configure(data.count, data.videos.source_fps)
        for spin in (self.start_frame, self.end_frame):
            spin.blockSignals(True)
            spin.setRange(0, data.count - 1)
            spin.blockSignals(False)
        self.directory_edit.setText(str(data.directory))
        self.storage_folder_edit.setText(next(iter(data.dataset.settings.get("storage_folders", [])), ""))
        self._build_plots()
        self._settings_enabled(True)
        self.preview_detection()
        data.dataset.save()
        try:
            selected = annotation or data.dataset.annotation
        except ValueError as error:
            self.message.setText(str(error))
            return
        if selected:
            self._load_annotation(selected)

    def _settings_enabled(self, enabled):
        for widget in (self.seed, self.recover, self.gap, self.preview_button, self.start_button):
            widget.setEnabled(enabled)

    def _build_plots(self):
        while self.plots_holder.count():
            widget = self.plots_holder.widget(0)
            widget.setParent(None)
            widget.deleteLater()
        interactions = PlotInteractions(seek=True)
        self.overview = TimelineOverview(self.data.count, interactions=interactions)
        self.zone_plot = ZoneTimeline(self.data.count, interactions=interactions)
        self.quality_plot = GraphTimeline(self.data.count, series=[
            SeriesSpec("margin", "Margin quality", self.data.quality, color="#f2df5c"),
            SeriesSpec("assigned_iou", "Assigned mean IoU", self.data.assigned_iou, color="#4fd1c5")])
        self.quality_plot.fit_shared_y_range(include=(0, 1))
        self.quality_panel = GraphPanel(self.quality_plot, checkable=True, title="Quality")
        self.zone_panel = ZonePanel(self.zone_plot)
        self.plots_holder.addWidget(self.overview)
        self.plots_holder.addWidget(self.zone_panel)
        self.plots_holder.addWidget(self.quality_panel)
        self.feature_plot = GraphTimeline(self.data.count, "Features")
        self.feature_plot.setMinimumHeight(95)
        labels = measure_labels(self.data.measures)
        self.feature_controller = FeaturePlotController(self.feature_plot, {}, self.data.measures, labels)
        groups = tuple(SeriesGroup(key, label, tuple(
            spec.key for (direction, metric), spec in self.feature_controller.specs.items()
            if metric == key and direction in ("FORWARD", "REVERSE")))
            for key, label in labels.items())
        groups = tuple(group for group in groups if group.series_keys)
        self.feature_panel = GraphPanel(self.feature_plot, checkable=True, groups=groups, title="Features")
        self.feature_panel.set_reference_lines([], suffix="F solid · R dashed")
        self.feature_panel.setVisible(bool(groups))
        self.plots_holder.addWidget(self.feature_panel)
        self.plots_holder.setSizes((100, 200, 160, 160))
        self.plots = (self.overview, self.zone_plot, self.quality_plot, self.feature_plot)
        self.player_cursors.set_plots(
            line=(self.quality_plot, self.feature_plot),
            brackets=(self.overview, self.zone_plot),
        )
        self.zone_plot.setMinimumHeight(125)
        self.quality_plot.setMinimumHeight(95)
        for plot in self.plots[1:]:
            plot.viewChanged.connect(self.overview.set_view)
            for other in self.plots[1:]:
                if other is not plot:
                    plot.viewChanged.connect(other.set_view)
            self.overview.windowMoved.connect(plot.set_view)
        for plot in self.plots:
            plot.setToolTip("Click the plot to move the active playback cursor. Drag brackets to adjust the loop. Right-click: zone menu. "
                            "Double-click a zone to edit it; double-click a square boundary handle to link. "
                            "Alt+drag a zone or its edges to move or resize it. "
            "Ctrl+scroll / drag: quality y-axis. Ctrl+right-click: reset y-axis.")
            plot.cursorSelected.connect(self._plot_cursor_selected)
            plot.frameClicked.connect(lambda frame: self._run(lambda: self._plot_seek(frame)))
            plot.frameContextRequested.connect(self._open_zone_menu)
        if self.diagram is not None:
            self.diagram_holder.removeWidget(self.diagram)
            self.diagram.deleteLater()
        self.diagram = LinkDiagram(self.data.n)
        self._connect_link_drag(self.diagram)
        self.diagram_holder.addWidget(self.diagram)

    def preview_detection(self):
        if self.data is None:
            raise ValueError("Load data first")
        if self.session is not None:
            raise ValueError("Start a new annotation set to change detector settings")
        self.detected = self.data.detect(self.seed.value(), self.recover.value(), self.gap.value())
        path = self.data.dataset.new_annotation(self.seed.value(), self.recover.value())
        self.annotation_edit.setText(self.data.dataset.relative(path))
        self._refresh()

    def start_revision(self):
        if self.data is None:
            raise ValueError("Load data first")
        if self.session is not None:
            return
        path = self.data.dataset.annotation_path(self.annotation_edit.text())
        if path.exists():
            self._load_annotation(path)
            return
        path = self.data.dataset.annotation_path(path, write=True)
        self.detected = self.data.detect(self.seed.value(), self.recover.value(), self.gap.value())
        session = RevisionSession.create(self.data.header(self.seed.value(), self.recover.value(), self.gap.value()),
                                         self.detected.final_zods, self.data.missing_forward, self.data.missing_reverse, self.data.n)
        session.save(path)
        self.data.dataset.select_annotation(path)
        self.session = session
        self._settings_enabled(False)
        self._refresh()
        if session.zones:
            self.select_zone(session.zones[0].id)

    def new_annotations(self):
        self._leave_session()
        self.session, self.zone_id, self.job_id, self.draft = None, None, None, None
        self._settings_enabled(True)
        self.preview_detection()
        self.message.setText("Choose thresholds and an annotation filename, then press Start.")

    def open_annotations(self):
        if self.data is None:
            raise ValueError("Load the dataset first")
        path, _ = QFileDialog.getOpenFileName(self, "Open annotation session", str(self.data.dataset.annotations), "Annotation files (*.json)")
        if path:
            self._leave_session()
            self._load_annotation(path)

    def _load_annotation(self, path):
        path = self.data.dataset.annotation_path(path)
        session = RevisionSession.load(path, self.data.missing_forward, self.data.missing_reverse, self.data.n)
        self.session = session
        self.data.dataset.select_annotation(path)
        self.annotation_edit.setText(self.data.dataset.relative(session.path))
        self.seed.setValue(session.header["seed"])
        self.recover.setValue(session.header["recover"])
        self.gap.setValue(session.header["gap"])
        self.detected = self.data.detect(self.seed.value(), self.recover.value(), self.gap.value())
        self.zone_id = self.job_id = self.draft = None
        self._settings_enabled(False)
        self._refresh()
        if session.zones:
            self.select_zone(session.zones[0].id)

    def save(self):
        if self.session is None:
            return
        try:
            path = self.data.dataset.annotation_path(self.annotation_edit.text(), write=True)
            self.session.save(path)
            self.data.dataset.select_annotation(path)
            self.statusBar().showMessage(f"Saved {self.session.path.name} · revision {self.session.revision}")
        except Exception as error:
            self.statusBar().showMessage(f"NOT SAVED: {error}")
            self.message.setText(f"NOT SAVED: {error}")
            raise

    def save_as(self):
        if self.session is None:
            raise ValueError("Press Start first")
        path, _ = QFileDialog.getSaveFileName(self, "Save annotation session", str(self.session.path), "JSON (*.json)")
        if path:
            path = self.data.dataset.annotation_path(path, write=True)
            self.session.save(path)
            self.data.dataset.select_annotation(path)
            self.annotation_edit.setText(self.data.dataset.relative(path))

    def _changed(self):
        self.autosave.start(400)
        self._refresh()

    def _refresh(self):
        if self.data is None or self.detected is None:
            return
        self._updating = True
        try:
            self.zone_list.clear()
            for i, zone in enumerate(self.session.zones if self.session else ()):
                item = QListWidgetItem(f"{i+1} · {zone.start:,}–{zone.end:,} · {zone.source} · {zone.status}")
                item.setData(Qt.ItemDataRole.UserRole, zone.id)
                self.zone_list.addItem(item)
                if zone.id == self.zone_id:
                    self.zone_list.setCurrentItem(item)
                    self.start_frame.setValue(zone.start)
                    self.end_frame.setValue(zone.end)
                    self.source.setCurrentText(zone.source)
            self.boundary_list.clear()
            links = self.link_colors.links(self.session) if self.session else None
            for job in self.session.jobs if self.session else ():
                record = self.session.links[job.id]
                status = boundary_status(record, links.automatic.get(job.id))
                item = QListWidgetItem(f"{job.left.end:,} → {job.right.start:,} · {status}" + (" · invalid bridge" if job.skipped_frames else ""))
                item.setData(Qt.ItemDataRole.UserRole, job.id)
                self.boundary_list.addItem(item)
                if job.id == self.job_id:
                    self.boundary_list.setCurrentItem(item)
            self.zone_plot.set_rows(editor_rows(self.data, self.detected, self.session,
                                    on_zone=lambda hit: self._run(lambda: self.select_zone(hit.interval.key)),
                                    on_boundary=lambda marker: self._run(lambda: self.select_boundary(marker.payload)),
                                    on_edit=lambda request: self._run(lambda: self._plot_edit(request))))
            self.overview.set_rows(editor_rows(self.data, self.detected, self.session, overview=True,
                                   on_zone=lambda hit: self._run(lambda: self.select_zone(hit.interval.key)),
                                   on_boundary=lambda marker: self._run(lambda: self.select_boundary(marker.payload))))
            self._coloring = None
            width = max(self.zone_plot.required_left_margin(), self.quality_plot.required_left_axis_width())
            for plot in self.plots:
                plot.set_left_margin(width)
            self.zone_plot.set_selection(("annotations", "zones", self.zone_id) if self.zone_id else None)
            self.quality_plot.set_reference_lines([ReferenceLine("seed", self.seed.value(), "#ffd43b"),
                                                   ReferenceLine("recovery", self.recover.value(), "#f4f4f4")])
            self.quality_panel.set_reference_lines([(f"seed = {self.seed.value():.2f}", "#ffd43b"),
                                                     (f"recovery = {self.recover.value():.2f}", "#f4f4f4")])
            self.undo_button.setEnabled(bool(self.session and self.session.undo_stack))
            self.redo_button.setEnabled(bool(self.session and self.session.redo_stack))
            self.result_button.setEnabled(bool(self.session and self.session.export))
        finally:
            self._updating = False
        self._render_frames()

    def refresh_font_geometry(self):
        """Realign the plot axes after the application font changes."""
        required = ("plots", "overview", "zone_plot", "quality_plot", "feature_plot")
        if not all(hasattr(self, name) for name in required):
            return
        width = max(
            self.overview.required_left_margin(),
            self.zone_plot.required_left_margin(),
            self.quality_plot.required_left_axis_width(),
            self.feature_plot.required_left_axis_width(),
        )
        for plot in self.plots:
            plot.set_left_margin(width)
        self.zone_plot.setMinimumHeight(max(125, self.zone_plot.minimum_content_height()))
        self.overview.setMinimumHeight(self.overview.minimum_content_height())
        for plot in self.plots:
            plot.updateGeometry()

    def _zone_item(self, item, previous=None):
        if item is not None and not self._updating:
            self._run(lambda: self.select_zone(item.data(Qt.ItemDataRole.UserRole)))

    def _boundary_item(self, item, previous):
        if item is not None and not self._updating:
            self._run(lambda: self.select_boundary(item.data(Qt.ItemDataRole.UserRole)))

    def _tab_changed(self, index):
        linking = self.tabs.widget(index) is self.boundary_list
        if hasattr(self, "panel_docks"):
            title = "Link" if linking else "Zone"
            self.panel_docks[title].raise_()
            self._select_cursor(self.viewers[title].active_side)

    def _exit_linking(self):
        self.viewers["Link"].stop()
        self.job_id = self.draft = self._drag = None
        self.warnings = []

    def select_zone(self, zone_id, focus=True):
        if self.session is None:
            return
        zone = next(z for z in self.session.zones if z.id == zone_id)
        self.zone_id = zone.id
        self.session.visit_zone(zone.id)
        self.autosave.start(400)
        self.viewers["Zone"].stop()
        self.tabs.setCurrentWidget(self.zone_list)
        self.panel_docks["Zone"].raise_()
        self._select_cursor("zone")
        self.players["zone"].set_loop(*zone_loop(zone, self.data.count))
        if focus:
            self.players["zone"].set_frame(zone.start)
            self._focus(*self.players["zone"].bounds)
        self._refresh()

    def _job(self):
        return next((j for j in self.session.jobs if j.id == self.job_id), None) if self.session else None

    def select_boundary(self, job_id):
        self.viewers["Link"].stop()
        self.tabs.setCurrentWidget(self.boundary_list)
        self.job_id = job_id
        job = self._job()
        if job is None:
            raise ValueError("Boundary no longer exists after the last zone edit")
        self.draft, self.warnings = self.data.linking.draft(job, self.session.links[job.id], self.seed.value())
        # An unchanged approved mapping retains its status.
        self.session.set_draft(job.id, self.draft, history=False)
        loops = link_loops(job, self.session.zones, self.data.count)
        for side in ("left", "right"):
            self.players[side].set_frame(self.draft[f"{side}_frame"])
            self.players[side].set_loop(*loops[side])
        self.panel_docks["Link"].raise_()
        self._select_cursor("left")
        self._focus(min(a for a, b in loops.values()), max(b for a, b in loops.values()))
        self._changed()

    def _focus(self, start, end):
        pad = max(8, (end - start + 1) * .2)
        for plot in self.plots:
            plot.set_view(start - pad, end + pad)

    def _edit(self, operation, **kwargs):
        if self.session is None:
            raise ValueError("Press Start first")
        old_frame = self.players["zone"].frame
        old_link = deepcopy(self.session.links.get(self.job_id))
        self.session.edit(self.zone_id, operation,
                          link_transport=lambda job, record, old: self.data.linking.transport_source_edit(job, record, old, self.seed.value()),
                          **kwargs)
        if operation == "add":
            self.zone_id = next(z.id for z in self.session.zones if z.start == kwargs["start"])
        if self.zone_id not in {z.id for z in self.session.zones}:
            self.zone_id = next((z.id for z in self.session.zones if z.end >= old_frame), None)
        if self.job_id and self.session.links.get(self.job_id) != old_link:
            jobs = self.session.jobs
            job = self._job() or next((j for j in jobs if j.right.start >= old_frame), jobs[-1] if jobs else None)
            if job:
                selected_tab = self.tabs.currentWidget()
                self.select_boundary(job.id)
                self.tabs.setCurrentWidget(selected_tab)
            else:
                self._exit_linking()
        self._changed()

    def _plot_edit(self, request):
        self.zone_id = request.hit.interval.key
        zone = next(z for z in self.session.zones if z.id == self.zone_id)
        if request.action == "move":
            self._edit("bounds", start=request.frame, end=request.frame + zone.end - zone.start)
        else:
            self._edit("bounds", start=request.frame if request.action == "start" else zone.start,
                       end=request.frame if request.action == "end" else zone.end)

    def _change_zone_source(self, source):
        if not self._updating and self.session and self.zone_id:
            self._edit("source", source=source)

    def _join(self):
        if not self.session or not self.zone_id:
            raise ValueError("Select a zone")
        self._edit("join", source=self.source.currentText())

    def _zone_menu(self, frame):
        menu = QMenu(self)
        boundary = next((j for j in self.session.jobs if j.left.end == frame and j.right.start == frame + 1), None)
        action = menu.addAction("Link boundary" if boundary else "Add linking boundary")
        action.setToolTip(f"Link {frame:,} → {frame + 1:,} without changing zones")
        action.setEnabled(frame < self.data.count - 1)
        action.triggered.connect(lambda: self._run(lambda: self.select_boundary(boundary.id) if boundary else self._add_link_frame(frame)))
        if frame in self.session.link_frames:
            menu.addAction("Remove linking boundary", lambda: self._run(lambda: self._remove_link_frame(frame)))
        menu.addSeparator()
        zones = self.session.zones
        index = next((i for i, z in enumerate(zones) if z.start <= frame <= z.end), None)
        if index is None:
            menu.addAction("Add zone here", lambda: self._run(lambda: self._edit("add", start=frame, end=frame)))
            return menu
        zone = zones[index]

        def edit(operation, **kwargs):
            self.zone_id = zone.id
            self._edit(operation, **kwargs)

        for label, enabled, callback in (
            ("Split here", frame < zone.end, lambda: edit("split", frame=frame)),
            ("Join left", index > 0, lambda: edit("join", side="left", source=zone.source)),
            ("Join right", index + 1 < len(zones), lambda: edit("join", side="right", source=zone.source)),
            ("Delete", True, lambda: edit("delete")),
        ):
            action = menu.addAction(label)
            action.triggered.connect(lambda checked=False, cb=callback: self._run(cb))
            action.setEnabled(enabled)
            if label.startswith("Join"):
                action.setToolTip("Includes any gap; keeps this zone's mask direction")
        return menu

    def _add_link_frame(self, frame):
        job_id = self.session.set_link_frame(frame, True)
        self.select_boundary(job_id)

    def _remove_link_frame(self, frame):
        self.session.set_link_frame(frame, False)
        if self.job_id == f"frame:{frame}":
            self._exit_linking()
            self.tabs.setCurrentWidget(self.zone_list)
        self._changed()

    def _open_zone_menu(self, frame, position):
        if self.session is None:
            self.statusBar().showMessage("Press Start to edit zones.", 3000)
            return
        menu = self._zone_menu(frame)
        menu.exec(position)
        menu.deleteLater()

    def _select_cursor(self, side):
        owner = side.split(":")[0]
        if owner not in self.players:
            return
        self.active_cursor = side
        self.active_viewer = "Zone" if owner == "zone" else "Link"
        self.viewers[self.active_viewer].active_side = owner
        for key, player in self.players.items():
            player.select.setChecked(key == owner)
        self._update_cursors()

    def _plot_cursor_selected(self, key):
        zone = key.split(":")[0] == "zone"
        self.tabs.setCurrentWidget(self.zone_list if zone else self.boundary_list)
        self.panel_docks["Zone" if zone else "Link"].raise_()
        self._select_cursor(key)

    def _viewer_visible(self, title, visible):
        if visible:
            self._select_cursor(self.viewers[title].active_side)
            if self.data and hasattr(self, "plots"):
                self._run(lambda: self._render_frames(title))
        self._update_cursors()

    def _update_cursors(self):
        plots = getattr(self, "plots", ())
        self.player_cursors.set_plots(
            line=tuple(plot for plot in plots if not isinstance(plot, ZoneTimeline)),
            brackets=tuple(plot for plot in plots if isinstance(plot, ZoneTimeline)),
        )
        for plot in plots:
            plot.set_active_cursor(self._cursor_for_plot(plot))

    def _cursor_for_plot(self, plot):
        return self.active_cursor if isinstance(plot, ZoneTimeline) else self.active_cursor.split(":")[0]

    def _plot_seek(self, frame):
        key = self.active_cursor.split(":")[0]
        self._select_cursor(key)
        self._anchor_changed(key, frame)

    def _step_frame(self, side, offset):
        if self.data is not None:
            player = self.players[side]
            player.seek(player.frame + offset)

    def _stop_players(self):
        for viewer in self.viewers.values():
            viewer.stop()

    def _prime_player(self, key):
        player = self.players[key]
        if self.data and player.play.isChecked():
            frames = upcoming_frames(player.frame, *player.bounds, player.looping, VideoBuffer.capacity)
            self.data.prefetch(frames, stream=key)

    def _range_changed(self, key):
        self._prime_player(key)
        self._update_cursors()

    def _playing_changed(self, key, playing):
        if self.data is None:
            self.players[key].stop()
            return
        if playing:
            self._select_cursor(key)
            self._prime_player(key)
        else:
            self.data.stop_prefetch(key)

    def review_zone(self):
        if not self.zone_id:
            return
        self._edit("review")
        index = next(i for i, z in enumerate(self.session.zones) if z.id == self.zone_id)
        if index + 1 < len(self.session.zones):
            self.select_zone(self.session.zones[index + 1].id)

    def next_boundary(self):
        if not self.session:
            raise ValueError("Press Start first")
        links = self.link_colors.links(self.session)
        pending = [j for j in self.session.jobs if j.id not in links.mappings]
        if not pending:
            self.message.setText("All boundaries linked. Review remaining zones, then validate/export.")
            return
        self.tabs.setCurrentWidget(self.boundary_list)
        job = self._job()
        frame = job.right.start if job else self.players["right"].frame
        self.select_boundary(next((j.id for j in pending if j.right.start > frame), pending[0].id))

    def _anchor_changed(self, side, frame, playback=False):
        if self._updating or self.data is None:
            return
        key, _, endpoint = side.partition(":")
        if key not in self.players:
            return
        player = self.players[key]
        if endpoint:
            player.move_cursor(side, frame)
            return
        if not playback:
            player.stop()
        player.set_frame(frame)
        if not playback and key != "zone" and self._job() and self.draft:
            self.draft = self._displayed_draft()
            self.warnings = self.data.linking.anchor_warnings(self._job(), self.draft, self.seed.value())
            self.session.set_draft(self.job_id, self.draft, history=False)
            self.autosave.start(400)
            item = self.boundary_list.currentItem()
            if item is not None:
                job = self._job()
                status = boundary_status(self.session.links[job.id], self.link_colors.links(self.session).automatic.get(job.id))
                item.setText(f"{job.left.end:,} → {job.right.start:,} · {status}")
        title = "Zone" if key == "zone" else "Link"
        if not playback or self.viewers[title].isVisible():
            self._render_frames(title)
        else:
            self._update_cursors()
        if playback:
            self._prime_player(key)

    def _displayed_draft(self):
        return {**self.draft, **{f"{side}_frame": self.players[side].frame for side in ("left", "right")}}

    def _direction_changed(self, side, value):
        if self._updating or side == "zone" or not self._job() or not self.draft:
            return
        self.viewers["Link"].stop()
        self.draft, self.warnings = self.data.linking.change_anchors(self._job(), self._displayed_draft(), self.seed.value(), **{f"{side}_source": value})
        self.draft["manual"] = True
        self.session.set_draft(self.job_id, self.draft)
        self._changed()

    def _render_frames(self, viewer_name=None):
        if self.data is None:
            return
        self._updating = True
        try:
            job = self._job()
            source = next((z.source for z in self.session.zones if z.id == self.zone_id), "forward") if self.session else "forward"
            sources = {side: self.draft[f"{side}_source"] if job and self.draft else "forward" for side in ("left", "right")}
            grid = self.viewers["Link"].grid
            mapping = (grid.mapping if self._drag else self.draft["mapping"]) if job and self.draft else tuple(range(self.data.n))
            live = {**self._displayed_draft(), "mapping": mapping} if self._drag else self.draft
            preview = self.link_colors.preview(self.session, job, live, self.seed.value()) if self.session else None
            color_changed = preview is not self._coloring
            if color_changed:
                self._coloring = preview
                self.overview.set_rows(editor_rows(self.data, self.detected, self.session, overview=True,
                    on_zone=lambda hit: self._run(lambda: self.select_zone(hit.interval.key)),
                    on_boundary=lambda marker: self._run(lambda: self.select_boundary(marker.payload)),
                    regions=preview.regions if preview else (), links=preview))
            for title, viewer in self.viewers.items():
                if viewer_name is None or title == viewer_name or (color_changed and viewer.color_mode.currentData() == "global"):
                    selected_sources = sources if title == "Link" else {"zone": source if source != "invalid" else "forward"}
                    viewer.render(self.data, selected_sources, self.seed.value(), preview, mapping,
                                  linking=bool(title == "Link" and job and self.draft))
            selected = {side: grid.panes[side, sources[side]] for side in ("left", "right")}
            self.diagram.set_masks({side: pane.colors for side, pane in selected.items()},
                                   {side: tuple(center is not None for center in pane.centers) for side, pane in selected.items()})
            self.diagram.set_mapping(mapping)
            self.diagram.setEnabled(bool(job))
            grid.set_mapping(mapping)
            self._update_cursors()
            summary = f"{self.data.directory.name} · {len(self.session.zones) if self.session else len(self.detected.final_zods)} zones"
            if self.session:
                done = sum(r["status"] == "approved" for r in self.session.links.values())
                automatic = len(self.link_colors.links(self.session).automatic)
                summary += f" · {done} approved, {automatic} auto / {len(self.session.links)} boundaries · revision {self.session.revision}"
            if preview:
                summary += f" · Color preview: {len(set(preview.components.values()))} linked regions"
            if not job and source == "invalid":
                summary += " · Selected zone INVALID; masks shown only for inspection"
            if job:
                status = boundary_status(self.session.links[job.id], self.link_colors.links(self.session).automatic.get(job.id))
                summary += f" · boundary status: {status}"
            warnings = (self.warnings if job else []) + (list(preview.errors) if preview else [])
            self.message.setText(summary + ("\n" + " · ".join(warnings) if warnings else ""))
        finally:
            self._updating = False

    def _connect_link_drag(self, widget):
        widget.dragStarted.connect(self._drag_started)
        widget.dragTarget.connect(self._drag_target)
        widget.dragFinished.connect(lambda valid: self._run(lambda: self._drag_finished(valid)))

    def _drag_started(self, side, raw_label):
        if self._job() and self.draft:
            self.viewers["Link"].stop()
            initial = tuple(self.draft["mapping"])
            fixed_label = inverse(initial)[raw_label] if side == "right" else initial[raw_label]
            self._drag = (initial, side, fixed_label)

    def _drag_target(self, raw_label):
        if self._drag:
            initial, side, fixed = self._drag
            mapping = swap_endpoint(initial, side, fixed, raw_label)
            grid = self.viewers["Link"].grid
            if mapping != grid.mapping:
                grid.set_mapping(mapping)
                self._render_frames()

    def _drag_finished(self, valid):
        if not self._drag:
            return
        initial, _, _ = self._drag
        mapping = self.viewers["Link"].grid.mapping if valid else initial
        self._drag = None
        if tuple(mapping) != tuple(initial):
            self.draft = {**self._displayed_draft(), "mapping": list(mapping), "manual": True}
            self.session.set_draft(self.job_id, self.draft)
            self._changed()
        else:
            self._render_frames()

    def repropose(self):
        if not self._job() or not self.draft:
            raise ValueError("Select a boundary first")
        self.viewers["Link"].stop()
        record = {"draft": {**self._displayed_draft(), "mapping": None, "manual": False}}
        self.draft, self.warnings = self.data.linking.draft(self._job(), record, self.seed.value())
        self.session.set_draft(self.job_id, self.draft)
        self._changed()

    def approve_link(self):
        if not self._job() or not self.draft:
            raise ValueError("Select a boundary first")
        self.viewers["Link"].stop()
        draft = self._displayed_draft()
        mapping, warnings = self.data.linking.canonical_mapping(self._job(), draft, self.seed.value())
        warnings = list(dict.fromkeys(self.warnings + warnings))
        if warnings and QMessageBox.question(self, "Verify uncertain link", "\n".join(warnings) + "\n\nApprove this manually verified assignment?") != QMessageBox.StandardButton.Yes:
            return
        self.draft = draft
        self.session.set_draft(self.job_id, draft)
        self.session.approve(self.job_id, mapping)
        self.save()
        self._refresh()
        self.next_boundary()

    def _history(self, redo):
        if self.session:
            self._stop_players()
            self.session.redo() if redo else self.session.undo()
            self._exit_linking()
            self._changed()

    def validate(self):
        if not self.session:
            raise ValueError("Press Start first")
        errors = validation_errors(self.session, self.link_colors.links(self.session))
        QMessageBox.information(self, "Linking validation", "\n".join(errors[:40]) + (f"\n… {len(errors)} issues total" if len(errors) > 40 else "") if errors else "All zones reviewed and boundaries automatically linked or approved. Ready to export.")

    def export_global(self):
        if not self.session:
            raise ValueError("Press Start first")
        errors = validation_errors(self.session, self.link_colors.links(self.session))
        if errors:
            raise ValueError("Resolve these before export:\n" + "\n".join(errors[:20]))
        path, _ = QFileDialog.getSaveFileName(self, "New globally relabeled masks (source masks remain unchanged)",
                    str(self.data.dataset.linked_masks(self.session.path)), "HDF5 (*.h5)")
        if not path:
            return
        self.data.dataset.export_name(path)
        self.save()
        self.progress = QProgressDialog("Streaming forward, reverse and preferred global masks…", "Cancel", 0, self.data.count, self)
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.export_worker = ExportWorker(self.session, self.data, path)
        self.export_worker.progress.connect(lambda value, total: self.progress.setValue(value))
        self.progress.canceled.connect(self.export_worker.requestInterruption)
        self.export_worker.completed.connect(self._export_done)
        self.export_worker.failed.connect(self._export_failed)
        self.export_worker.start()
        self.progress.show()

    def _export_done(self, result):
        self.progress.close()
        self.session.export = result
        self.session.dirty = True
        self._run(self.save)
        self._refresh()
        self._run(self.open_result)

    def _export_failed(self, message):
        self.progress.close()
        QMessageBox.warning(self, "Export incomplete", message)

    def open_result(self):
        if not self.session or not self.session.export:
            raise ValueError("No global mask export for this annotation set")
        if self.session.export["fingerprint"] != self.session.fingerprint():
            raise ValueError("This export is stale after annotation changes. Validate and export again.")
        from bivosia.ui.global_mask_viewer import GlobalMaskViewer
        window = GlobalMaskViewer(self.data.paths["original"], self.data.dataset.export_path(self.session.export["path"]), self.data.n)
        window.dataset = self.data.dataset
        self.result_windows.append(window)
        window.showMaximized()

    def closeEvent(self, event):
        if self.export_worker is not None and self.export_worker.isRunning():
            QMessageBox.information(self, "Export running", "Cancel the export and wait for it to stop before closing.")
            event.ignore()
            return
        try:
            self._leave_session()
        except Exception as error:
            QMessageBox.warning(self, "Annotations not saved", str(error))
            event.ignore()
            return
        if self.data:
            self.data.close()
        for dock in self.panel_docks.values():
            dock.hide()
        for window in self.result_windows:
            window.close()
        self.frame_keys.close()
        event.accept()
