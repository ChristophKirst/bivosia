"""Standard mask, annotation, quality and feature viewer.

Examples
--------
With an existing dataset containing mask analysis results:

>>> from PySide6.QtWidgets import QApplication
>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.ui.recording_viewer import create_recording_viewer
>>> app = QApplication.instance() or QApplication([])  # doctest: +SKIP
>>> viewer = create_recording_viewer(Dataset("/path/to/dataset"))  # doctest: +SKIP
>>> viewer.showMaximized()  # doctest: +SKIP
>>> app.exec()  # doctest: +SKIP
"""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from bivosia.analysis.zones_of_disagreement import ZodEvidence
    from bivosia.ui.timeline_base import TimelineBase

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import (
    QImage,
    QKeyEvent,
    QKeySequence,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
import cv2
import numpy as np

from bivosia.config import ANALYSIS_DEFAULTS
from bivosia.analysis.zones_of_disagreement import (
    detect_zods, frame_event_signals, mask_count_seed_signal, zone_frame_summary,
)
from bivosia.analysis.mask_linking import Linker
from bivosia.dataset.video import RecordingStreams
from bivosia.processing.analysis.keypoints import calculate_keypoint_counts
from bivosia.ui.annotations import ZodView, annotate_zone_missing
from bivosia.ui.colors import bgr_colors
from bivosia.ui.dock_panel import PanelSplitter, PanelHost, DockPanel, add_view_menu
from bivosia.ui.legend_panel import LegendPanel
from bivosia.ui.link_colors import global_label_colors
from bivosia.ui.keypoints import configure_displacement_plot
from bivosia.ui.plot_models import ReferenceLine
from bivosia.ui.plot_styles import (
    METRIC_COLORS, QUALITY_METRIC_LABELS, STANDARD_QUALITY_PLOT_LABELS,
    QUALITY_METRIC_RANGES, QUALITY_METRIC_HIGHER_IS_BETTER,
)
from bivosia.ui.player import FrameKeyNavigation, FramePlayer, PlayerCursorController
from bivosia.ui.playback import padded_loop
from bivosia.ui.quality_plot import GraphTimeline, GraphPanel
from bivosia.ui.video_panels import (
    MaskDisplayControl, VideoViewGroup, ZoomableVideoLabel,
)
from bivosia.ui.viewer_plot_config import (
    compose_zone_tracks, viewer_zone_rows, configure_quality_plot, FeaturePlotController,
    keypoint_alignment_tracks, linked_zone_row,
)
from bivosia.ui.zone_plot import ZoneTimeline, TimelineOverview, ZonePanel
from bivosia.ui.timeline_group import TimelineGroup


ZONE_INDEX_ROLE = int(Qt.ItemDataRole.UserRole)
SORT_VALUE_ROLE = ZONE_INDEX_ROLE + 1


class _SortableTableItem(QTableWidgetItem):
    """Table item with independent display and native sort values."""

    def __lt__(self, other):
        left = self.data(SORT_VALUE_ROLE)
        right = other.data(SORT_VALUE_ROLE)
        if left is not None and right is not None:
            return left < right
        return super().__lt__(other)


class ZoneVideoViewer(QMainWindow):
    metric_colors = METRIC_COLORS
    quality_metric_labels = QUALITY_METRIC_LABELS
    standard_quality_labels = STANDARD_QUALITY_PLOT_LABELS
    quality_metric_ranges = QUALITY_METRIC_RANGES
    quality_higher_is_better = QUALITY_METRIC_HIGHER_IS_BETTER

    def __init__(
        self, paths: dict[str, Path], zones: list[ZodView], *,
        expected_masks: int | None = None, evidence: ZodEvidence | None = None,
        quality_arrays=None, feature_data=None, feature_labels=None, sparse_features=None,
        missing_signal=None, forward_missing_signal=None, reverse_missing_signal=None,
        missing_direction_known=None, zone_tracks=None, mask_palettes=None, keypoints=None,
        videos=None, global_links=None, link_seed=ANALYSIS_DEFAULTS['zones_of_disagreement']['theta_quality'], zone_rows=(),
        annotation_label='Hand', controls=None,
    ) -> None:
        super().__init__()
        self.videos = videos if videos is not None else RecordingStreams(paths)
        self.keypoints = keypoints
        self.alignment_tracks = (
            keypoint_alignment_tracks(keypoints, global_links, point_counts=calculate_keypoint_counts(
                keypoints.reader, node_ids=keypoints.selected_node_ids))
            if keypoints is not None and keypoints.alignment is not None and global_links is not None else {}
        )
        self.zones = list(zones)
        self.annotation_label = annotation_label
        self.controls = controls
        self.expected_masks = expected_masks
        self.frame_analysis = sparse_features or {}
        self.dense_frame_analysis = feature_data or {}
        self.measure_labels = feature_labels or {}
        self.measure_frame_analysis = self.dense_frame_analysis if self.measure_labels else {}
        self.extra_zone_tracks = dict(zone_tracks or {})
        self.extra_zone_rows = tuple(zone_rows)
        self.zone_tracks = dict(self.extra_zone_tracks)
        self.zod_evidence = evidence
        self.global_links = global_links
        self.link_seed = link_seed
        self.linking = None
        if global_links is not None:
            if evidence is None or expected_masks is None:
                raise ValueError('Global colors require mask counts and linking evidence')
            self.linking = Linker(self.videos.read, expected_masks, evidence.quality,
                                 evidence.missing, evidence.switch_frames)
        self.quality_metric_arrays = dict(quality_arrays or {})
        if evidence is not None:
            self.quality_metric_arrays.setdefault("zod_local_quality", evidence.quality)
        for key, values in self.quality_metric_arrays.items():
            if np.asarray(values).shape != (self.videos.frame_count,):
                raise ValueError(f"Quality measure {key} must match the video frame count")
        self.quality_labels = {key: label for key, label in self.quality_metric_labels.items()
                               if key in self.quality_metric_arrays}
        self.quality_plot_labels = {key: label for key, label in self.standard_quality_labels.items()
                                    if key in self.quality_metric_arrays}
        forward_missing = self.dense_frame_analysis.get("FORWARD", {}).get(
            "missing_masks"
        )
        reverse_missing = self.dense_frame_analysis.get("REVERSE", {}).get(
            "missing_masks"
        )
        self.zod_mask_count = (
            None
            if self.zod_evidence is None
            else mask_count_seed_signal(
                self.zod_evidence, forward_missing, reverse_missing
            )
        )
        if missing_signal is not None:
            if np.asarray(missing_signal).shape != (self.videos.frame_count,):
                raise ValueError("Missing-mask signal must match the video frame count")
            self.zod_mask_count = np.asarray(missing_signal, dtype=bool)
        self.forward_missing_signal = (
            forward_missing_signal if forward_missing_signal is not None else forward_missing
        )
        self.reverse_missing_signal = (
            reverse_missing_signal if reverse_missing_signal is not None else reverse_missing
        )
        self.missing_direction_known = missing_direction_known
        self.zones = annotate_zone_missing(
            self.zones, self.zod_mask_count, self.forward_missing_signal,
            self.reverse_missing_signal, self.missing_direction_known,
        )
        if self.zod_evidence is not None and len(self.zod_evidence.quality) != self.videos.frame_count:
            raise ValueError("Quality evidence must match the video frame count")
        if self.zod_evidence is not None:
            default_quality = self.quality_metric_arrays["zod_local_quality"]
            self._set_zod_detection(
                theta_quality=ANALYSIS_DEFAULTS['zones_of_disagreement']['theta_quality'],
                expand_to_baseline=ANALYSIS_DEFAULTS['zones_of_disagreement']['expand_to_baseline'],
                theta_recovery=ANALYSIS_DEFAULTS['zones_of_disagreement']['theta_recovery'],
                theta_gap=ANALYSIS_DEFAULTS['zones_of_disagreement']['theta_gap'],
                quality_signal=default_quality,
                recovery_signal=default_quality,
            )
        self.mask_palettes = mask_palettes or {"forward": None, "reverse": None}
        self.current_frame = self.zones[0].frames[0] if self.zones else 0
        self.playback_zone_pinned = False
        self.playback_zone_kind = "Zone"
        self.selected_zone_index = None
        self._last_images: dict[str, QImage] = {}
        self._initial_video_zoom = True
        self.setWindowTitle(f"Zone mask viewer — {paths['original'].parent.name}")
        self.resize(1200, 760)
        self._build_ui()
        self.frame_keys = FrameKeyNavigation(
            self, lambda _side, offset: self.player.seek(self.player.frame + offset)
        )
        if self.zones:
            self._select_zone_table_row(0)
        else:
            self.show_frame(0)


    def _set_zod_detection(
        self,
        *,
        theta_quality: float,
        expand_to_baseline: bool,
        theta_recovery: float,
        theta_gap: int,
        quality_signal: np.ndarray | None = None,
        recovery_signal: np.ndarray | None = None,
        quality_higher_is_better: bool = True,
        recovery_higher_is_better: bool = True,
    ) -> None:
        if self.zod_evidence is None:
            return
        result = detect_zods(
            self.zod_evidence,
            theta_quality,
            expand_to_baseline=expand_to_baseline,
            theta_recovery=theta_recovery,
            theta_gap=theta_gap,
            quality_signal=quality_signal,
            recovery_signal=recovery_signal,
            missing_signal=self.zod_mask_count,
            quality_higher_is_better=quality_higher_is_better,
            recovery_higher_is_better=recovery_higher_is_better,
        )
        self.zod_result = result


    def _display_zone_tracks(self):
        result = getattr(self, "zod_result", None)
        tracks = compose_zone_tracks(
            self.zone_tracks,
            missing_signal=self.zod_mask_count,
            forward_missing_signal=self.forward_missing_signal,
            reverse_missing_signal=self.reverse_missing_signal,
            missing_direction_known=self.missing_direction_known,
            switch_frames=(
                self.zod_evidence.switch_frames if self.zod_evidence is not None else None
            ),
            seed_intervals=(result.initial_zods if result is not None else None),
            auto_intervals=(result.final_zods if result is not None else None),
        )
        tracks.update(self.alignment_tracks)
        return tracks

    def _refresh_zone_plots(self) -> None:
        tracks = self._display_zone_tracks()

        def select_auto(hit):
            for start, end in tracks['Auto zones']:
                if start <= hit.frame <= end:
                    self._auto_zone_clicked('Auto zones', start, end)
                    return
            for timeline in (self.zone_timeline, self.timeline_overview):
                timeline.set_selection(None)

        callbacks = {
            "Hand labels": lambda hit: self._timeline_clicked(hit.interval.payload, hit.frame),
            "ZoD": select_auto,
        }
        detail_rows = viewer_zone_rows(
            tracks, self.zones,
            callbacks=callbacks, annotation_label=self.annotation_label,
        )
        overview_rows = viewer_zone_rows(
            tracks, self.zones, callbacks=callbacks,
            show_names=False, annotation_label=self.annotation_label,
        )
        if self.global_links is not None:
            row = linked_zone_row(self.global_links.regions, links=self.global_links, visible=False)
            detail_rows.append(row)
            overview_rows.append(row)
        self.zone_timeline.set_rows([*detail_rows, *self.extra_zone_rows])
        self.timeline_overview.set_rows([*overview_rows, *self.extra_zone_rows])

    def set_zone_tracks(self, tracks: dict[str, list[tuple[int, int]]]) -> None:
        """Register named external interval tracks without duplicating rendering."""
        for label in self.extra_zone_tracks:
            self.zone_tracks.pop(label, None)
        self.extra_zone_tracks = dict(tracks)
        self.zone_tracks.update(tracks)
        self._refresh_zone_plots()


    def _build_ui(self) -> None:
        root = QWidget(self)
        layout = QVBoxLayout(root)

        self.zone_table = QTableWidget(len(self.zones), 7)
        self.zone_table.setMinimumWidth(390)
        self.zone_table.setHorizontalHeaderLabels(
            ("Zone", "Start", "End", "Size", "Missing", "Switch", "Quality")
        )
        self.zone_table.verticalHeader().hide()
        self.zone_table.setAlternatingRowColors(True)
        self.zone_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.zone_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.zone_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        header = self.zone_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column, width in enumerate((68, 68, 55, 72, 52, 70), start=1):
            header.resizeSection(column, width)
        quality = self.quality_metric_arrays.get("zod_local_quality")
        events = frame_event_signals(
            self.zod_mask_count, self.forward_missing_signal, self.reverse_missing_signal,
            self.missing_direction_known,
            None if self.zod_evidence is None else self.zod_evidence.switch_frames)
        for row, zone in enumerate(self.zones):
            summary = zone_frame_summary(zone.frames, events=events, quality=quality)
            minimum_quality, has_switch = summary['minimum_quality'], summary['has_switch']
            values = self._zone_table_values(
                row, zone, minimum_quality, has_switch
            )
            sort_values = (
                row, *values[1:4], values[4], bool(values[5]),
                float("inf") if minimum_quality is None else minimum_quality,
            )
            for column, (value, sort_value) in enumerate(zip(values, sort_values)):
                item = _SortableTableItem(str(value))
                item.setData(ZONE_INDEX_ROLE, row)
                item.setData(SORT_VALUE_ROLE, sort_value)
                item.setToolTip(zone.details)
                if column in (4, 5):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.zone_table.setItem(row, column, item)
        self.zone_table.currentCellChanged.connect(
            lambda row, _column, _old_row, _old_column:
            self._zone_table_selection_changed(row)
        )
        self.zone_table.setSortingEnabled(True)
        self.zone_table.sortItems(0, Qt.SortOrder.AscendingOrder)
        sidebar = QVBoxLayout()
        sidebar.addWidget(self.zone_table, 2)
        self.sidebar = sidebar

        self.zod_checkbox = QCheckBox("ZoD detector")
        self.zod_checkbox.setEnabled(self.zod_evidence is not None)
        self.zod_checkbox.setToolTip(
            "Tune generalized unweighted-IoU ZoD detection"
            if self.zod_evidence is not None
            else "Pass both an IoU matrix and its Hungarian match array to enable this detector"
        )
        sidebar.addWidget(self.zod_checkbox)
        self.zod_panel = QWidget()
        zod_layout = QVBoxLayout(self.zod_panel)
        zod_layout.setContentsMargins(0, 0, 0, 0)
        self.zod_table = QTableWidget(7, 2)
        self.zod_table.setHorizontalHeaderLabels(("Parameter", "Value"))
        self.zod_table.verticalHeader().hide()
        self.zod_table.setMinimumHeight(235)
        self.zod_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.zod_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.zod_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        self.zod_quality_measure_combo = QComboBox()
        self.zod_recovery_measure_combo = QComboBox()
        for key, label in self.quality_labels.items():
            self.zod_quality_measure_combo.addItem(label, key)
            self.zod_recovery_measure_combo.addItem(label, key)
        default_metric = self.zod_quality_measure_combo.findData("zod_local_quality")
        if default_metric >= 0:
            self.zod_quality_measure_combo.setCurrentIndex(default_metric)
            self.zod_recovery_measure_combo.setCurrentIndex(default_metric)
        self.zod_quality_spin = QDoubleSpinBox()
        self.zod_quality_spin.setRange(-1.0, 1.0)
        self.zod_quality_spin.setSingleStep(0.01)
        self.zod_quality_spin.setDecimals(3)
        self.zod_quality_spin.setValue(ANALYSIS_DEFAULTS['zones_of_disagreement']['theta_quality'])
        self.zod_expand_checkbox = QCheckBox("Enabled")
        self.zod_expand_checkbox.setChecked(ANALYSIS_DEFAULTS['zones_of_disagreement']['expand_to_baseline'])
        self.zod_recovery_spin = QDoubleSpinBox()
        self.zod_recovery_spin.setRange(-1.0, 1.0)
        self.zod_recovery_spin.setSingleStep(0.01)
        self.zod_recovery_spin.setDecimals(3)
        self.zod_recovery_spin.setValue(ANALYSIS_DEFAULTS['zones_of_disagreement']['theta_recovery'])
        self.zod_gap_spin = QSpinBox()
        self.zod_gap_spin.setRange(0, self.videos.frame_count - 1)
        self.zod_gap_spin.setValue(ANALYSIS_DEFAULTS['zones_of_disagreement']['theta_gap'])
        zod_rows = (
            ("Mask-count seed", QLabel("missing or F/R mismatch")),
            ("Seed measure", self.zod_quality_measure_combo),
            ("Poor below θQ", self.zod_quality_spin),
            ("Expand to baseline", self.zod_expand_checkbox),
            ("Recovery measure", self.zod_recovery_measure_combo),
            ("Recovered at θrecovery", self.zod_recovery_spin),
            ("Merge gap θgap", self.zod_gap_spin),
        )
        for row, (name, editor) in enumerate(zod_rows):
            self.zod_table.setItem(row, 0, QTableWidgetItem(name))
            self.zod_table.setCellWidget(row, 1, editor)
        self.zod_status = QLabel()
        self.zod_status.setWordWrap(True)
        self.zod_apply_button = QPushButton("Apply ZoD parameters (Enter)")
        self.zod_apply_button.clicked.connect(self._update_zod_detection)
        zod_layout.addWidget(self.zod_table)
        zod_layout.addWidget(self.zod_apply_button)
        zod_layout.addWidget(self.zod_status)
        self.zod_panel.hide()
        self.zod_checkbox.toggled.connect(self._toggle_zod_panel)
        for spin in (
            self.zod_quality_spin, self.zod_recovery_spin, self.zod_gap_spin
        ):
            spin.valueChanged.connect(self._mark_zod_parameters_pending)
        self.zod_quality_measure_combo.currentIndexChanged.connect(
            self._zod_measure_selection_changed
        )
        self.zod_recovery_measure_combo.currentIndexChanged.connect(
            self._zod_measure_selection_changed
        )
        self.zod_expand_checkbox.toggled.connect(self._mark_zod_parameters_pending)
        self.zod_enter_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Return), self.zod_panel)
        self.zod_enter_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.zod_enter_shortcut.activated.connect(self._update_zod_detection)
        self.zod_keypad_enter_shortcut = QShortcut(
            QKeySequence(Qt.Key.Key_Enter), self.zod_panel
        )
        self.zod_keypad_enter_shortcut.setContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self.zod_keypad_enter_shortcut.activated.connect(self._update_zod_detection)
        sidebar.addWidget(self.zod_panel)
        sidebar_widget = QWidget()
        sidebar_widget.setLayout(sidebar)

        panels = QGridLayout()
        video_controls = QHBoxLayout()
        self.player = FramePlayer(
            "zone", "#ff3b4f", selectable=False,
        )
        self.player.configure(
            self.videos.frame_count, self.videos.source_fps or 30.,
        )
        self.player.loop.setText("Loop zone")
        self.player.frameRequested.connect(
            lambda frame, _playback: self.show_frame(frame)
        )
        video_controls.addWidget(self.player)
        video_controls.addStretch()
        self.mask_display = MaskDisplayControl(keypoints=self.keypoints is not None,
                                               global_colors=self.global_links is not None)
        self.mask_display.valueChanged.connect(
            lambda: self.show_frame(self.current_frame)
        )
        video_controls.addWidget(self.mask_display)
        reset_video_zoom = QPushButton("Reset zoom")
        reset_video_zoom.clicked.connect(lambda: self.navigation.reset())
        video_controls.addWidget(reset_video_zoom)
        panels.addLayout(video_controls, 0, 0, 1, 2)
        self.forward_label = self._image_label()
        self.reverse_label = self._image_label()
        self.navigation = VideoViewGroup(
            (self.forward_label, self.reverse_label),
            lock_aspect=True,
            minimum_span=.01,
        )
        panels.addWidget(self.forward_label, 1, 0)
        panels.addWidget(self.reverse_label, 1, 1)
        panels.setColumnStretch(0, 1)
        panels.setColumnStretch(1, 1)
        panels.setRowStretch(1, 1)
        start, end = ((self.zones[0].frames[0], self.zones[0].frames[-1]) if self.zones else
                      (0, self.videos.frame_count - 1))
        self.player.set_loop(*padded_loop(start, end, self.videos.frame_count))
        panels_widget = QWidget()
        panels_widget.setLayout(panels)

        lower_widget = QWidget()
        lower_layout = QVBoxLayout(lower_widget)
        lower_layout.setContentsMargins(0, 0, 0, 0)

        self.timeline_overview = TimelineOverview(self.videos.frame_count)
        self.timeline = GraphTimeline(self.videos.frame_count, "features")
        self.quality_timeline = GraphTimeline(self.videos.frame_count, "Quality scores")
        self.zone_timeline = ZoneTimeline(self.videos.frame_count)
        self.displacement_timeline = None
        if self.keypoints is not None and self.keypoints.displacement is not None:
            self.displacement_timeline = GraphTimeline(self.videos.frame_count, 'Keypoint displacement')
            configure_displacement_plot(self.displacement_timeline, self.keypoints.displacement)
        self._refresh_zone_plots()
        self.timeline.set_left_margin(self.zone_timeline.left_margin)
        self.metric_labels = self.measure_labels or {
            "missing_masks": "Missing",
            "mean_mask_area_px": "Area",
            "mean_centroid_displacement_px": "Displacement",
            "mean_temporal_iou": "IoU",
            "mean_relative_area_change": "Area change",
            "mean_fragmentation_index": "Fragmentation",
            "mean_hole_count": "Holes",
        }
        self.feature_plots = FeaturePlotController(
            self.timeline, self.frame_analysis, self.dense_frame_analysis, self.metric_labels,
        )
        self.quality_plot_labels = configure_quality_plot(
            self.quality_timeline,
            self.quality_metric_arrays,
            self.quality_plot_labels,
            left_axis_width=self.zone_timeline.left_margin, colors=self.metric_colors,
        )
        self.has_quality_data = bool(self.quality_plot_labels)
        self.has_feature_data = bool(self.frame_analysis or self.dense_frame_analysis)
        self.detail_timelines: tuple[TimelineBase, ...] = (
            self.quality_timeline, self.timeline, self.zone_timeline
        )
        if self.displacement_timeline is not None:
            self.detail_timelines += (self.displacement_timeline,)
        aligned_timelines = [self.zone_timeline]
        if self.displacement_timeline is not None:
            aligned_timelines.append(self.displacement_timeline)
        if self.has_quality_data:
            aligned_timelines.append(self.quality_timeline)
        if self.has_feature_data:
            aligned_timelines.append(self.timeline)
        self.timeline_group = TimelineGroup(self.timeline_overview, self.detail_timelines,
            aligned=aligned_timelines, content_heights={self.timeline_overview: 112, self.zone_timeline: 0})
        self.timeline_group.align_axes(current=True)
        self.quality_timeline.set_frame_axis_labels_visible(False)
        self.timeline.set_frame_axis_labels_visible(False)
        for source in self.detail_timelines:
            source.frameClicked.connect(self.show_frame)
        self.timeline_overview.frameClicked.connect(self.show_frame)
        self.player_cursors = PlayerCursorController((self.player,))
        self.player_cursors.set_plots(
            line=tuple(plot for plot in self.detail_timelines if plot is not self.zone_timeline),
            brackets=(self.timeline_overview, self.zone_timeline),
        )
        self.timeline_overview.setMinimumHeight(112)
        timeline_splitter = PanelSplitter(Qt.Orientation.Vertical)
        timeline_splitter.setChildrenCollapsible(False)
        timeline_splitter.setToolTip("Drag a divider to resize the timeline plots")

        self.metric_checkboxes: dict[str, QCheckBox] = {}
        feature_legend: QWidget | None = None
        if self.has_feature_data:
            metric_controls = QHBoxLayout()
            for key, label_text in self.metric_labels.items():
                checkbox = QCheckBox(label_text)
                checkbox.setStyleSheet(f"QCheckBox {{ color: {self.metric_colors[key]}; }}")
                checkbox.setChecked(True)
                checkbox.toggled.connect(self._update_timeline_metrics)
                metric_controls.addWidget(checkbox)
                self.metric_checkboxes[key] = checkbox
            self.metric_envelope_checkbox = QCheckBox("Min–max envelope")
            self.metric_envelope_checkbox.setChecked(True)
            self.metric_envelope_checkbox.toggled.connect(self.timeline.set_envelope_visible)
            metric_controls.addWidget(self.metric_envelope_checkbox)
            self.metric_difference_checkbox = QCheckBox("F−R difference")
            self.metric_difference_checkbox.setToolTip(
                (
                    "Checked: plot raw max-animal forward badness minus raw max-animal "
                    "reverse badness; positive means forward is worse. Values "
                    "are not normalized by the largest difference. At overview "
                    "zoom, one-pixel curves show the minimum and maximum F−R "
                    "within each screen-resolution bucket; close zoom shows "
                    "the exact curve. Unchecked: plot separate forward and reverse curves."
                    if self.measure_frame_analysis
                    else "Checked: plot forward minus reverse. At overview zoom, the "
                    "one-pixel curves show the minimum and maximum within each "
                    "screen-resolution bucket; close zoom shows the exact curve. "
                    "Unchecked: plot separate forward and reverse curves."
                )
            )
            self.metric_difference_checkbox.toggled.connect(
                self._set_feature_difference_mode
            )
            metric_controls.addWidget(self.metric_difference_checkbox)
            metric_controls.addWidget(QLabel("Difference scale"))
            self.difference_scale_combo = QComboBox()
            self.difference_scale_combo.addItem("95th percentile", "p95")
            self.difference_scale_combo.addItem("3σ", "3sigma")
            self.difference_scale_combo.addItem("6σ", "6sigma")
            self.difference_scale_combo.addItem("Raw F/R maximum", "raw_max")
            self.difference_scale_combo.setToolTip(
                "Symmetric vertical limit for each raw F−R curve. Values beyond "
                "a robust limit are shown at the top or bottom edge."
            )
            self.difference_scale_combo.currentIndexChanged.connect(
                lambda _index: self.feature_plots.set_scale(
                    str(self.difference_scale_combo.currentData())
                )
            )
            metric_controls.addWidget(self.difference_scale_combo)
            metric_controls.addStretch()
            feature_legend = LegendPanel(metric_controls)
            self._update_timeline_metrics()
            self.metric_difference_checkbox.setChecked(True)

        self.quality_checkboxes: dict[str, QCheckBox] = {}
        analysis_panels = {}
        if self.displacement_timeline is not None:
            analysis_panels['Keypoint displacement'] = GraphPanel(self.displacement_timeline)
        self.quality_plot_panel = GraphPanel(self.quality_timeline, checkable=True)
        analysis_panels['Quality'] = self.quality_plot_panel
        self.quality_checkboxes = self.quality_plot_panel.checkboxes
        self._update_quality_timeline_metrics()

        self.feature_plot_panel = GraphPanel(self.timeline, legend=feature_legend)
        analysis_panels['Features'] = self.feature_plot_panel
        self.zone_panel = ZonePanel(self.zone_timeline, checkable=True)
        self.zone_panel.rowVisibilityChanged.connect(self.timeline_overview.set_row_visible)
        self.zone_panel.rowVisibilityChanged.connect(lambda *_: self.refresh_font_geometry())
        timeline_panels = [self.timeline_overview, self.zone_panel]
        for index, panel in enumerate(timeline_panels):
            timeline_splitter.addWidget(panel)
            timeline_splitter.setStretchFactor(index, 1)
        timeline_splitter.setSizes([150, 600])
        self.timeline_splitter = timeline_splitter
        lower_layout.addWidget(timeline_splitter, 1)

        if self.zod_evidence is not None:
            self._update_zod_detection()

        panel_host = PanelHost()
        self.panel_docks = {}
        widgets = dict([self.controls]) if self.controls is not None else {}
        widgets.update({'Zone list': sidebar_widget, 'Video': panels_widget,
                        'Zones': lower_widget, **analysis_panels})
        for title, widget in widgets.items():
            dock = DockPanel(title, widget, panel_host, closable=True)
            panel_host.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, dock)
            self.panel_docks[title] = dock
        self.view_menu = add_view_menu(self, self.panel_docks)
        left_dock = self.panel_docks[self.controls[0] if self.controls is not None else 'Zone list']
        previous_dock = left_dock
        for title in ('Zones', *analysis_panels):
            dock = self.panel_docks[title]
            panel_host.splitDockWidget(previous_dock, dock, Qt.Orientation.Vertical)
            previous_dock = dock
        panel_host.splitDockWidget(left_dock, self.panel_docks['Video'], Qt.Orientation.Horizontal)
        if self.controls is not None:
            panel_host.splitDockWidget(left_dock, self.panel_docks['Zone list'], Qt.Orientation.Vertical)
        for title in analysis_panels:
            if ((title == 'Quality' and not self.has_quality_data) or
                    (title == 'Features' and not self.has_feature_data)):
                self.panel_docks[title].hide()
        panel_host.resizeDocks([left_dock, self.panel_docks['Video']], [320, 880], Qt.Orientation.Horizontal)
        panel_host.resizeDocks([self.panel_docks['Video'], self.panel_docks['Zones']], [650, 350], Qt.Orientation.Vertical)
        layout.addWidget(panel_host, 1)
        self.setCentralWidget(root)
        self._initial_panel_layout = True

    def showEvent(self, event):
        super().showEvent(event)
        if self._initial_video_zoom:
            self._initial_video_zoom = False
            original = self.videos.read('original', self.current_frame)
            masks = [mask for direction in ('forward', 'reverse')
                     if (mask := self._mask_frame(direction, self.current_frame)) is not None]
            self.navigation.fit_masks(masks, original.shape)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._initial_panel_layout:
            self._initial_panel_layout = False
            # First paint follows the native window's initial maximization.
            dock = self.panel_docks['Zones']
            video = self.panel_docks['Video']
            height = round(dock.height() * .6)
            video_height = video.height() + dock.height() - height
            QTimer.singleShot(0, self, lambda: dock.parentWidget().resizeDocks(
                [video, dock], [video_height, height], Qt.Orientation.Vertical,
            ))

    def refresh_font_geometry(self) -> None:
        """Realign timeline axes and row heights after a font-size change."""
        if hasattr(self, 'timeline_group'):
            self.timeline_group.refresh_geometry()

    def _update_timeline_metrics(self) -> None:
        if not hasattr(self, "metric_checkboxes"):
            return
        selected = [key for key, checkbox in self.metric_checkboxes.items() if checkbox.isChecked()]
        self.feature_plots.set_selected(selected)

    def _set_feature_difference_mode(self, enabled: bool) -> None:
        self.feature_plots.set_difference(enabled)
        self.metric_envelope_checkbox.setEnabled(not enabled)
        if hasattr(self, "difference_scale_combo"):
            self.difference_scale_combo.setEnabled(enabled)

    def _update_quality_timeline_metrics(self) -> None:
        if not hasattr(self, "quality_checkboxes"):
            return
        selected = [
            key for key, checkbox in self.quality_checkboxes.items()
            if checkbox.isChecked()
        ]
        self.quality_timeline.set_visible_series(selected)

    def _toggle_zod_panel(self, visible: bool) -> None:
        self.zod_panel.setVisible(visible)

    def _mark_zod_parameters_pending(self, *_args: object) -> None:
        if not hasattr(self, "zod_status") or self.zod_evidence is None:
            return
        self.zod_status.setText("Parameters changed — press Enter or Apply to recalculate.")

    def _selected_zod_metric(self, combo: QComboBox) -> str:
        key = combo.currentData()
        if not isinstance(key, str) or key not in self.quality_metric_arrays:
            raise ValueError("The selected ZoD measure is unavailable")
        return key

    def _configure_zod_threshold_range(
        self, combo: QComboBox, spin: QDoubleSpinBox
    ) -> None:
        if combo.count() == 0:
            return
        key = self._selected_zod_metric(combo)
        low, high = self.quality_metric_ranges.get(key, (-1.0, 1.0))
        values = np.asarray(self.quality_metric_arrays[key])
        finite = values[np.isfinite(values)]
        if finite.size and key not in self.quality_metric_ranges:
            low, high = float(np.min(finite)), float(np.max(finite))
        if high <= low:
            high = low + 1.0
        old_value = spin.value()
        spin.blockSignals(True)
        spin.setRange(low, high)
        spin.setValue(max(low, min(old_value, high)))
        spin.blockSignals(False)

    def _zod_measure_selection_changed(self, *_args: object) -> None:
        self._configure_zod_threshold_range(
            self.zod_quality_measure_combo, self.zod_quality_spin
        )
        self._configure_zod_threshold_range(
            self.zod_recovery_measure_combo, self.zod_recovery_spin
        )
        if hasattr(self, "quality_checkboxes"):
            for combo in (
                self.zod_quality_measure_combo, self.zod_recovery_measure_combo
            ):
                key = self._selected_zod_metric(combo)
                checkbox = self.quality_checkboxes.get(key)
                if checkbox is not None:
                    checkbox.setChecked(True)
            self._update_quality_timeline_metrics()
        quality_key = self._selected_zod_metric(self.zod_quality_measure_combo)
        recovery_key = self._selected_zod_metric(self.zod_recovery_measure_combo)
        quality_higher_is_better = self.quality_higher_is_better.get(
            quality_key, True
        )
        recovery_higher_is_better = self.quality_higher_is_better.get(
            recovery_key, True
        )
        self.zod_table.item(2, 0).setText(
            "Poor below θQ" if quality_higher_is_better else "Poor above θQ"
        )
        self.zod_table.item(5, 0).setText(
            "Recovered at/above θrecovery"
            if recovery_higher_is_better
            else "Recovered at/below θrecovery"
        )
        self._mark_zod_parameters_pending()

    def _update_zod_detection(self, *_args: object) -> None:
        if self.zod_evidence is None:
            return
        theta_quality = self.zod_quality_spin.value()
        theta_recovery = self.zod_recovery_spin.value()
        expansion = self.zod_expand_checkbox.isChecked()
        quality_key = self._selected_zod_metric(self.zod_quality_measure_combo)
        recovery_key = self._selected_zod_metric(self.zod_recovery_measure_combo)
        quality_higher_is_better = self.quality_higher_is_better.get(
            quality_key, True
        )
        recovery_higher_is_better = self.quality_higher_is_better.get(
            recovery_key, True
        )
        invalid_same_measure_recovery = (
            recovery_key == quality_key
            and quality_higher_is_better == recovery_higher_is_better
            and (
                (quality_higher_is_better and theta_recovery < theta_quality)
                or (not quality_higher_is_better and theta_recovery > theta_quality)
            )
        )
        if expansion and invalid_same_measure_recovery:
            self.zod_recovery_spin.blockSignals(True)
            self.zod_recovery_spin.setValue(theta_quality)
            self.zod_recovery_spin.blockSignals(False)
            theta_recovery = theta_quality
        self._set_zod_detection(
            theta_quality=theta_quality,
            expand_to_baseline=expansion,
            theta_recovery=theta_recovery,
            theta_gap=self.zod_gap_spin.value(),
            quality_signal=self.quality_metric_arrays[quality_key],
            recovery_signal=self.quality_metric_arrays[recovery_key],
            quality_higher_is_better=quality_higher_is_better,
            recovery_higher_is_better=recovery_higher_is_better,
        )
        self._refresh_zone_plots()
        for key in (quality_key, recovery_key):
            checkbox = self.quality_checkboxes.get(key)
            if checkbox is not None:
                checkbox.setChecked(True)
        self._update_quality_timeline_metrics()
        reference_lines: dict[str, list[tuple[str, float, str]]] = {
            quality_key: [
                (f"threshold = {theta_quality:g}", theta_quality, "#ffd43b")
            ]
        }
        if expansion:
            reference_lines.setdefault(recovery_key, []).append(
                (f"recovery = {theta_recovery:g}", theta_recovery, "#f4f4f4")
            )
        self.quality_timeline.set_reference_lines([
            ReferenceLine(label, value, color, series_key=key)
            for key, lines in reference_lines.items() for label, value, color in lines
        ])
        legend_lines = [(f"threshold = {theta_quality:g}", "#ffd43b")]
        if expansion:
            legend_lines.append((f"recovery = {theta_recovery:g}", "#f4f4f4"))
        self.quality_plot_panel.set_reference_lines(legend_lines)
        result = self.zod_result
        self.zod_status.setText(
            f"Seeds: {len(result.initial_zods):,} combined "
            f"({len(result.missing_seeds):,} mask-count, "
            f"{len(result.quality_seeds):,} quality, "
            f"{len(result.switch_seeds):,} switch) · "
            f"final ZoDs: {len(result.final_zods):,} · "
            f"seed: {self.quality_labels[quality_key]} · "
            f"recovery: {self.quality_labels[recovery_key]}"
        )

    @staticmethod
    def _image_label() -> ZoomableVideoLabel:
        return ZoomableVideoLabel()

    @staticmethod
    def _zone_table_values(
        index: int, zone: ZodView, minimum_quality: float | None,
        has_switch: bool,
    ) -> tuple[str, int, int, int, str, str, str]:
        label = (
            "Full video"
            if zone.name == "Full video (no hand labels)"
            else f"Zone {index + 1}"
        )
        missing = (
            "B" if zone.missing_in_forward and zone.missing_in_reverse
            else "F" if zone.missing_in_forward
            else "R" if zone.missing_in_reverse
            else "?" if zone.missing_direction_unknown
            else "∅"
        )
        switch = "✓" if has_switch else ""
        quality = "—" if minimum_quality is None else f"{minimum_quality:.3f}"
        return (
            label, zone.frames[0], zone.frames[-1], len(zone.frames),
            missing, switch, quality,
        )

    @property
    def zone(self) -> ZodView | None:
        index = self._zone_index_at_row(self.zone_table.currentRow())
        return self.zones[0 if index is None else index] if self.zones else None

    def _zone_index_at_row(self, row: int) -> int | None:
        item = self.zone_table.item(row, 0) if row >= 0 else None
        return None if item is None else int(item.data(ZONE_INDEX_ROLE))

    def _zone_table_selection_changed(self, row: int) -> None:
        index = self._zone_index_at_row(row)
        if index is None or index == self.selected_zone_index:
            return
        self.selected_zone_index = index
        self._select_zone(index)

    def _select_zone_table_row(self, index: int) -> None:
        for row in range(self.zone_table.rowCount()):
            if self._zone_index_at_row(row) == index:
                self.zone_table.setCurrentCell(row, 0)
                return

    def _select_zone(self, index: int) -> None:
        if index < 0:
            return
        self.playback_zone_pinned = False
        self.playback_zone_kind = "Zone"
        self.player.set_loop(*padded_loop(
            self.zones[index].frames[0], self.zones[index].frames[-1],
            self.videos.frame_count,
        ))
        self.player.loop.setChecked(True)
        for timeline in (self.zone_timeline, self.timeline_overview):
            timeline.set_selection(("Hand labels", "Hand labels", str(index)))
        self.show_frame(self.zones[index].frames[0])

    def _auto_zone_clicked(self, label: str, start: int, end: int) -> None:
        """Select an automatic interval as the active playback region."""
        if label != "Auto zones":
            return
        self.zone_table.blockSignals(True)
        self.zone_table.clearSelection()
        self.zone_table.setCurrentCell(-1, -1)
        self.zone_table.blockSignals(False)
        self.selected_zone_index = None
        for timeline in (self.zone_timeline, self.timeline_overview):
            row = next(row for row in timeline.rows if any(track.key == label for track in row.tracks))
            timeline.set_selection((row.key, label, f"{start}:{end}"))
        self.player.loop.setChecked(True)
        self.playback_zone_kind = "Auto zone"
        self.focus_interval(
            start, end, border_fraction=0.15, pin_playback_zone=True
        )


    def focus_zone(self, index: int, border_fraction: float = 0.15) -> None:
        """Select a zone and zoom every detailed timeline around its interval."""
        if not 0 <= index < len(self.zones):
            raise IndexError(f"Zone index is out of range: {index}")
        if not np.isfinite(border_fraction) or border_fraction < 0:
            raise ValueError("border_fraction must be a finite non-negative value")
        zone = self.zones[index]
        start, end = zone.frames[0], zone.frames[-1]
        border = (end - start + 1) * border_fraction
        view_start = max(0.0, start - border)
        view_end = min(float(self.videos.frame_count - 1), end + border)
        self._select_zone_table_row(index)
        self.playback_zone_pinned = False
        self.playback_zone_kind = "Zone"
        self.player.set_loop(*padded_loop(
            start, end, self.videos.frame_count,
        ))
        self.player.loop.setChecked(True)
        for timeline in self.detail_timelines:
            timeline.set_view(view_start, view_end)
        self.timeline_overview.set_view(view_start, view_end)

    def focus_interval(
        self, start: int, end: int, border_fraction: float = 0.15,
        *, pin_playback_zone: bool = False,
    ) -> None:
        """Zoom every detailed timeline around an arbitrary inclusive interval."""
        if not 0 <= start <= end < self.videos.frame_count:
            raise ValueError(f"Invalid frame interval: {start}-{end}")
        if not np.isfinite(border_fraction) or border_fraction < 0:
            raise ValueError("border_fraction must be a finite non-negative value")
        self.playback_zone_pinned = pin_playback_zone
        self.player.set_loop(*padded_loop(
            start, end, self.videos.frame_count,
        ))
        border = (end - start + 1) * border_fraction
        view_start = max(0.0, start - border)
        view_end = min(float(self.videos.frame_count - 1), end + border)
        for timeline in self.detail_timelines:
            timeline.set_view(view_start, view_end)
        self.timeline_overview.set_view(view_start, view_end)
        self.show_frame(start)


    def _timeline_clicked(self, zone_index: int, frame_number: int) -> None:
        self._select_zone_table_row(zone_index)
        zone = self.zones[zone_index]
        nearest = min(range(len(zone.frames)), key=lambda position: abs(zone.frames[position] - frame_number))
        self.show_frame(zone.frames[nearest])

    def show_frame(self, frame_number: int) -> None:
        if not 0 <= frame_number < self.videos.frame_count:
            return
        try:
            original = self.videos.read("original", frame_number)
            forward = self._mask_frame('forward', frame_number)
            reverse = self._mask_frame('reverse', frame_number)
        except ValueError as error:
            self.player.stop()
            QMessageBox.warning(self, "Video error", str(error))
            return
        for direction, masks, pane in (('forward', forward, self.forward_label),
                                       ('reverse', reverse, self.reverse_label)):
            colors = None
            if self.mask_display.color_mode is not None and self.mask_display.color_mode.currentData() == 'global':
                colors = bgr_colors(global_label_colors(
                    self.linking, self.global_links, direction, frame_number, self.link_seed))
            view = (self.mask_display.composite(original, masks, self.mask_palettes[direction], colors)
                    if masks is not None else original.copy() if self.mask_display.video_visible
                    else np.zeros_like(original))
            view = self.mask_display.composite_keypoints(view, self.keypoints, frame_number, direction, colors)
            self._set_image(direction, pane, view)
        self.current_frame = frame_number
        self.player.set_frame(frame_number)
        for timeline in self.detail_timelines:
            timeline.set_current_frame(frame_number)
        self.timeline_overview.set_current_frame(frame_number)

    def _mask_frame(self, direction, frame):
        """Leave unavailable or not-yet-segmented frames as the original video."""
        if frame >= self.videos.counts.get(direction, 0):
            return None
        masks = self.videos.masks.get(direction)
        if masks is not None and not masks.valid[frame]:
            return None
        return self.videos.read(direction, frame)

    def _set_image(self, key: str, label: ZoomableVideoLabel, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
        self._last_images[key] = image
        label.set_image(image)

    def move(self, offset: int) -> None:
        self.show_frame(max(0, min(self.current_frame + offset, self.videos.frame_count - 1)))

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_PageUp:
            self.move(-10)
        elif event.key() == Qt.Key.Key_PageDown:
            self.move(10)
        elif event.key() == Qt.Key.Key_Home:
            self.show_frame(self.player.loop_start if self.playback_zone_pinned else
                            self.zone.frames[0] if self.zone else 0)
        elif event.key() == Qt.Key.Key_End:
            self.show_frame(self.player.loop_end if self.playback_zone_pinned else
                            self.zone.frames[-1] if self.zone else self.videos.frame_count - 1)
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.player.stop()
        self.frame_keys.close()
        if self.linking is not None:
            self.linking.clear()
        self.videos.close()
        if hasattr(self, 'recording'):
            self.recording.close()
        for dock in self.panel_docks.values():
            dock.hide()
        event.accept()
