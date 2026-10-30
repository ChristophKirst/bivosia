"""Analysis track grouping, frame categories and native hand-row hit testing."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from types import SimpleNamespace
import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtCore import Qt, QPoint, QPointF, QEvent
from PySide6.QtGui import QWheelEvent, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QTableWidget, QWidget
import numpy as np

from bivosia.ui.annotations import Zone, annotate_zone_missing, revision_zone
from bivosia.analysis.alignment import align_scores
from bivosia.analysis.zones_of_disagreement import RevisionZone
from bivosia.ui.legend_panel import LegendPanel
from bivosia.ui.player import FramePlayer, PlayerCursorController
from bivosia.ui.plot_models import RowSpec, SeriesSpec
from bivosia.ui.plot_styles import (
    HAND_BASE_COLORS,
    HAND_SWAP_COLOR,
    HAND_SWAP_COLORS,
    HAND_UNKNOWN_COLOR,
    HAND_BACKGROUND_COLOR,
    SWITCH_COLOR,
    MISSING_COLORS,
    REGION_COLORS,
    STANDARD_QUALITY_PLOT_LABELS,
)
from bivosia.ui.quality_plot import CurveLevel, GraphTimeline, GraphPanel
from bivosia.ui.zone_video_viewer import (
    SORT_VALUE_ROLE, ZoneVideoViewer, _SortableTableItem,
)
from bivosia.ui.viewer_plot_config import (compose_zone_tracks, _track_lanes, viewer_zone_rows,
                                   configure_quality_plot, keypoint_alignment_tracks)
from bivosia.ui.zone_plot import ZonePanel, ZoneTimeline, TimelineOverview


class ZoneTimelineTests(unittest.TestCase):
    def test_zone_legend_checkboxes_control_rows_and_survive_refreshes(self):
        app = QApplication.instance() or QApplication([])
        rows = viewer_zone_rows({'Hand labels': [], 'Auto zones': [(1, 3)]},
                                [Zone('hand', range(2, 4))])
        rows.append(RowSpec('connections', 'Links', legend=(), visible=False))
        detail = ZoneTimeline(6, rows=rows)
        overview = TimelineOverview(6, rows=rows)
        panel = ZonePanel(detail, checkable=True)
        panel.rowVisibilityChanged.connect(overview.set_row_visible)
        panel.resize(1000, 300)
        panel.show()
        app.processEvents()
        self.assertEqual([box.text() for box in panel.checkboxes.values()], ['Hand', 'ZoD', 'Links'])
        self.assertNotIn('Keypoints', panel.checkboxes)
        for key, checkbox in panel.checkboxes.items():
            QTest.mouseClick(checkbox, Qt.MouseButton.LeftButton)
            expected = key == 'connections'
            for timeline in (detail, overview):
                self.assertEqual(next(row.visible for row in timeline.rows if row.key == key), expected)
            self.assertIs(panel.checkboxes[key], checkbox)
            self.assertTrue(checkbox.isVisible())
        for timeline in (detail, overview):
            timeline.set_rows(rows)
        self.assertEqual([row.visible for row in detail.rows], [False, False, True])
        self.assertEqual([row.visible for row in overview.rows], [False, False, True])
        self.assertEqual([box.isChecked() for box in panel.checkboxes.values()], [False, False, True])
        for key, checkbox in panel.checkboxes.items():
            self.assertEqual(panel._row_labels[key].text(), detail.row_legend_html(
                next(row for row in rows if row.key == key)))
            checkbox.setChecked(False)
        self.assertEqual(list(detail.row_rects()), [])
        panel.grab()
        panel.checkboxes['Hand labels'].setChecked(True)
        detail.set_rows(rows[:2])
        self.assertEqual(list(panel.checkboxes), ['Hand labels', 'ZoD'])
        self.assertTrue(panel.checkboxes['Hand labels'].isChecked())
        panel.close()
        overview.close()

    def test_alignment_tracks_show_linked_swaps_ties_and_unassigned_identities(self):
        scores = np.array([[[2, 0], [0, 2]], [[0, 2], [2, 0]], [[1, 1], [1, 1]],
                           [[0, 0], [0, 0]], [[2, 0], [0, 0]], [[0, 0], [0, 0]]], dtype=float)
        valid = np.array([True, True, True, True, True, False])
        points = SimpleNamespace(alignment=align_scores(scores, valid), scores=scores,
                                 source=np.array([0, 0, 0, 0, 0, 2]),
                                 mask_present=np.repeat(valid[:, None], 2, axis=1))
        links = SimpleNamespace(partition=(RevisionZone('a', 0, 5),), maps={'a': (0, 1)},
                                components={'a': 0}, errors=())
        tracks = keypoint_alignment_tracks(points, links,
            point_counts=np.array([[1, 1], [1, 1], [1, 1], [1, 1], [1, 0], [0, 0]]))
        self.assertEqual(tracks['Tie'], [])
        self.assertEqual(tracks['Swap'], [(0, 1)])
        self.assertEqual(tracks['Swap over invalid zone'], [])
        self.assertEqual(tracks['Partial resolved swap'], [(1, 4)])
        self.assertEqual(tracks['Partial swap over invalid zone'], [])
        self.assertEqual(tracks['No overlap'], [(3, 3)])
        self.assertEqual(tracks['Unassigned'], [])
        self.assertEqual(tracks['Tie + unassigned'], [(2, 2)])
        self.assertEqual(tracks['Missing skeleton'], [(4, 4)])
        rows = viewer_zone_rows(tracks)
        self.assertEqual(rows[-1].key, 'Keypoints')
        self.assertEqual([label for group in rows[-1].legend for label, _color in group.items],
                         ['Swap', 'Swap over invalid zone', 'Partial resolved swap', 'Swap over invalid zone',
                          'No overlap', 'Tie', 'Unassigned',
                          'Tie + unassigned', 'Missing skeleton'])
        np.testing.assert_allclose([track.intervals[0].bands[0].offset for track in rows[-1].tracks
                                    if track.intervals], [0., .2, .4, .6, .8])
        app = QApplication.instance() or QApplication([])
        widget = ZoneTimeline(6, rows=rows[-1:])
        widget.resize(600, 200)
        widget.show()
        app.processEvents()
        screenshot = widget.grab().toImage()
        _, rect = next(widget.row_rects())
        for track in rows[-1].tracks:
            if not track.intervals:
                continue
            interval = track.intervals[0]
            x = round(widget.frame_to_x((interval.start + interval.end) / 2,
                                       widget._plot_rect(), *widget.display_range()))
            band = track.intervals[0].bands[0]
            y = round(rect.top() + 1 + (rect.height() - 2) * (band.offset + band.height / 2))
            self.assertEqual(screenshot.pixelColor(x, y).name(), band.color)
        widget.close()

    def test_graph_panel_collapses_generated_and_supplied_legends(self):
        app = QApplication.instance() or QApplication([])
        for custom_legend in (False, True):
            graph = GraphTimeline(5, series=[SeriesSpec('value', 'Value', np.arange(5))])
            legend = LegendPanel.containing(QWidget()) if custom_legend else None
            panel = GraphPanel(graph, title='Features', checkable=True, legend=legend)
            panel.show()
            app.processEvents()
            QTest.mouseClick(panel.collapse_button, Qt.MouseButton.LeftButton)
            app.processEvents()
            self.assertTrue(graph.isHidden())
            self.assertTrue(panel.legend.isHidden())
            self.assertTrue(panel.collapse_button.isVisible())
            QTest.mouseClick(panel.collapse_button, Qt.MouseButton.LeftButton)
            app.processEvents()
            self.assertTrue(graph.isVisible())
            self.assertTrue(panel.legend.isVisible())
            if custom_legend:
                self.assertIs(panel.legend, legend)
            else:
                panel.checkboxes['value'].setChecked(False)
                self.assertNotIn('value', graph.visible_series)
            panel.close()

    def test_reviewed_hand_row_shows_directional_validity_and_swaps(self):
        zones = [revision_zone(RevisionZone(
            'hand-1', 2, 4, 'reverse', forward_valid=False, reverse_valid=True,
            forward_swap=1, reverse_swap=2)), revision_zone(RevisionZone(
            'hand-2', 6, 8, 'invalid', forward_valid=False, reverse_valid=None)),
            revision_zone(RevisionZone('hand-3', 10, 12, 'forward', forward_valid=True,
                                       reverse_valid=False, forward_swap=1)),
            revision_zone(RevisionZone('hand-4', 14, 16, 'reverse', forward_valid=True,
                                       reverse_valid=True, reverse_swap=1)),
            revision_zone(RevisionZone('hand-5', 18, 20, 'invalid', forward_valid=False,
                                       reverse_valid=False))]
        row = viewer_zone_rows({'Hand labels': []}, zones)[0]
        intervals = row.tracks[0].intervals
        self.assertEqual([[band.color for band in interval.bands] for interval in intervals], [
            [HAND_BASE_COLORS['UNKNOWN'], HAND_BASE_COLORS['REVERSE'],
             HAND_BASE_COLORS['REVERSE'], HAND_SWAP_COLORS['BOTH']],
            [HAND_BASE_COLORS['UNKNOWN'], HAND_UNKNOWN_COLOR, HAND_UNKNOWN_COLOR, HAND_BACKGROUND_COLOR],
            [HAND_BASE_COLORS['FORWARD'], HAND_BASE_COLORS['UNKNOWN'],
             HAND_BASE_COLORS['FORWARD'], HAND_SWAP_COLORS['FORWARD']],
            [HAND_BASE_COLORS['FORWARD'], HAND_BASE_COLORS['REVERSE'],
             HAND_BASE_COLORS['REVERSE'], HAND_SWAP_COLORS['REVERSE']],
            [HAND_BASE_COLORS['UNKNOWN'], HAND_BASE_COLORS['UNKNOWN'], HAND_UNKNOWN_COLOR, HAND_BACKGROUND_COLOR],
        ])
        for interval in intervals:
            self.assertEqual([(band.offset, band.height) for band in interval.bands],
                             [(0, 1/4), (1/4, 1/4), (2/4, 1/4), (3/4, 1/4)])
        self.assertEqual([interval.payload for interval in intervals], list(range(5)))
        self.assertIn('Swaps F: 1, R: 2', zones[0].details)
        self.assertIn('Swaps F: 0, R: 0', zones[1].details)
        self.assertEqual([group.label for group in row.legend],
                         ['valid', 'preferred', 'swap'])
        self.assertIsNone(row.legend_title)
        self.assertFalse(any(label in ('yes', 'no', 'unknown') for group in row.legend for label, _ in group.items))
        self.assertEqual([label for label, _ in row.legend[0].items], ['F', 'R', 'invalid', 'none'])
        self.assertEqual([label for label, _ in row.legend[1].items], ['F', 'R', 'none'])
        self.assertEqual([label for label, _ in row.legend[2].items], ['F', 'R', 'B'])
        self.assertEqual(row.legend[2].items,
                         (('F', HAND_SWAP_COLORS['FORWARD']), ('R', HAND_SWAP_COLORS['REVERSE']),
                          ('B', HAND_SWAP_COLORS['BOTH'])))

    def test_zone_table_items_sort_by_numeric_value(self):
        app = QApplication.instance() or QApplication([])
        table = QTableWidget(2, 1)
        for row, value in enumerate((10, 2)):
            item = _SortableTableItem(f"Zone {value}")
            item.setData(SORT_VALUE_ROLE, value)
            table.setItem(row, 0, item)
        table.setSortingEnabled(True)
        table.sortItems(0, Qt.SortOrder.AscendingOrder)
        self.assertEqual([table.item(row, 0).text() for row in range(2)], [
            "Zone 2", "Zone 10",
        ])

    def test_zone_table_uses_display_order_and_frame_columns(self):
        zone = Zone(
            "8c759f1147504a588793772628e301e8", tuple(range(20, 31)),
            missing_in_forward=True,
        )
        self.assertEqual(
            ZoneVideoViewer._zone_table_values(2, zone, 0.4138, True),
            ("Zone 3", 20, 30, 11, "F", "✓", "0.414"),
        )
        no_missing = Zone("id", (20, 21))
        unknown = Zone("id", (20, 21), missing_direction_unknown=True)
        self.assertEqual(
            ZoneVideoViewer._zone_table_values(0, no_missing, None, False)[4], "∅"
        )
        self.assertEqual(
            ZoneVideoViewer._zone_table_values(0, unknown, None, False)[4], "?"
        )
        historical_hand_swap = Zone("id", (20, 21), swap_in_forward=True)
        self.assertEqual(
            ZoneVideoViewer._zone_table_values(
                0, historical_hand_swap, None, False
            )[5],
            "",
        )

    def test_shared_player_cursor_controller_syncs_lines_and_loop_brackets(self):
        app = QApplication.instance() or QApplication([])
        player = FramePlayer("zone", "red", selectable=False)
        player.configure(20, 50)
        graph = GraphTimeline(20)
        zones = ZoneTimeline(20)
        controller = PlayerCursorController((player,))
        controller.set_plots(line=(graph,), brackets=(zones,))
        player.set_loop(3, 12)
        self.assertEqual([cursor.key for cursor in graph.cursors], ["zone"])
        self.assertEqual(
            [cursor.key for cursor in zones.cursors],
            ["zone", "zone:start", "zone:end"],
        )
        player.move_cursor("zone:start", 5)
        self.assertEqual((player.loop_start, player.loop_end), (5, 12))
        self.assertEqual(zones.cursors[1].frame, 5)
        graph.close()
        zones.close()
        player.close()

    def test_temporal_overview_retains_bin_extrema(self):
        values = np.array([-4.0, 2.0, 3.0, 5.0], dtype=np.float32)
        level = CurveLevel(1, values, None, None, values, values)
        reduced = GraphTimeline._reduce_curve_level(level)
        np.testing.assert_allclose(reduced.temporal_minimum, [-4.0, 3.0])
        np.testing.assert_allclose(reduced.temporal_maximum, [2.0, 5.0])

    def test_legend_panels_inherit_the_interface_palette(self):
        app = QApplication.instance() or QApplication([])
        panel = LegendPanel.containing(QWidget())
        self.assertEqual(panel.styleSheet(), "")
        self.assertEqual(panel.layout().contentsMargins().left(), 8)
        self.assertEqual(panel.layout().spacing(), 10)
        panel.close()

    def test_directional_and_event_colors_are_distinct(self):
        self.assertEqual(HAND_BASE_COLORS["FORWARD"], "#8bd17c")
        self.assertEqual(HAND_BASE_COLORS["REVERSE"], "#2f8f5b")
        self.assertTrue(
            set(HAND_BASE_COLORS.values()).isdisjoint(
                {HAND_SWAP_COLOR, SWITCH_COLOR, *HAND_SWAP_COLORS.values()}
            )
        )
        def luminance(color):
            red, green, blue = (
                int(color[index : index + 2], 16) for index in (1, 3, 5)
            )
            return 0.2126 * red + 0.7152 * green + 0.0722 * blue
        for colors in (HAND_BASE_COLORS, MISSING_COLORS, HAND_SWAP_COLORS):
            self.assertGreater(
                luminance(colors["FORWARD"]), luminance(colors["REVERSE"])
            )
        for colors in (MISSING_COLORS, HAND_SWAP_COLORS):
            self.assertGreater(
                luminance(colors["REVERSE"]), luminance(colors["BOTH"])
            )

    def test_double_click_reports_exact_frame_for_zone_and_quality(self):
        app=QApplication.instance() or QApplication([])
        for widget in (ZoneTimeline(100), GraphTimeline(100,'quality')):
            widget.resize(700,200); widget.set_view(20,40); widget.show(); app.processEvents()
            frames=[]; widget.frameDoubleClicked.connect(frames.append)
            plot=widget._plot_rect(); x=widget.frame_to_x(31,plot,20,40)
            QTest.mouseDClick(widget,Qt.MouseButton.LeftButton,pos=QPoint(round(x),round(plot.center().y())))
            self.assertEqual(frames,[31]); widget.close()

    def test_quality_y_fit_includes_zero_for_positive_zero_negative_and_missing_data(self):
        app=QApplication.instance() or QApplication([])
        for values in (np.array([.2,.4]),np.array([0.,.8]),np.array([-.35,.9]),np.full(2,np.nan)):
            graph=GraphTimeline(2,'quality')
            graph.set_series([SeriesSpec('q','q',values),SeriesSpec('old','old',np.array([.1,.8]))])
            graph.fit_shared_y_range(include=(0.,1.))
            low,high=graph.shared_y_range
            self.assertLess(low,0.); self.assertGreater(high,1.)
            if np.isfinite(values).any(): self.assertLess(low,float(np.nanmin(values)))
            self.assertEqual(graph.metric_ranges['q'],graph.metric_ranges['old'])
            graph.close()

    def test_shared_quality_configuration_and_metric_toggles_are_keyed(self):
        app=QApplication.instance() or QApplication([])
        graph=GraphTimeline(4,'quality')
        arrays={
            'zod_local_quality': np.array([.1,.2,.3,.4]),
            'mean_raw_assigned_iou': np.array([.8,.7,.6,.5]),
        }
        available=configure_quality_plot(
            graph,arrays,STANDARD_QUALITY_PLOT_LABELS,left_axis_width=120
        )
        self.assertEqual(list(available),[
            'zod_local_quality','mean_raw_assigned_iou'
        ])
        self.assertEqual({k:s.label for k,s in graph.series.items()},available)
        self.assertEqual([k for k in graph.series if k in graph.visible_series],list(available))
        self.assertEqual(graph.left_margin,120)
        self.assertGreater(graph.shared_y_range[1], 1.)
        self.assertIn(1., graph._quality_axis_tick_values())
        self.assertEqual(graph._format_axis_value(1.), "1.0")
        graph.set_series_visible('zod_local_quality',False)
        self.assertNotIn('zod_local_quality',graph.visible_series)
        self.assertIn('mean_raw_assigned_iou',graph.visible_series)
        graph.set_series_visible('zod_local_quality',True)
        self.assertEqual(graph.visible_series,set(available))
        graph.close()

    def test_ctrl_y_zoom_pan_reset_do_not_change_time_or_open_video(self):
        app=QApplication.instance() or QApplication([])
        graph=GraphTimeline(100,'quality'); graph.resize(800,250)
        graph.set_left_margin(120)
        graph.set_series([SeriesSpec('q','q',np.linspace(-.2,1,100)),SeriesSpec('old','old',np.linspace(0,1,100))])
        graph.fit_shared_y_range(include=(0.,1.))
        graph.set_view(10,70); graph.show(); app.processEvents()
        default=graph.shared_y_range; view=(graph.view_start,graph.view_end)
        point=graph._metric_rect().center(); ctrl=Qt.KeyboardModifier.ControlModifier
        frames=[]; graph.frameDoubleClicked.connect(frames.append)
        def wheel(modifiers):
            event=QWheelEvent(point,point,QPoint(0,0),QPoint(0,120),Qt.MouseButton.NoButton,
                              modifiers,Qt.ScrollPhase.NoScrollPhase,False)
            QApplication.sendEvent(graph,event)
        wheel(ctrl)
        zoomed=graph.shared_y_range
        self.assertLess(zoomed[1]-zoomed[0],default[1]-default[0])
        self.assertEqual((graph.view_start,graph.view_end),view)
        QTest.mousePress(graph,Qt.MouseButton.LeftButton,ctrl,point.toPoint())
        destination=point+QPointF(0,25)
        move=QMouseEvent(QEvent.Type.MouseMove,destination,destination,Qt.MouseButton.NoButton,
                         Qt.MouseButton.LeftButton,ctrl)
        QApplication.sendEvent(graph,move)
        QTest.mouseRelease(graph,Qt.MouseButton.LeftButton,ctrl,destination.toPoint())
        self.assertGreater(graph.shared_y_range[0],zoomed[0])
        self.assertAlmostEqual(graph.shared_y_range[1]-graph.shared_y_range[0],zoomed[1]-zoomed[0])
        self.assertEqual((graph.view_start,graph.view_end),view)
        self.assertEqual(graph.metric_ranges['q'],graph.metric_ranges['old'])
        QTest.mouseClick(graph,Qt.MouseButton.RightButton,ctrl,point.toPoint())
        self.assertEqual(graph.shared_y_range,default)
        self.assertEqual((graph.view_start,graph.view_end),view)
        QTest.mouseDClick(graph,Qt.MouseButton.LeftButton,ctrl,point.toPoint())
        self.assertEqual(frames,[])
        wheel(Qt.KeyboardModifier.NoModifier)
        self.assertLess(graph.view_end-graph.view_start,view[1]-view[0])
        self.assertEqual(graph.shared_y_range,default)
        graph.grab()  # Exercise fitted-axis painting as well as interaction.
        graph.close()

    def test_hand_row_click_follows_explicit_position(self):
        app = QApplication.instance() or QApplication([])
        hits=[]
        regions={'Hand labels':[],
            'Missing forward only':[], 'Missing both':[], 'Switch frames':[], 'Combined seeds':[], 'Auto zones':[]}
        rows=viewer_zone_rows(regions,[Zone('hand',tuple(range(4,9)))],
            callbacks={'Hand labels':lambda hit:hits.append((hit.interval.payload,hit.frame))})
        widget=ZoneTimeline(20,rows=rows)
        widget.resize(600,200)
        widget.show(); app.processEvents()
        plot=widget._plot_rect()
        x=widget.frame_to_x(6,plot,widget.view_start,widget.view_end)
        y=next(rect.center().y() for row, rect in widget.row_rects() if row.key == 'Hand labels')
        QTest.mouseClick(widget,Qt.MouseButton.LeftButton,pos=QPoint(round(x),round(y)))
        self.assertEqual(hits,[(0,6)])
        widget.close()

    def test_hand_row_labels_use_display_numbers_not_internal_names(self):
        regions = {'Hand labels': []}
        zones = [
            Zone('8c759f1147504a588793772628e301e8', tuple(range(4, 9))),
            Zone('internal-zone-name', tuple(range(12, 16))),
        ]
        row = viewer_zone_rows(regions, zones)[0]
        self.assertEqual([interval.label for interval in row.tracks[0].intervals], ['1', '2'])

    def test_shared_viewer_combines_zod_bands_in_detail_and_overview(self):
        m=np.array([0,1,1,0,0],bool)
        f=np.array([0,1,0,0,0],bool)
        r=np.array([0,0,1,0,0],bool)
        switches=np.array([0,0,1,0,0],bool)
        tracks=compose_zone_tracks(
            {},
            missing_signal=m, forward_missing_signal=f,
            reverse_missing_signal=r, switch_frames=switches,
            seed_intervals=[(1,2)], auto_intervals=[(1,3)],
        )
        self.assertEqual(_track_lanes(tracks), ['Hand labels', 'ZoD'])
        row = viewer_zone_rows(tracks)[1]
        self.assertEqual([group.label for group in row.legend], ['Auto', 'Fail: mask', 'Seed'])
        self.assertEqual([label for label, _ in row.legend[1].items], ['F', 'R', 'B', 'switch', 'mask+switch'])
        self.assertEqual(row.tracks[0].key, 'Auto zones')
        self.assertTrue(all(not interval.bands for interval in row.tracks[0].intervals))
        for track in row.tracks[1:]:
            offset = 2/3 if track.key == 'Combined seeds' else 1/3
            self.assertTrue(all((interval.bands[0].offset, interval.bands[0].height) == (offset, 1/3)
                                for interval in track.intervals))
        self.assertEqual(tracks['Missing forward only'],[(1,1)])
        self.assertEqual(tracks['Missing reverse only'],[(2,2)])
        self.assertEqual(tracks['Switch frames'],[(2,2)])
        self.assertEqual(tracks['Missing mask + switch'],[(2,2)])
        app = QApplication.instance() or QApplication([])
        hits = []
        fake = SimpleNamespace(_display_zone_tracks=lambda: tracks, zones=(), global_links=None,
                               annotation_label='Hand', extra_zone_rows=(),
                               _auto_zone_clicked=lambda *args: hits.append(args),
                               zone_timeline=ZoneTimeline(5), timeline_overview=TimelineOverview(5))
        ZoneVideoViewer._refresh_zone_plots(fake)
        self.assertEqual(fake.zone_timeline.rows[1], fake.timeline_overview.rows[1])
        for widget in (fake.zone_timeline, fake.timeline_overview):
            widget.resize(600, 240)
            widget.show()
            app.processEvents()
            screenshot = widget.grab().toImage()
            _, rect = list(widget.row_rects())[1]
            x = round(widget.frame_to_x(2, widget._plot_rect(), *widget.display_range()))
            for band, key in enumerate(('Auto zones', 'Missing mask + switch', 'Combined seeds')):
                y = round(rect.top() + 1 + (rect.height() - 2) * (band + .5) / 3)
                if widget is fake.zone_timeline:
                    self.assertEqual(screenshot.pixelColor(x, y).name(), REGION_COLORS[key])
            QTest.mouseClick(widget, Qt.MouseButton.LeftButton, pos=QPoint(x, y))
            self.assertEqual(hits[-1], ('Auto zones', 1, 3))
            widget.close()

    def test_overview_wheel_zooms_shared_detail_window(self):
        app=QApplication.instance() or QApplication([])
        overview=TimelineOverview(100)
        overview.resize(700,100); overview.set_view(20,80); overview.show(); app.processEvents()
        moved=[]; overview.windowMoved.connect(lambda a,b:moved.append((a,b)))
        point=overview._plot_rect().center()
        event=QWheelEvent(point,point,QPoint(0,0),QPoint(0,120),Qt.MouseButton.NoButton,
                          Qt.KeyboardModifier.NoModifier,Qt.ScrollPhase.NoScrollPhase,False)
        QApplication.sendEvent(overview,event)
        self.assertLess(overview.view_end-overview.view_start,60)
        self.assertEqual(moved[-1],(overview.view_start,overview.view_end))
        overview.close()

    def test_hand_zone_bands_keep_directional_missing_and_swap_flags(self):
        zone=Zone('Region 14',tuple(range(2,5)),preferred_pass='FORWARD',
                  swap_in_forward=True,swap_in_reverse=True)
        missing=np.array([0,0,1,1,0,0],bool)
        forward=np.array([0,0,1,0,0,0],bool)
        reverse=np.array([0,0,0,1,0,0],bool)
        annotated=annotate_zone_missing([zone],missing,forward,reverse)[0]
        self.assertTrue(annotated.missing_in_forward)
        self.assertTrue(annotated.missing_in_reverse)
        self.assertTrue(annotated.swap_in_forward)
        self.assertTrue(annotated.swap_in_reverse)


if __name__ == '__main__':
    unittest.main()
