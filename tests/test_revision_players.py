"""Review ranges, independent playback and shared timeline markers."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from PySide6.QtCore import QEvent, QPointF, QRectF, QSettings, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox, QVBoxLayout, QWidget
import numpy as np

from bivosia.ui.playback import (
    link_loops, next_frame, padded_loop, upcoming_frames, zone_loop,
)
from bivosia.dataset.annotations.masks import RevisionSession
from bivosia.analysis.mask_linking import automatic_link, boundaries, critical_links, initial_draft, validate_draft
from bivosia.analysis.zones_of_disagreement import ZodAnnotation, critical_zones
from bivosia.ui.link_colors import LinkColors
from bivosia.ui.plot_models import IntervalSpec, IntervalTrack, RowSpec
from bivosia.ui.quality_plot import GraphTimeline
from bivosia.ui.revision_linking_editor import RevisionLinkingWindow
from bivosia.ui.review_video_viewer import ReviewVideoViewer
from bivosia.ui.font_zoom import FontZoomController
from bivosia.ui.player import FrameKeyNavigation, FramePlayer
from bivosia.ui.timeline_base import TimelineBase
from bivosia.ui.video_panels import VideoViewGroup, ZoomableVideoLabel
from bivosia.ui.zone_plot import ZoneTimeline, TimelineOverview


class PlaybackRangesTests(unittest.TestCase):
    def test_zone_and_background_ranges(self):
        zone = ZodAnnotation("z", 200, 250)
        entry, exit = boundaries([zone], 500)
        self.assertEqual(padded_loop(200, 250, 500), (150, 300))
        self.assertEqual(zone_loop(zone, 500), (150, 300))
        self.assertEqual(link_loops(entry, [zone], 500), {"left": (100, 199), "right": (200, 300)})
        self.assertEqual(link_loops(exit, [zone], 500), {"left": (150, 250), "right": (251, 350)})

    def test_adjacent_short_background_and_internal_links(self):
        zones = [ZodAnnotation("a", 5, 30), ZodAnnotation("b", 31, 60), ZodAnnotation("c", 70, 99)]
        jobs = boundaries(zones, 100, (20,))
        adjacent = next(j for j in jobs if j.left.end == 30)
        self.assertEqual(link_loops(adjacent, zones, 100), {"left": (0, 30), "right": (31, 99)})
        internal = next(j for j in jobs if j.id == "frame:20")
        self.assertEqual(link_loops(internal, zones, 100), {"left": (0, 20), "right": (21, 99)})
        short = next(j for j in jobs if j.left.end == 60)
        self.assertEqual(link_loops(short, zones, 100)["right"], (61, 99))

    def test_short_zone_loops_have_at_least_100_frames(self):
        zone = ZodAnnotation("z", 200, 205)
        entry, exit = boundaries([zone], 500)
        self.assertEqual(link_loops(entry, [zone], 500), {"left": (100, 199), "right": (200, 299)})
        self.assertEqual(link_loops(exit, [zone], 500), {"left": (106, 205), "right": (206, 305)})

    def test_background_loops_are_not_clipped_at_neighbor_zones(self):
        zones = [ZodAnnotation("a", 100, 160), ZodAnnotation("b", 170, 180)]
        jobs = boundaries(zones, 500)
        exit = next(j for j in jobs if j.left.end == 160)
        entry = next(j for j in jobs if j.right.start == 170)
        self.assertEqual(link_loops(exit, zones, 500)["right"], (161, 260))
        self.assertEqual(link_loops(entry, zones, 500)["left"], (70, 169))

    def test_adjacent_long_zones_pad_only_the_outer_edges(self):
        zones = [ZodAnnotation("a", 100, 300), ZodAnnotation("b", 301, 600)]
        job = next(j for j in boundaries(zones, 1000) if j.left.end == 300)
        self.assertEqual(link_loops(job, zones, 1000), {"left": (50, 300), "right": (301, 650)})

    def test_invalid_bridge_keeps_its_separate_inner_anchors(self):
        zones = [ZodAnnotation("a", 100, 150), ZodAnnotation("bad", 151, 170, "invalid"),
                 ZodAnnotation("b", 171, 180)]
        job = next(j for j in boundaries(zones, 500) if j.left.end == 150)
        self.assertEqual(link_loops(job, zones, 500), {"left": (50, 150), "right": (171, 270)})

    def test_inclusive_loop_and_nonloop_stop(self):
        self.assertEqual(list(upcoming_frames(10, 10, 12, True, 12)), [11, 12, 10])
        self.assertEqual(list(upcoming_frames(10, 10, 12, False, 12)), [11, 12])
        self.assertEqual(next_frame(50, 10, 12, True), 10)
        self.assertEqual(next_frame(12, 10, 12, True), 10)
        self.assertIsNone(next_frame(12, 10, 12, False))
        self.assertEqual(next_frame(10, 10, 10, True), 10)

    def test_crossed_anchors_are_valid(self):
        draft = dict(left_frame=80, right_frame=20, left_source="forward", right_source="reverse", mapping=[1, 0])
        validate_draft(draft, 2, 100)
        with self.assertRaises(ValueError):
            validate_draft({**draft, "left_frame": 100}, 2, 100)


class PlayerWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_critical_zone_navigation_skips_ordinary_automatic_choices(self):
        missing = np.zeros(80, dtype=int)
        missing[40] = missing[50] = 1
        zones = [ZodAnnotation('first', 0, 4, status='automatic'),
                 ZodAnnotation('ordinary', 10, 14, status='automatic'),
                 ZodAnnotation('invalid', 20, 24, source='invalid', status='automatic'),
                 ZodAnnotation('edited', 30, 34, status='needs_review'),
                 ZodAnnotation('incomplete', 40, 44, status='reviewed'),
                 ZodAnnotation('permitted', 50, 54, status='automatic'),
                 ZodAnnotation('last', 60, 64, status='reviewed')]
        session = RevisionSession({'mode': 'automatic'}, (), zones, missing, np.zeros(80, int), 2)
        window = RevisionLinkingWindow()
        window.session, window.zone_id = session, 'first'
        try:
            self.assertFalse(window.zone_critical_only.isChecked())
            before, revision = session._state(), session.revision
            critical = critical_zones(zones, session.missing_forward, session.missing_reverse, automatic=True)
            self.assertEqual([z.id for z in critical], ['invalid', 'edited', 'incomplete'])
            self.assertEqual(session._state(), before)
            self.assertEqual(session.revision, revision)
            with patch.object(window, 'select_zone', side_effect=lambda key: setattr(window, 'zone_id', key)), \
                 patch.object(window, '_edit', side_effect=lambda operation: session.edit(window.zone_id, operation)):
                window.next_zone()
                self.assertEqual(window.zone_id, 'ordinary')
                window.zone_critical_only.setChecked(True)
                window.next_zone()
                self.assertEqual(window.zone_id, 'invalid')
                window.approve_zone()
                self.assertEqual(window.zone_id, 'edited')
                self.assertEqual(session.zones[2].status, 'reviewed')
                window.approve_zone()
                self.assertEqual(window.zone_id, 'incomplete')
                window.next_zone()
                self.assertEqual(window.zone_id, 'incomplete')
                self.assertEqual(window.message.text(), 'No later critical zones.')
                window.zone_critical_only.setChecked(False)
                window.next_zone()
                self.assertEqual(window.zone_id, 'permitted')
        finally:
            window.session = None
            window.close()

    def test_critical_link_navigation_uses_saved_warnings_and_acknowledges_approval(self):
        missing = np.zeros(100, dtype=int)
        session = RevisionSession({'mode': 'automatic'}, (),
                                  [ZodAnnotation('invalid', 70, 79, source='invalid', status='automatic')],
                                  missing, missing, 2, link_frames=(9, 19, 29, 39, 49, 59))
        labels = np.array([[1, 0], [0, 2]], dtype=np.uint8)
        jobs = session.jobs
        session.links = {j.id: automatic_link(j, lambda source, frame: labels, 2) for j in jobs}
        session.links[jobs[1].id]['warnings'] = ['Hungarian proposal is ambiguous (tied assignments)']
        session.links[jobs[2].id]['quality'] = None
        session.set_draft(jobs[3].id, dict(initial_draft(jobs[3]), mapping=[0, 1], manual=True))
        session.links[jobs[4].id]['warnings'] = ['Some proposed pairs have zero pixel overlap']
        session.set_draft(jobs[4].id, dict(initial_draft(jobs[4]), mapping=[0, 1], manual=True))
        session.approve(jobs[4].id, (0, 1))
        session.links[jobs[5].id]['context'] = 'stale'
        window = RevisionLinkingWindow()
        window.session, window.job_id = session, jobs[0].id
        window.link_colors = LinkColors(None)
        try:
            self.assertFalse(window.link_critical_only.isChecked())
            before, revision = session._state(), session.revision
            with patch('bivosia.analysis.mask_linking.match_masks', side_effect=AssertionError('Recomputed masks')):
                links = window.link_colors.links(session)
                self.assertEqual([j.id for j in critical_links(links)], [jobs[i].id for i in (1, 2, 3, 5, 6)])
            self.assertEqual(session._state(), before)
            self.assertEqual(session.revision, revision)
            with patch.object(window, 'select_boundary', side_effect=lambda key: setattr(window, 'job_id', key)), \
                 patch.object(window, '_refresh'), patch.object(window, 'save'):
                window.next_boundary()
                self.assertEqual(window.job_id, jobs[3].id)
                window.job_id = jobs[0].id
                window.link_critical_only.setChecked(True)
                window.next_boundary()
                self.assertEqual(window.job_id, jobs[1].id)
                window.draft = dict(initial_draft(jobs[1]), mapping=[0, 1])
                window.data = SimpleNamespace(linking=SimpleNamespace(canonical_mapping=lambda *args: ((0, 1), [])))
                window.approve_link()
                self.assertEqual(session.links[jobs[1].id]['status'], 'approved')
                self.assertEqual(window.job_id, jobs[2].id)
                window.job_id = jobs[-1].id
                window.next_boundary()
                self.assertEqual(window.job_id, jobs[2].id)
        finally:
            window.session = window.data = None
            window.close()

    def test_arrow_keys_target_players_without_changing_loop_bounds(self):
        window = QWidget()
        layout = QVBoxLayout(window)
        players = {name: FramePlayer(name, 'red') for name in ('left', 'right', 'zone')}
        for player in players.values():
            layout.addWidget(player)
            player.configure(100, 50)
            player.set_frame(50)
            player.set_loop(40, 60)
        linking = [True]
        keys = FrameKeyNavigation(window, lambda side, offset: players[side].seek(
            players[side].frame + offset), lambda: linking[0])
        try:
            window.show()
            window.activateWindow()
            players['right'].spin.setFocus()
            self.app.processEvents()
            for key, side, expected in ((Qt.Key.Key_Left, 'left', 49),
                                        (Qt.Key.Key_Right, 'left', 50),
                                        (Qt.Key.Key_Up, 'right', 51),
                                        (Qt.Key.Key_Down, 'right', 50)):
                QTest.keyClick(players['right'].spin, key)
                self.assertEqual(players[side].frame, expected)
            linking[0] = False
            for key, expected in ((Qt.Key.Key_Left, 49), (Qt.Key.Key_Down, 48),
                                  (Qt.Key.Key_Up, 49), (Qt.Key.Key_Right, 50)):
                QTest.keyClick(players['left'].spin, key)
                self.assertEqual(players['zone'].frame, expected)
            self.assertEqual((players['left'].frame, players['right'].frame), (50, 50))
            self.assertTrue(all(player.bounds == (40, 60) for player in players.values()))
        finally:
            keys.close()
            window.close()

    def test_zone_move_and_resize_require_alt_when_enabled(self):
        for action, frame in (("move", 30), ("start", 19.5), ("end", 40.5)):
            for modifier in (Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.AltModifier):
                with self.subTest(action=action, modifier=modifier):
                    edits = []
                    row = RowSpec("zones", "Hand", (IntervalTrack("zones", "", (
                        IntervalSpec("z", 20, 40),)),), editable=True, draggable=True,
                        edit_requires_alt=True, on_edit=edits.append)
                    plot = ZoneTimeline(100, rows=(row,))
                    plot.resize(900, 180)
                    rect = plot._plot_rect()
                    y = next(plot.row_rects())[1].center().y()
                    start = QPointF(plot.frame_to_x(frame, rect, *plot.display_range()), y)
                    end = QPointF(plot.frame_to_x(frame + 5, rect, *plot.display_range()), y)
                    try:
                        QTest.mousePress(plot, Qt.MouseButton.LeftButton, modifier, start.toPoint())
                        self.app.sendEvent(plot, QMouseEvent(QEvent.Type.MouseMove, end, end,
                                                            Qt.MouseButton.NoButton,
                                                            Qt.MouseButton.LeftButton, modifier))
                        QTest.mouseRelease(plot, Qt.MouseButton.LeftButton, modifier, end.toPoint())
                        if modifier == Qt.KeyboardModifier.AltModifier:
                            self.assertEqual(len(edits), 1)
                            self.assertEqual(edits[0].action, action)
                            self.assertEqual(edits[0].frame, 45 if action == "end" else 25)
                        else:
                            self.assertFalse(edits)
                    finally:
                        plot.close()

    def test_font_zoom_reaches_legend_and_custom_painted_widgets(self):
        root = QWidget()
        legend = QCheckBox("Quality", root)
        legend.setStyleSheet("QCheckBox { color: red; }")
        plot = GraphTimeline(10)
        plot.setParent(root)
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        settings = QSettings(str(Path(temporary.name) / 'preferences.ini'), QSettings.Format.IniFormat)
        controller = FontZoomController(self.app, settings=settings)
        try:
            target = controller.default_point_size + 2
            controller.set_point_size(target)
            self.app.processEvents()
            self.assertEqual(legend.font().pointSizeF(), target)
            self.assertEqual(plot.font().pointSizeF(), target)
        finally:
            controller.reset()
            self.app.removeEventFilter(controller)
            root.close()

    def test_viewers_share_components_not_controls(self):
        zone = ReviewVideoViewer((("zone", "red"),))
        link = ReviewVideoViewer((("left", "green"), ("right", "orange")))
        self.assertEqual(len(zone.grid.panes), 2)
        self.assertEqual(len(link.grid.panes), 4)
        for player in (*zone.players.values(), *link.players.values()):
            player.configure(100, 50)
            self.assertTrue(player.loop.isChecked())
        link.players["left"].set_frame(80)
        link.players["right"].set_frame(20)
        link.players["left"]._step()
        self.assertEqual(link.players["left"].frame, 81)
        self.assertEqual(link.players["right"].frame, 20)
        self.assertEqual(zone.players["zone"].frame, 0)
        zone.mask_display.slider.setValue(20)
        zone.mask_display.video.setChecked(False)
        self.assertEqual(link.mask_display.slider.value(), 45)
        self.assertTrue(link.mask_display.video.isChecked())
        view = QRectF(.2, .2, .5, .5)
        link.grid.navigation.set_view(view)
        self.assertTrue(all(pane._view == view for pane in link.grid.panes.values()))
        self.assertTrue(all(pane._view == QRectF(0, 0, 1, 1) for pane in zone.grid.panes.values()))

    def test_shared_video_navigation_can_preserve_source_aspect(self):
        panes = (ZoomableVideoLabel(), ZoomableVideoLabel())
        navigation = VideoViewGroup(
            panes, lock_aspect=True, minimum_span=.01,
        )
        navigation.set_view(QRectF(.2, .3, .3, .5))
        self.assertEqual(navigation.view, QRectF(.1, .3, .5, .5))
        self.assertTrue(all(pane._view == navigation.view for pane in panes))

    def test_player_shows_loop_bounds_after_checkbox_and_dims_when_off(self):
        zone = ReviewVideoViewer((("zone", "red"),))
        link = ReviewVideoViewer((("left", "green"), ("right", "orange")))
        for player in (*zone.players.values(), *link.players.values()):
            player.configure(10000, 50)
            self.assertEqual(player.layout().indexOf(player.loop_bounds), player.layout().indexOf(player.loop) + 1)
            self.assertEqual(player.loop_bounds.text(), "0–9,999")
            player.set_loop(1200, 1300)
            self.assertEqual(player.loop_bounds.text(), "1,200–1,300")
            self.assertTrue(player.loop_bounds.isEnabled())
            player.loop.setChecked(False)
            self.assertFalse(player.loop_bounds.isEnabled())
            self.assertEqual(player.bounds, (0, 9999))
            self.assertEqual(player.loop_bounds.text(), "1,200–1,300")
            player.set_loop(1234, 1350)
            self.assertEqual(player.loop_bounds.text(), "1,234–1,350")
            self.assertFalse(player.loop_bounds.isEnabled())
            player.loop.setChecked(True)
            self.assertTrue(player.loop_bounds.isEnabled())
            self.assertEqual(player.bounds, (1234, 1350))
            player.set_loop(-10, 11000)
            self.assertEqual(player.loop_bounds.text(), "0–9,999")

    def test_loop_handles_share_one_row_and_prefer_active_cursor(self):
        zone = ReviewVideoViewer((("zone", "red"),))
        link = ReviewVideoViewer((("left", "green"), ("right", "orange")))
        players = {**zone.players, **link.players}
        plot = TimelineBase(100)
        plot.resize(900, 180)
        for player in players.values():
            player.configure(100, 50)
            player.set_loop(10, 50)
        plot.set_cursors([c for p in players.values() for c in p.cursors()])
        self.assertEqual([c.label for c in plot.cursors if not c.line], ["[", "]"] * 3)
        self.assertTrue(all(not c.dot for c in plot.cursors if not c.line))
        self.assertEqual({c.level for c in plot.cursors if not c.line}, {0})
        rect = plot._plot_rect()
        marker_y = rect.top() - 6
        for active in ("zone", "left", "right", "right:start", "zone:end"):
            plot.set_active_cursor(active)
            for frame, endpoint in ((10, "start"), (50, "end")):
                x = plot.frame_to_x(frame, rect, *plot.display_range())
                self.assertEqual(plot._cursor_at(QPointF(x, marker_y)),
                                 f"{active.split(':')[0]}:{endpoint}")
        players["right"].set_loop(50, 80)
        plot.set_cursors([c for p in players.values() for c in p.cursors()])
        x = plot.frame_to_x(50, rect, *plot.display_range())
        for active, endpoint in (("left", "end"), ("right", "start")):
            plot.set_active_cursor(active)
            self.assertEqual(plot._cursor_at(QPointF(x, marker_y)), f"{active}:{endpoint}")
        plot.set_active_cursor("left")
        x = plot.frame_to_x(80, rect, *plot.display_range())
        self.assertEqual(plot._cursor_at(QPointF(x, marker_y)), "right:end")

    def test_only_zone_plots_show_loop_handles(self):
        window = RevisionLinkingWindow()
        plots = (TimelineOverview(100), ZoneTimeline(100), GraphTimeline(100), GraphTimeline(100))
        window.plots = plots
        for player in window.players.values():
            player.configure(100, 50)
            player.set_loop(10, 50)
        window._select_cursor("left:start")
        for plot in plots[:2]:
            self.assertEqual(len(plot.cursors), 9)
            self.assertEqual(plot.active_cursor, "left:start")
        for plot in plots[2:]:
            self.assertEqual([c.key for c in plot.cursors], ["zone", "left", "right"])
            self.assertEqual(plot.active_cursor, "left")
            self.assertEqual(plot._plot_rect().top(), 14)
        with patch.object(window, "_anchor_changed") as move:
            window._plot_seek(20)
            move.assert_called_once_with("left", 20)
        window.close()

    def test_loop_handles_move_only_by_drag_and_plot_clicks_seek_player(self):
        window = RevisionLinkingWindow()
        window.data = SimpleNamespace(count=100)
        row = RowSpec("zones", "Hand", (IntervalTrack("zones", "", (IntervalSpec("z", 20, 60),)),),
                      editable=True, draggable=True)
        plots = (TimelineOverview(100, rows=(row,)), ZoneTimeline(100, rows=(row,)), GraphTimeline(100))
        window.plots = plots
        edits = []
        for plot in plots:
            plot.resize(900, 180)
            plot.cursorSelected.connect(window._plot_cursor_selected)
            plot.cursorMoved.connect(window._anchor_changed)
            plot.frameClicked.connect(window._plot_seek)
            if isinstance(plot, ZoneTimeline):
                plot.editRequested.connect(edits.append)
        for player in window.players.values():
            player.configure(100, 50)
            player.set_loop(10, 70)
        with patch.object(window, "_render_frames", side_effect=lambda *args: window._update_cursors()):
            try:
                for plot in plots[:2]:
                    for owner in window.players:
                        for endpoint, frame, moved in (("start", 10, 15), ("end", 70, 75)):
                            with self.subTest(plot=type(plot).__name__, owner=owner, endpoint=endpoint):
                                for player in window.players.values():
                                    player.set_frame(0)
                                    player.set_loop(10, 70)
                                window._select_cursor(owner)
                                rect = plot._plot_rect()
                                point = QPointF(plot.frame_to_x(frame, rect, *plot.display_range()), rect.top() - 6)
                                QTest.mouseClick(plot, Qt.MouseButton.LeftButton, pos=point.toPoint())
                                self.assertTrue(all(p.bounds == (10, 70) for p in window.players.values()))
                                for seek_plot in plots:
                                    seek_rect = seek_plot._plot_rect()
                                    x = seek_plot.frame_to_x(40, seek_rect, *seek_plot.display_range())
                                    window._select_cursor(f"{owner}:{endpoint}")
                                    QTest.mouseClick(seek_plot, Qt.MouseButton.LeftButton,
                                                     pos=QPointF(x, seek_rect.center().y()).toPoint())
                                    self.assertEqual(window.players[owner].frame, 40)
                                    self.assertTrue(all(p.bounds == (10, 70) for p in window.players.values()))
                                    self.assertFalse(edits)
                                QTest.mousePress(plot, Qt.MouseButton.LeftButton, pos=point.toPoint())
                                target = QPointF(plot.frame_to_x(moved, rect, *plot.display_range()), point.y())
                                self.app.sendEvent(plot, QMouseEvent(QEvent.Type.MouseMove, target, target,
                                                                    Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
                                                                    Qt.KeyboardModifier.NoModifier))
                                QTest.mouseRelease(plot, Qt.MouseButton.LeftButton, pos=target.toPoint())
                                expected = (15, 70) if endpoint == "start" else (10, 75)
                                self.assertEqual(window.players[owner].bounds, expected)
                                self.assertEqual(window.players[owner].frame, 40)
                                self.assertTrue(all(p.bounds == (10, 70) for name, p in window.players.items() if name != owner))
                                self.assertFalse(edits)
            finally:
                window.data = None
                window.close()

    def test_cursor_click_switches_mode_without_resetting_players_or_link(self):
        window = RevisionLinkingWindow()
        self.addCleanup(window.close)
        plot = ZoneTimeline(100)
        plot.resize(900, 180)
        window.plots = (plot,)
        plot.cursorSelected.connect(window._plot_cursor_selected)
        for frame, player in zip((10, 30, 70), window.players.values()):
            player.configure(100, 50)
            player.set_frame(frame)
            player.set_loop(frame - 5, frame + 5)
        draft = window.draft = {"mapping": [1, 0]}
        state = [(p.frame, p.bounds) for p in window.players.values()]
        window.show()
        self.app.processEvents()
        for key in ("right", "zone", "left", "zone:end", "right:start"):
            cursor = next(c for c in plot.cursors if c.key == key)
            rect = plot._plot_rect()
            x = plot.frame_to_x(cursor.frame, rect, *plot.display_range())
            y = rect.top() - 6 - 12 * cursor.level
            QTest.mouseClick(plot, Qt.MouseButton.LeftButton, pos=QPointF(x, y).toPoint())
            self.app.processEvents()
            title = "Zone" if key.startswith("zone") else "Link"
            self.assertEqual(window.active_viewer, title)
            self.assertEqual(window.active_cursor, key)
            self.assertEqual(window.annotation_mode, title)
            self.assertTrue(window.viewers[title].isVisible())
            self.assertEqual(len(plot.cursors), 9)
            self.assertEqual([(p.frame, p.bounds) for p in window.players.values()], state)
            self.assertIs(window.draft, draft)
        window._select_cursor("zone")
        self.assertEqual(window.annotation_mode, "Link")

    def test_panel_dividers_are_visible_without_plots_title(self):
        window = RevisionLinkingWindow()
        self.addCleanup(window.close)
        window.show()
        self.app.processEvents()
        self.assertEqual(window.centralWidget().x() - window.sidebar_dock.geometry().right() - 1, 5)
        plots = window.panel_docks["Plots"]
        self.assertEqual(plots.windowTitle(), "")
        self.assertEqual(plots.y() - window.panel_docks["Zone"].geometry().bottom() - 1, 5)
        plots.setFloating(True)
        self.assertTrue(plots.isFloating())
        plots.setFloating(False)
        self.assertFalse(plots.isFloating())

    def test_analysis_panels_start_hidden_and_reopen_with_synchronized_plots(self):
        window = RevisionLinkingWindow()
        self.addCleanup(window.close)
        values = np.arange(100, dtype=float)
        window.data = SimpleNamespace(count=100, n=2, quality=values / 100, assigned_iou=values / 100,
                                      measures={direction: {"missing_masks": values} for direction in ("FORWARD", "REVERSE")})
        window._build_plots()
        window.data = None
        splitter = window.plots_holder
        self.assertEqual(splitter.orientation(), Qt.Orientation.Vertical)
        self.assertEqual([splitter.widget(i) for i in range(splitter.count())],
                         [window.overview, window.zone_panel])
        self.assertIs(window.zone_panel.legend.parentWidget(), window.zone_plot.parentWidget())
        self.assertLess(window.zone_panel.layout().indexOf(window.zone_panel.legend),
                        window.zone_panel.layout().indexOf(window.zone_plot))
        self.assertGreaterEqual(
            window.quality_panel.layout().itemAt(0).layout().indexOf(
                window.quality_panel.legend
            ),
            0,
        )
        self.assertEqual(
            window.quality_panel.layout().indexOf(window.quality_plot), 1
        )
        window.show()
        self.app.processEvents()
        for title, panel in (("Quality", window.quality_panel), ("Features", window.feature_panel)):
            dock = window.panel_docks[title]
            action = dock.toggleViewAction()
            self.assertIs(dock.widget(), panel)
            self.assertTrue(dock.isHidden())
            self.assertFalse(action.isChecked())
            self.assertIn(action, window.view_menu.actions())
            action.trigger()
            self.app.processEvents()
            self.assertTrue(panel.graph.isVisible())
            dock.setFloating(True)
            self.assertTrue(dock.isFloating())
            dock.close()
            self.assertFalse(action.isChecked())
            action.trigger()
            self.app.processEvents()
            self.assertTrue(dock.isVisible())
            self.assertTrue(panel.graph.isVisible())
            dock.setFloating(False)
        for index, plot in enumerate((window.zone_plot, window.quality_plot, window.feature_plot)):
            plot.set_view(10 + index, 70 + index)
            plot.viewChanged.emit(plot.view_start, plot.view_end)
            self.assertEqual(window.zone_plot.display_range(), window.quality_plot.display_range())
            self.assertEqual(window.feature_plot.display_range(), window.quality_plot.display_range())
        for dock in (*window.annotation_docks.values(), window.processing_dock):
            dock.close()
            self.assertTrue(dock.isHidden())
            dock.toggleViewAction().trigger()
            self.app.processEvents()
            self.assertFalse(dock.isHidden())
        window.panel_docks["Quality"].close()
        window.data = SimpleNamespace(count=100, n=2, quality=values / 100, assigned_iou=values / 100,
                                      measures={}, measure_labels={})
        window._build_plots()
        window.data = None
        self.assertEqual(splitter.count(), 2)
        self.assertTrue(window.panel_docks["Quality"].isHidden())
        self.assertTrue(window.panel_docks["Features"].isVisible())
        self.assertFalse(window.feature_plot.series)
        self.assertIs(window.panel_docks["Features"].widget(), window.feature_panel)

    def test_zone_click_selects_and_double_click_focuses_from_timelines_and_list(self):
        window = RevisionLinkingWindow()
        missing = np.zeros(1000, bool)
        zones = [ZodAnnotation("a", 100, 180), ZodAnnotation("b", 500, 600)]
        window.data = SimpleNamespace(count=1000, n=2, quality=np.ones(1000), assigned_iou=np.ones(1000),
                                      missing=missing, missing_forward=missing, missing_reverse=missing,
                                      evidence=SimpleNamespace(switch_frames=missing), measures={}, measure_labels={})
        window.detected = SimpleNamespace(final_zods=((100, 180), (500, 600)), initial_zods=())
        window.session = RevisionSession({}, window.detected.final_zods,
                                         zones, missing, missing, 2)
        window.link_colors = SimpleNamespace(links=lambda session: SimpleNamespace(automatic={}))
        window.autosave.timeout.disconnect()
        with patch.object(window, "_render_frames"):
            try:
                window._build_plots()
                for spin in (window.start_frame, window.end_frame):
                    spin.setRange(0, 999)
                for player in window.players.values():
                    player.configure(1000, 50)
                window._refresh()
                window.show()
                self.app.processEvents()
                for plot, zone in ((window.overview, zones[1]), (window.zone_plot, zones[0])):
                    window._plot_cursor_selected("right")
                    window.players["zone"].set_frame(900)
                    plot.set_view(0, 999)
                    row = next(rect for spec, rect in plot.row_rects() if spec.key == "annotations")
                    x = plot.frame_to_x((zone.start + zone.end) / 2, plot._plot_rect(), *plot.display_range())
                    QTest.mouseClick(plot, Qt.MouseButton.LeftButton, pos=QPointF(x, row.center().y()).toPoint())
                    self.assertEqual(window.zone_id, zone.id)
                    self.assertEqual(window.zone_list.currentItem().data(Qt.ItemDataRole.UserRole), zone.id)
                    self.assertEqual((window.start_frame.value(), window.end_frame.value()), (zone.start, zone.end))
                    self.assertEqual(window.source.currentText(), zone.source)
                    self.assertEqual(window.annotation_mode, "Zone")
                    QTest.mouseDClick(plot, Qt.MouseButton.LeftButton, pos=QPointF(x, row.center().y()).toPoint())
                    self.assertEqual(window.zone_id, zone.id)
                    self.assertEqual(window.players["zone"].frame, zone.start)
                    self.assertEqual(window.players["zone"].bounds, zone_loop(zone, 1000))
                    self.assertEqual(window.annotation_mode, "Zone")
                    self.assertEqual(window.active_viewer, "Zone")
                    self.assertEqual(window.active_cursor, "zone")
                    self.assertEqual((window.start_frame.value(), window.end_frame.value()), (zone.start, zone.end))
                # The already selected list item must also return from the Link viewer.
                window.panel_docks["Link"].raise_()
                window._select_cursor("left")
                window.players["zone"].set_frame(900)
                item = window.zone_list.currentItem()
                point = window.zone_list.visualItemRect(item).center()
                QTest.mouseClick(window.zone_list.viewport(), Qt.MouseButton.LeftButton, pos=point)
                QTest.mouseDClick(window.zone_list.viewport(), Qt.MouseButton.LeftButton, pos=point)
                self.assertEqual(window.zone_id, "a")
                self.assertEqual(window.players["zone"].frame, 100)
                self.assertEqual(window.active_viewer, "Zone")
                self.assertTrue(window.viewers["Zone"].isVisible())
            finally:
                window.autosave.stop()
                window.session = window.data = None
                window.close()


if __name__ == "__main__":
    unittest.main()
