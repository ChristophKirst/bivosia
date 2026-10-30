"""Playback optimizations preserve mask pixels, colors, zoom and frame order."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from types import SimpleNamespace
from unittest.mock import Mock, patch
import gc
import threading
import unittest
import weakref

from PySide6.QtCore import QRectF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
import cv2
import numpy as np

from bivosia.analysis.linking import LinkResult
from bivosia.analysis.mask_features import mask_centers
from bivosia.dataset.video import VideoBuffer
from bivosia.ui.link_editor import MaskPane
from bivosia.ui.colors import identity_colors
from bivosia.ui.player import FramePlayer
from bivosia.ui.review_video_viewer import ReviewVideoViewer
from bivosia.ui.video_panels import MaskDisplayControl, image_from_bgr, overlay_mask


def reference_overlay(original, masks, opacity, outlines, colors, video_visible=True):
    labels = cv2.resize(masks, original.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    result = original.copy() if video_visible else np.zeros_like(original)
    for identity in np.unique(labels):
        if identity <= 0:
            continue
        mask = labels == identity
        color = np.full_like(original, colors[(int(identity) - 1) % len(colors)])
        blended = (
            cv2.addWeighted(original, 1 - opacity, color, opacity, 0)
            if video_visible else color
        )
        result[mask] = blended[mask]
        if outlines:
            result[mask & ~cv2.erode(mask.astype(np.uint8), None).astype(bool)] = 255
    return result


class RenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_overlay_matches_full_frame_reference(self):
        rng = np.random.default_rng(2)
        masks = np.zeros((9, 13), np.uint8)
        masks[:6, :5], masks[2:8, 5:10], masks[6:, 10:] = 1, 2, 7
        masks[1, 12] = 7  # Disconnected pixels and image-edge outlines.
        colors = np.asarray([(20, 180, 75), (140, 33, 220), (5, 60, 150)], np.uint8)
        for labels in (masks, np.zeros_like(masks), np.ones_like(masks), masks.astype(np.uint16)):
            for height, width in ((9, 13), (4, 7), (36, 52), (29, 43)):
                original = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
                for opacity in (0, .45, 1):
                    for outlines in (False, True):
                        for video_visible in (True, False):
                            with self.subTest(
                                shape=original.shape, opacity=opacity,
                                outlines=outlines, video_visible=video_visible,
                            ):
                                np.testing.assert_array_equal(
                                    overlay_mask(
                                        original, labels, opacity, outlines,
                                        None, colors,
                                        video_visible=video_visible,
                                    ),
                                    reference_overlay(
                                        original, labels, opacity, outlines,
                                        colors, video_visible,
                                    ),
                                )

    def test_mask_display_control_owns_opacity_video_and_outlines(self):
        control = MaskDisplayControl()
        original = np.full((4, 6, 3), 90, np.uint8)
        control.slider.setValue(30)
        self.assertEqual(control.opacity, .3)
        np.testing.assert_array_equal(control.background(original), original)
        control.video.setChecked(False)
        self.assertFalse(control.video_visible)
        np.testing.assert_array_equal(
            control.background(original), np.zeros_like(original),
        )
        masks = np.zeros(original.shape[:2], np.uint8)
        masks[1:3, 2:4] = 1
        colors = np.array([(20, 180, 75)], np.uint8)
        np.testing.assert_array_equal(control.composite(original, masks, None, colors)[1, 2], 255)
        changed = Mock()
        control.valueChanged.connect(changed)
        control.outline.setChecked(False)
        self.assertFalse(control.outlines_visible)
        changed.assert_called_once_with()
        np.testing.assert_array_equal(control.composite(original, masks, None, colors)[1, 2], colors[0])

    def test_centers_include_missing_labels_and_ignore_background(self):
        for labels in (np.zeros((7, 9), np.uint8), np.arange(63).reshape(7, 9) % 5):
            expected = tuple((float(xs.mean()), float(ys.mean())) if xs.size else None
                             for i in range(1, 4) for ys, xs in (np.nonzero(labels == i),))
            self.assertEqual(mask_centers(labels, 3), expected)

    def test_keypoint_display_hides_analysis_exclusions_by_default(self):
        control = MaskDisplayControl(keypoints=True)
        points = np.array([[[10., 10.], [30., 30.]]])
        data = Mock(node_names=['nose', 'tailEnd'], excluded_keypoints=('tailEnd',),
                    alignment=SimpleNamespace(assignment=np.array([[0]])),
                    source=np.array([0]), image_shape=(40, 40))
        data.__getitem__ = Mock(return_value=points)
        original = np.zeros((40, 40, 3), np.uint8)
        self.assertFalse(control.all_keypoints.isChecked())
        for mode in ('original', 'assigned', 'both'):
            control.keypoint_mode.setCurrentIndex(control.keypoint_mode.findData(mode))
            control.all_keypoints.setChecked(False)
            selected = control.composite_keypoints(original, data, 0, 'forward')
            self.assertTrue(selected[10, 10].any())
            self.assertFalse(selected[30, 30].any())
            control.all_keypoints.setChecked(True)
            all_points = control.composite_keypoints(original, data, 0, 'forward')
            self.assertTrue(all_points[30, 30].any())
        np.testing.assert_array_equal(points, [[[10., 10.], [30., 30.]]])
        control.keypoints.setChecked(False)
        self.assertFalse(control.all_keypoints.isEnabled())
        np.testing.assert_array_equal(
            control.composite_keypoints(original, data, 0, 'forward'), original)

    def test_image_retains_buffer_and_bgr_colors(self):
        frame = np.full((7, 9, 3), (11, 72, 130), dtype=np.uint8)
        ref = weakref.ref(frame)
        image = image_from_bgr(frame)
        del frame
        gc.collect()
        self.assertIsNotNone(ref())
        self.assertEqual(image.pixelColor(0, 0).getRgb(), (130, 72, 11, 255))
        del image
        gc.collect()
        self.assertIsNone(ref())

    def test_unchanged_pane_is_cached_but_visual_changes_redraw(self):
        pane = MaskPane("left", "forward")
        original = np.zeros((24, 32, 3), np.uint8)
        masks = np.ones((12, 16), np.uint8)
        colors = identity_colors(2)
        centers = mask_centers(masks, 2)
        with patch("bivosia.ui.link_editor.overlay_mask", wraps=overlay_mask) as overlay:
            def render(**changes):
                pane.set_frame(**dict(original=original, masks=masks, centers=centers,
                                      colors=colors, active=True, linking=False,
                                      opacity=.45, video_visible=True, outlines=True) | changes)
            render()
            render()
            render(active=False, linking=True)
            self.assertEqual(overlay.call_count, 1)
            pane.set_view(QRectF(.25, .25, .5, .5))
            self.assertFalse(pane.pixmap().isNull())
            render(opacity=.7)
            render(colors=identity_colors(2, .2))
            render(original=original.copy())
            render(masks=masks.copy())
            render(video_visible=False)
            render(outlines=False)
            self.assertEqual(overlay.call_count, 7)
            self.assertFalse(overlay.call_args.args[3])

    def test_playback_includes_last_frame_and_subtracts_render_time(self):
        player = FramePlayer("zone", "red")
        player.configure(100, 50)
        self.assertEqual(player.play.text(), "")
        self.assertFalse(player.play.icon().isNull())
        self.assertEqual(player.play.accessibleName(), "Play")
        player.set_loop(10, 12)
        cursors = player.cursors()
        self.assertEqual([cursor.key for cursor in cursors], [
            "zone", "zone:start", "zone:end",
        ])
        self.assertEqual([cursor.label for cursor in cursors], ["", "[", "]"])
        self.assertEqual([cursor.level for cursor in cursors], [0, 0, 0])
        player.set_frame(10)
        player.play.setChecked(True)
        player.timer.stop()
        frames = []
        player.frameRequested.connect(lambda frame, playback: frames.append(frame))
        for _ in range(4):
            with patch("bivosia.ui.player.perf_counter", side_effect=(100, 100.006)):
                player._step()
            self.assertEqual(player.timer.interval(), 14)
        self.assertEqual(player.play.accessibleName(), "Pause")
        player.stop()
        self.assertEqual(frames, [11, 12, 10, 11])

    def test_playback_starts_at_cursor_and_loops_only_when_started_inside(self):
        from bivosia.ui.playback import upcoming_frames

        player = FramePlayer("zone", "red")
        player.configure(100, 50)
        player.set_loop(10, 12)
        frames = []
        player.frameRequested.connect(lambda frame, playback: frames.append(frame))
        for start, enabled, expected in (
            (9, True, [10, 11, 12, 13]),
            (13, True, [14, 15, 16, 17]),
            (11, True, [12, 10, 11, 12]),
            (12, True, [10, 11, 12, 10]),
            (11, False, [12, 13, 14, 15]),
            (98, True, [99]),
            (99, True, []),
        ):
            with self.subTest(start=start, loop=enabled):
                player.seek(start)
                frames.clear()
                player.loop.setChecked(enabled)
                player.play.setChecked(True)
                player.timer.stop()
                self.assertEqual(player.frame, start)
                self.assertEqual(player.looping, enabled and 10 <= start <= 12)
                prefetched = list(upcoming_frames(player.frame, *player.bounds, player.looping, 4))
                self.assertEqual(prefetched, expected[:len(prefetched)])
                for _ in range(4):
                    if not player.play.isChecked():
                        break
                    player._step()
                    player.timer.stop()
                self.assertEqual(frames, expected)
                if start >= 98:
                    self.assertFalse(player.play.isChecked())
                self.assertEqual((player.loop_start, player.loop_end), (10, 12))
                self.assertEqual(player.loop.isChecked(), enabled)
                player.stop()
        player.seek(11)
        player.play.setChecked(True)
        player.set_frame(13)
        player._step()
        self.assertEqual(player.frame, 14)
        self.assertFalse(player.looping)
        player.stop()
        player.close()

    def test_global_colors_preserve_links_and_raw_label_dragging(self):
        viewer = ReviewVideoViewer((("left", "green"), ("right", "orange")))
        self.addCleanup(viewer.close)
        masks = np.tile(np.array([1] * 16 + [2] * 16, dtype=np.uint8), (24, 1))
        data = Mock(n=2)
        data.image.return_value = np.zeros((24, 32, 3), np.uint8)
        data.mask.return_value = masks
        data.centers.return_value = mask_centers(masks, 2)
        data.linking.conversion.return_value = ((0, 1), ())
        data.linking.n = 2
        sources = {"left": "forward", "right": "reverse"}
        partition = [SimpleNamespace(id="s", start=0, end=1, source="forward")]
        preview = LinkResult(tuple(partition), {"s": (1, 0)}, {"s": 0}, ((0, 1, 0),), (), (), {})
        viewer.grid.set_mapping((1, 0))
        viewer.resize(1000, 600)
        viewer.show()
        colors = identity_colors(2, .27)
        for mode in ("labels", "global", "link", "labels"):
            viewer.color_mode.setCurrentIndex(viewer.color_mode.findData(mode))
            viewer.render(data, sources, .6, preview, (1, 0), linking=True)
            self.app.processEvents()
            self.assertTrue(viewer.grid.linking)
            self.assertEqual(viewer.grid.mapping, (1, 0))
            for side, direction in sources.items():
                pane = viewer.grid.panes[side, direction]
                self.assertTrue(pane.active and pane.linking)
                reversed_colors = mode == "global" or (mode == "link" and side == "right")
                self.assertEqual(pane.colors, colors[::-1] if reversed_colors else colors)
                started, targets = [], []
                pane.dragStarted.connect(lambda s, label, values=started: values.append((s, label)))
                pane.dragTarget.connect(targets.append)
                QTest.mousePress(pane, Qt.MouseButton.LeftButton, pos=pane.center_point(0).toPoint())
                QTest.mouseRelease(pane, Qt.MouseButton.LeftButton, pos=pane.center_point(1).toPoint())
                self.assertEqual(started, [(side, 0)])
                self.assertEqual(targets, [1])
        viewer.render(data, sources, .6, linking=False)
        self.assertFalse(viewer.grid.linking)

    def test_global_disconnected_regions_have_different_palettes(self):
        viewer = ReviewVideoViewer((("left", "green"), ("right", "orange")))
        self.addCleanup(viewer.close)
        data = Mock(n=2)
        masks = np.array([[1, 2], [1, 2]], dtype=np.uint8)
        data.image.return_value = np.zeros((2, 2, 3), np.uint8)
        data.mask.return_value = masks
        data.centers.return_value = mask_centers(masks, 2)
        data.linking.conversion.return_value = ((0, 1), ())
        data.linking.n = 2
        partition = tuple(SimpleNamespace(id=str(t), start=t, end=t, source="forward") for t in range(2))
        preview = LinkResult(partition, {"0": (0, 1), "1": (0, 1)}, {"0": 0, "1": 1},
                             ((0, 0, 0), (1, 1, 1)), (), (), {})
        for side, player in viewer.players.items():
            player.configure(2, 50)
            player.set_frame(0 if side == "left" else 1)
        viewer.color_mode.setCurrentIndex(viewer.color_mode.findData("global"))
        viewer.render(data, {"left": "forward", "right": "forward"}, .6, preview=preview, linking=True)
        left = viewer.grid.panes["left", "forward"].colors
        right = viewer.grid.panes["right", "forward"].colors
        self.assertTrue(all(a != b for a, b in zip(left, right)))


class VideoBufferTests(unittest.TestCase):
    def test_reverse_steps_decode_forward_batches_and_reuse_bounded_cache(self):
        class Capture:
            position = 0

            def __init__(self):
                self.seeks, self.reads = [], []

            def read(self, frame):
                if frame != self.position:
                    self.seeks.append(frame)
                self.position = frame
                self.reads.append(frame)
                self.position += 1
                return np.full((2, 3, 3), frame, np.uint8)

        capture = Capture()
        buffer = VideoBuffer(capture)
        buffer.cache_bytes = 5 * 2 * 3 * 3
        try:
            buffer.read(20)
            for frame in range(19, 14, -1):
                self.assertTrue(np.all(buffer.read(frame) == frame))
            self.assertEqual(capture.seeks, [20, 15])
            self.assertEqual(capture.reads, [20, 15, 16, 17, 18, 19])
            self.assertLessEqual(buffer.cached_bytes, buffer.cache_bytes)
            buffer.prefetch(())
            reads = len(capture.reads)
            self.assertTrue(np.all(buffer.read(18) == 18))
            self.assertEqual(len(capture.reads), reads)
        finally:
            buffer.close()
        self.assertEqual(buffer.cached_bytes, 0)

    def test_bounded_buffer_reads_in_order_after_loop_and_seek(self):
        class Capture:
            position = 0

            def __init__(self):
                self.threads = set()

            def read(self, frame):
                self.threads.add(threading.get_ident())
                self.position = frame
                self.position += 1
                return np.full((2, 3, 3), frame, np.uint8)

        capture = Capture()
        buffer = VideoBuffer(capture)
        try:
            buffer.prefetch(range(30))
            self.assertEqual(len(buffer.pending), buffer.capacity)
            for frame in (0, 1, 2, 0, 1, 29, 30):
                self.assertTrue(np.all(buffer.read(frame) == frame))
                buffer.prefetch((frame + 1, frame + 2, 0, 1))
        finally:
            buffer.close()
        self.assertEqual(len(capture.threads), 1)
        self.assertNotIn(threading.get_ident(), capture.threads)
        self.assertEqual(buffer.pending, {})

    def test_decoder_errors_reach_the_caller(self):
        capture = Mock()
        capture.read.side_effect = ValueError('Could not read frame 0')
        buffer = VideoBuffer(capture)
        try:
            buffer.prefetch((0,))
            with self.assertRaisesRegex(ValueError, "Could not read frame 0"):
                buffer.read(0)
        finally:
            buffer.close()


if __name__ == "__main__":
    unittest.main()
