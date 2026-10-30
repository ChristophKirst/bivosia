"""View recordings before and after masks and analysis become available."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from copy import deepcopy
import time
import unittest

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
import cv2
import h5py
import numpy as np

from bivosia.analysis.zones_of_disagreement import detection_parameters
from bivosia.analysis.linking import automatic_link, initial_draft, prepare_links
from bivosia.dataset.layout import Dataset
from bivosia.dataset.jsonio import read_json
from bivosia.dataset.session import RevisionSession
from bivosia.processing.quantify import quantify
from bivosia.processing.zod_detection import detect_link_job
from bivosia.ui.recording_viewer import create_recording_viewer
from bivosia.ui.keypoint_alignment_window import KeypointAlignmentWindow
from bivosia.ui.revision_linking_editor import RevisionLinkingWindow
from bivosia.ui.window_manager import dataset_windows
from bivosia.ui.zone_video_viewer import ZoneVideoViewer


class RecordingViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.dataset = Dataset(temporary.name, video='test.avi')
        self.dataset.recording.mkdir()
        writer = cv2.VideoWriter(str(self.dataset.video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (32, 24))
        self.assertTrue(writer.isOpened())
        for frame in range(5):
            writer.write(np.full((24, 32, 3), 40 + frame * 10, dtype=np.uint8))
        writer.release()
        self.dataset.save()

    def masks(self, direction, frames=5, valid=None):
        self.dataset.masks.mkdir(exist_ok=True)
        values = np.zeros((frames, 24, 32), np.uint8)
        values[:, 2:8, 3:9] = 1
        values[:, 12:18, 20:27] = 2
        with h5py.File(self.dataset.mask(direction), 'w') as file:
            file['masks'] = values
            if valid is not None:
                file['valid'] = valid

    def viewer(self):
        viewer = create_recording_viewer(self.dataset)
        self.addCleanup(viewer.close)
        return viewer

    def finish_automatic(self, editor):
        deadline = time.monotonic() + 10
        while editor.automatic_task is not None and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertIsNone(editor.automatic_task)
        self.assertEqual(editor.monitor.job_status.text(), 'Complete')

    def test_video_then_partial_masks_keep_empty_plots_and_full_playback(self):
        viewer = self.viewer()
        self.assertFalse(viewer.zones)
        self.assertFalse(viewer.quality_timeline.series)
        self.assertFalse(viewer.timeline.series)
        self.assertTrue(all(not track.intervals for row in viewer.zone_timeline.rows for track in row.tracks))
        viewer.show_frame(4)
        self.assertEqual(viewer.current_frame, 4)
        viewer.close()
        self.masks('forward', frames=3, valid=[True, False, True])
        viewer = self.viewer()
        self.assertEqual(viewer.videos.frame_count, 5)
        self.assertIsNotNone(viewer._mask_frame('forward', 0))
        self.assertIsNone(viewer._mask_frame('forward', 1))
        self.assertIsNone(viewer._mask_frame('reverse', 0))
        viewer.show_frame(4)
        self.assertEqual(viewer.current_frame, 4)

    def test_saved_quality_and_zones_do_not_require_full_analysis(self):
        self.dataset.analysis.mkdir()
        np.save(self.dataset.array('mask_quality'), np.linspace(0, 1, 5))
        missing = np.zeros(5, dtype=int)
        session = RevisionSession.create(detection_parameters(.6, .9), [(1, 2)], missing, missing, 2,
                                         temporal_quality=np.ones((2, 4)))
        session.save(self.dataset.new_annotation(.6, .9))
        before = set(Path(self.dataset.directory).rglob('*'))
        viewer = self.viewer()
        self.assertIn('zod_local_quality', viewer.quality_timeline.series)
        self.assertIsNone(viewer.zod_evidence)
        self.assertEqual(len(viewer.zones), 1)
        self.assertEqual(viewer.zones[0].frames, (1, 2))
        self.assertEqual(set(self.dataset.directory.rglob('*')), before)

    def test_complete_analysis_still_loads_evidence_and_masks(self):
        self.masks('forward')
        self.masks('reverse')
        quantify(self.dataset, 2, workers=1)
        viewer = self.viewer()
        self.assertIsNotNone(viewer.zod_evidence)
        self.assertIn('mean_raw_assigned_iou', viewer.quality_timeline.series)
        self.assertIsNotNone(viewer.recording.linking)
        self.assertFalse(viewer.zones)
        viewer.show_frame(4)
        self.assertEqual(viewer.current_frame, 4)

    def test_alignment_window_runs_and_refreshes_the_existing_viewer(self):
        self.masks('forward')
        self.masks('reverse')
        quantify(self.dataset, 2, workers=1)
        annotation = detect_link_job(self.dataset)[0]
        self.dataset.select_annotation(annotation)
        self.dataset.settings['keypoints'] = {'format': 'sleap', 'image_shape': [24, 32]}
        self.dataset.settings['processing'] = {'align': {'n_animals': 2, 'workers': 1}}
        self.dataset.save()
        self.dataset.keypoints.mkdir()
        points = np.tile(np.array([[[[23., 15.]], [[5., 5.]]]]), (5, 1, 1, 1))
        with h5py.File(self.dataset.keypoint_file(), 'w') as file:
            file['tracks'] = points.transpose(1, 3, 2, 0)
            file['node_names'] = ['nose']
            file['track_names'] = ['a', 'b']

        window = KeypointAlignmentWindow(self.dataset, dataset_windows())
        self.addCleanup(window.close)
        window.show()
        deadline = time.monotonic() + 10
        while window.controls.task is not None and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertIsInstance(window.viewer, ZoneVideoViewer)
        self.assertFalse(window.viewer.isWindow())
        self.assertIsNone(window.viewer.keypoints.alignment)
        self.assertTrue(window.controls.run_button.isEnabled())
        self.assertFalse(self.dataset.array('keypoint_alignment').exists())
        window.viewer.show_frame(3)
        window.activateWindow()
        QTest.qWait(10)
        QTest.keyClick(window, Qt.Key.Key_Left)
        self.assertEqual(window.viewer.current_frame, 2)
        original_readers = window.viewer.videos
        window.controls.run_button.click()
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            QTest.qWait(20)
            if window.viewer.keypoints.alignment is not None and window.controls.task is None:
                break
        self.assertIsNotNone(window.viewer.keypoints.alignment, window.controls.processing.monitor.job_status.text())
        self.assertEqual(window.viewer.keypoints.alignment.assignment.tolist(), [[1, 0]] * 5)
        self.assertEqual(window.viewer.current_frame, 2)
        self.assertEqual(window.viewer.mask_display.keypoint_mode.currentData(), 'both')
        self.assertTrue(window.viewer.alignment_tracks)
        self.assertFalse(original_readers.readers)
        self.assertFalse(original_readers.masks)
        previous = window.viewer
        with patch('bivosia.ui.keypoint_alignment_window.create_recording_viewer',
                   side_effect=ValueError('Unreadable recording')):
            window.refresh_viewer()
        self.assertIs(window.viewer, previous)
        self.assertIs(previous.panel_docks['Keypoint alignment'].widget(), window.controls)
        self.assertEqual(window.controls.statusBar().currentMessage(), 'Unreadable recording')
        window.close()
        QTest.qWait(50)

    def test_automatic_saves_workflow_proposals_and_viewer_does_not_generate_links(self):
        self.masks('forward')
        self.masks('reverse')
        with h5py.File(self.dataset.mask('reverse'), 'r+') as file:
            masks = file['masks'][2]
            file['masks'][2] = np.where(masks > 0, 3 - masks, 0)
        with patch('bivosia.processing.zod_detection.create_automatic_annotation',
                   side_effect=AssertionError('Mask analysis must not generate links')):
            quantify(self.dataset, 2, workers=1)
        editor = RevisionLinkingWindow(self.dataset)
        self.addCleanup(editor.close)
        editor.run_automatic()
        self.finish_automatic(editor)
        session = editor.session
        self.assertTrue(session.jobs)
        self.assertEqual(session.header['mode'], 'automatic')
        self.assertTrue(all(zone.status == 'automatic' for zone in session.zones))
        self.assertTrue(all(link['status'] == 'automatic' and link['approved'] is None
                            for link in session.links.values()))
        self.assertEqual(editor.annotation_edit.text(), self.dataset.relative(session.path))
        saved = session.path.read_bytes()
        detected = read_json(detect_link_job(self.dataset)[0])
        self.assertEqual(detected['header'], session.header)
        self.assertEqual([(z['start'], z['end'], z['source']) for z in detected['zones']],
                         [(z.start, z.end, z.source) for z in session.zones])
        self.assertEqual([link['automatic'] for link in detected['links'].values()],
                         [link['automatic'] for link in session.links.values()])
        self.assertEqual(session.path.read_bytes(), saved)
        self.dataset.select_annotation(session.path)
        with patch('bivosia.analysis.linking.Linker.boundary_match', side_effect=AssertionError('Unexpected link calculation')):
            editor.load_data()
            viewer = self.viewer()
            self.assertEqual(len(viewer.global_links.automatic), len(session.jobs))
            viewer.close()
        self.assertEqual(session.path.read_bytes(), saved)
        zone = session.zones[0]
        session.edit(zone.id, 'bounds', start=zone.start + 1, end=zone.end)
        self.assertTrue(any(link['status'] == 'needs_review' for link in session.links.values()))
        session.save()
        saved = session.path.read_bytes()
        with patch('bivosia.analysis.linking.Linker.boundary_match', side_effect=AssertionError('Unexpected link calculation')):
            viewer = self.viewer()
            self.assertLess(len(viewer.global_links.automatic), len(session.jobs))
        self.assertEqual(session.path.read_bytes(), saved)

    def test_automatic_fill_preserves_manual_work_and_is_undoable(self):
        self.masks('forward')
        self.masks('reverse')
        quantify(self.dataset, 2, workers=1)
        editor = RevisionLinkingWindow(self.dataset)
        self.addCleanup(editor.close)
        session = RevisionSession.create(detection_parameters(.6, .9, direction_method='missing_mask_counts'), [(1, 1), (3, 3)],
                                         editor.data.missing_forward, editor.data.missing_reverse, 2,
                                         temporal_quality=editor.data.temporal_quality)
        approved, draft, automatic, unresolved = session.jobs
        for job in (approved, draft):
            session.set_draft(job.id, dict(initial_draft(job), mapping=[1, 0], manual=True))
        session.approve(approved.id, [1, 0])
        session.links[automatic.id] = automatic_link(automatic, editor.data.mask, 2)
        session.save(self.dataset.new_annotation(.6, .9))
        saved = session.path.read_bytes()
        with patch('bivosia.analysis.linking.match_masks', side_effect=AssertionError('Opening must not match masks')):
            editor._load_annotation(session.path)
        self.assertEqual(editor.session.zones[0].status, 'unseen')
        self.assertFalse(editor.session.dirty)
        self.assertFalse(editor.autosave.isActive())
        self.assertEqual(session.path.read_bytes(), saved)
        before = deepcopy(editor.session.links)
        with patch.object(editor, '_automatic_choice', return_value=None):
            editor.run_automatic()
        self.assertIsNone(editor.automatic_task)
        self.assertEqual(session.path.read_bytes(), saved)
        with patch.object(editor, '_automatic_choice', return_value='fill'), patch(
                'bivosia.processing.zod_detection.automatic_link', wraps=automatic_link) as calculate:
            editor.run_automatic()
            self.finish_automatic(editor)
        self.assertEqual(calculate.call_count, 1)
        self.assertEqual(calculate.call_args.args[0].id, unresolved.id)
        self.assertEqual(editor.session.zones, session.zones)
        self.assertEqual(editor.session.header, session.header)
        for job in (approved, draft, automatic):
            self.assertEqual(editor.session.links[job.id], before[job.id])
        self.assertIn(unresolved.id, prepare_links(editor.session).mappings)
        editor.session.undo()
        self.assertEqual(editor.session.links, before)
        editor.save()



if __name__ == '__main__':
    unittest.main()
