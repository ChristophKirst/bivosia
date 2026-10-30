"""Focused automatic-workflow, linking and annotation-preservation checks."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import unittest

import h5py
import numpy as np

from bivosia.analysis.linking import automatic_link, boundaries, initial_draft, linked_frame, match_masks, prepare_links, validation_errors
from bivosia.analysis.zones_of_disagreement import RevisionZone, detection_parameters
from bivosia.dataset.jsonio import read_json
from bivosia.dataset.layout import Dataset
from bivosia.dataset.session import RevisionSession
from bivosia.dataset.shape_features import ShapeResults, shape_analysis_inputs
from bivosia.dataset.workflow import inspect_workflow, workflow_snapshot
from bivosia.processing.remote_inventory import inspect_locations
from bivosia.processing.workflow import required_products, step_availability, workflow_job


class WorkflowTests(unittest.TestCase):
    @staticmethod
    def saved_annotation(dataset):
        dataset.save()
        dataset.analysis.mkdir(exist_ok=True)
        np.save(dataset.array('mask_areas'), np.ones((2, 5, 2), np.uint32))
        np.save(dataset.array('mask_intersections'), np.ones((5, 2, 2), np.uint32))
        session = RevisionSession.create(detection_parameters(.6, .9, direction_method='missing_mask_counts'),
                                          [(1, 3)], np.zeros(5, int), np.zeros(5, int), 2)
        session.edit(session.zones[0].id, 'review')
        for job in session.jobs:
            session.set_draft(job.id, dict(initial_draft(job), mapping=[0, 1], manual=True))
            session.approve(job.id, (0, 1))
        session.save(dataset.new_annotation(.6, .9))
        dataset.select_annotation(session.path)
        return session

    def test_saved_annotation_status_is_independent_of_export_and_other_annotations(self):
        with TemporaryDirectory() as root:
            dataset = Dataset(root)
            session = self.saved_annotation(dataset)
            # An unreviewed, unselected annotation must not affect the selected one.
            other = RevisionSession.create(detection_parameters(.6, .9, direction_method='missing_mask_counts'),
                                            [(0, 1)], np.zeros(5, int), np.zeros(5, int), 2)
            other.save(dataset.new_annotation(.6, .9, prefix='other_'))
            dataset.select_annotation(session.path)
            original = session.path.read_bytes()
            with patch('bivosia.analysis.linking.match_masks', side_effect=AssertionError('Recomputed links')):
                info = inspect_workflow(dataset)
            self.assertTrue(info['annotation']['valid'], info['annotation'])
            self.assertIsNone(info['export'])
            self.assertEqual(session.path.read_bytes(), original)
            loaded = RevisionSession.load(session.path, np.zeros(5, int), np.zeros(5, int), 2)
            self.assertEqual(loaded._state(), session._state())
            job = session.jobs[0]
            session.set_draft(job.id, dict(initial_draft(job), mapping=[1, 0], manual=True))
            session.save()
            info = inspect_workflow(dataset)
            self.assertFalse(info['annotation']['valid'])
            self.assertTrue(info['annotation']['issues'])

    def test_feature_status_uses_selected_output_variants(self):
        with TemporaryDirectory() as root:
            dataset = Dataset(root)
            self.saved_annotation(dataset)
            dataset.masks.mkdir()
            for direction in ('forward', 'reverse'):
                with h5py.File(dataset.mask(direction), 'w') as file:
                    file['masks'] = np.zeros((5, 3, 4), np.uint8)
            settings = {'preprocess_masks': True, 'z_score': True, 'downsample_factor': 2}
            dataset.settings['processing'] = {'shape_features': settings}
            dataset.save()
            results = ShapeResults(dataset, dict(settings, radius=20, degree=4))
            features = dict(results.feature_expected, feature_id='test-descriptors')
            for kind, metadata in (('mask_zernike', features),
                                    ('mask_umap', results.embedding_expected(features))):
                path = results.path(kind)
                path.parent.mkdir(parents=True, exist_ok=True)
                with h5py.File(path, 'w') as file:
                    file['features'] = np.zeros((2, 3))
                    file.attrs['metadata'] = json.dumps(metadata)
            info = inspect_workflow(dataset)
            self.assertTrue(all(item['exists'] and not item['error'] for item in info['features'].values()))
            original = {name: results.path(name).read_bytes() for name in info['features']}
            for changes in ({'radius': 21, 'degree': 4}, {'radius': 20, 'degree': 6}):
                mismatched = dict(settings, **changes)
                info = inspect_workflow(dataset, feature_settings=mismatched)
                self.assertTrue(all(item['error'] for item in info['features'].values()))
                with self.assertRaises(ValueError):
                    shape_analysis_inputs(dataset, mismatched)
            self.assertEqual(original, {name: results.path(name).read_bytes() for name in original})
            with h5py.File(results.path('mask_umap'), 'r+') as file:
                file.attrs['metadata'] = json.dumps(dict(results.embedding_expected(features), feature_id='different'))
            info = inspect_workflow(dataset)
            self.assertFalse(info['features']['mask_zernike']['error'])
            self.assertTrue(info['features']['mask_umap']['error'])
            original_resolution = inspect_workflow(dataset, feature_settings=dict(settings, downsample_factor=1))
            self.assertTrue(all(not item['exists'] for item in original_resolution['features'].values()))
            with h5py.File(dataset.mask('forward'), 'r+') as file:
                file.attrs['changed'] = True
            info = inspect_workflow(dataset)
            self.assertTrue(all(item['exists'] and item['error'] for item in info['features'].values()))

    def test_workflow_refresh_applies_results_and_receives_save_and_job_notifications(self):
        from PySide6.QtWidgets import QApplication
        from bivosia.ui.window_manager import dataset_windows, notify_dataset_changed
        from bivosia.ui.workflow_window import WorkflowWindow
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as root, TemporaryDirectory() as other_root:
            dataset = Dataset(root)
            self.saved_annotation(dataset)
            with patch.object(WorkflowWindow, 'refresh', lambda self: None), \
                 patch('bivosia.ui.job_monitor.JobMonitor._poll', lambda self: None):
                window = WorkflowWindow(dataset, dataset_windows())
                window.monitor.timer.stop()
                app.processEvents()
            try:
                window.task = object()
                window.refresh_timer.start()
                window.refresh()
                self.assertTrue(window._refresh_pending)
                self.assertFalse(window.refresh_timer.isActive())
                window._ready(inspect_workflow(dataset))
                self.assertEqual(window.indicators['annotations'].property('workflow_state'), 'available')
                self.assertEqual(window.indicators['export'].property('workflow_state'), 'missing')
                with patch.object(window, 'refresh') as refresh:
                    window._finished()
                    refresh.assert_called_once()
                    self.assertFalse(window._refresh_pending)
                    refresh.reset_mock()
                    notify_dataset_changed(Dataset(other_root))
                    refresh.assert_not_called()
                    notify_dataset_changed(dataset)
                    refresh.assert_called_once()
                    refresh.reset_mock()
                    result = {'jobs': [('done', 'quantify')], 'result': {
                        'id': 'done', 'offset': 0, 'log': '',
                        'status': {'state': 'complete', 'total': 5, 'completed': 5}}}
                    for _ in range(2):
                        window.monitor._ready(result, window.monitor._generation)
                    refresh.assert_called_once()
                window._finished()
                self.assertTrue(window.refresh_timer.isSingleShot())
                self.assertTrue(window.refresh_timer.isActive())
            finally:
                window.task = None
                window.close()
                window.deleteLater()
                app.processEvents()

    def test_remote_counts_validate_local_annotation_and_exports_match_selection(self):
        with TemporaryDirectory() as local_root, TemporaryDirectory() as remote_root:
            local, remote = Dataset(local_root), Dataset(remote_root)
            session = self.saved_annotation(local)
            self.saved_annotation(remote)
            local.settings['remote_locations'] = [{'host': 'workstation', 'path': remote_root}]
            local.save()
            for name in ('mask_areas', 'mask_intersections'):
                local.array(name).unlink()
            remote.masks.mkdir()
            for direction in ('forward', 'reverse'):
                with h5py.File(remote.mask(direction), 'w') as file:
                    file['masks'] = np.zeros((5, 3, 4), np.uint8)
            for name in ('mask_iou_match', 'mask_quality', 'mask_missing', 'mask_switch'):
                np.save(remote.array(name), np.zeros(5))
            remote.file('mask_quality_metadata').write_text('{"n_animals": 2}')
            profile = {'host': 'workstation', 'python': 'python', 'code': '/software/bivosia'}
            supplied = []
            def inspect_remote(operation, **kwargs):
                self.assertEqual(operation, 'workflow')
                supplied.append(kwargs['annotation_record'])
                snapshot = workflow_snapshot(remote, **kwargs)
                # A different selected remote annotation has an export.
                self.assertNotEqual(snapshot['annotation']['fingerprint'], session.fingerprint())
                snapshot['export'] = {'exists': True, 'path': str(remote.masks / 'linked.h5'),
                                       'metadata': None, 'error': ''}
                snapshot['export_current'] = True
                snapshot['features'] = {name: {'exists': True, 'path': str(remote.array(name)),
                                              'metadata': None, 'error': ''}
                                        for name in ('mask_zernike', 'mask_umap')}
                return snapshot
            with patch('bivosia.processing.remote_inventory.Remote.call', side_effect=inspect_remote):
                info = inspect_locations(local, [profile], include_workflow=True)
            self.assertTrue(info['annotation']['valid'], info['annotation'])
            self.assertEqual(info['annotation']['path'], session.path)
            self.assertEqual(supplied, [read_json(session.path)])
            self.assertIsNone(info['export'])
            self.assertFalse(info['export_current'])
            self.assertTrue(all(not item['exists'] for item in info['features'].values()))
            self.assertEqual(info['analysis_remote'].host, 'workstation')

    def test_existing_inputs_and_excluded_dependencies(self):
        selected = {'quantify', 'detect_link', 'export', 'shape_features', 'align'}
        enabled = step_availability({'masks', 'keypoints'}, selected)
        self.assertTrue(enabled['vos'])
        self.assertTrue(all(not enabled[name] for name in selected))
        enabled = step_availability({'masks'}, {'export', 'shape_features'})
        self.assertEqual(enabled['export'], {'analysis', 'annotations'})
        self.assertIn('consensus_masks', enabled['shape_features'])
        ready = {'masks', 'analysis', 'annotations'}
        self.assertFalse(step_availability(ready, ())['shape_features'])
        self.assertEqual(step_availability(ready, (), settings={
            'shape_features': {'mask_source': 'export'}})['shape_features'], {'export'})
        self.assertFalse(step_availability({'export', 'annotations'}, ())['shape_features'])
        self.assertEqual(required_products([{'operation': 'quantify'}, {'operation': 'detect_link'},
                                            {'operation': 'shape_features'}]), {'masks'})
        self.assertEqual(required_products([{'operation': 'quantify'}, {'operation': 'detect_link'},
                                            {'operation': 'export'}]), {'masks'})

    def test_invalid_interval_matches_at_next_valid_frame(self):
        first = np.array([[1, 1, 0, 2, 2]], dtype=np.uint8)
        propagated = np.array([[2, 2, 0, 1, 1]], dtype=np.uint8)
        zone = RevisionZone('invalid', 1, 3, source='invalid', status='automatic')
        job = boundaries([zone], 5)[0]
        # Incoming forward labels have moved across the invalid interval.
        calls = []
        def masks(source, frame):
            calls.append((source, frame))
            return propagated if frame == 4 else first
        record = automatic_link(job, masks, 2)
        self.assertEqual(calls, [('forward', 4), ('forward', 4)])
        self.assertEqual(record['automatic']['canonical_mapping'], [0, 1])
        self.assertEqual(match_masks(first, propagated, 2).mapping, (1, 0))
        self.assertEqual(record['status'], 'automatic')
        self.assertIsNone(record['approved'])
        reverse_zone = RevisionZone('reverse', 4, 4, source='reverse', status='automatic')
        job = boundaries([zone, reverse_zone], 5)[0]
        record = automatic_link(job, lambda source, frame: propagated if source == 'forward' and frame == 4 else first, 2)
        self.assertEqual(record['automatic']['canonical_mapping'], [1, 0])

    def test_one_occluded_animal_is_allowed_only_in_automatic_export(self):
        masks = np.array([[0, 0, 0, 2, 2]], dtype=np.uint8)
        with self.assertRaises(ValueError):
            linked_frame(masks, masks, 'forward', (0, 1), 2)
        output, source, valid = linked_frame(masks, masks, 'forward', (0, 1), 2, allow_missing=True)
        self.assertEqual(source, 0)
        np.testing.assert_array_equal(output['preferred'], masks)
        self.assertTrue(valid[0])
        session = RevisionSession.create(detection_parameters(.6, .9, direction_method='missing_mask_counts'),
                                          [(0, 0)], np.zeros(1, int), np.zeros(1, int), 2)
        self.assertTrue(validation_errors(session, prepare_links(session, lambda job: None)))

    def test_pipeline_preserves_annotations_and_cancellation(self):
        with TemporaryDirectory() as root:
            dataset = Dataset(root)
            dataset.save()
            dataset.masks.mkdir()
            masks = np.zeros((5, 24, 32), np.uint8)
            masks[:, 2:8, 3:9] = 1
            masks[:, 12:18, 20:27] = 2
            for direction in ('forward', 'reverse'):
                with h5py.File(dataset.mask(direction), 'w') as file:
                    file['masks'] = masks
            manual = RevisionSession.create(detection_parameters(.6, .9), [], np.zeros(5, int),
                                             np.zeros(5, int), 2, temporal_quality=np.ones((2, 4)))
            previous = dataset.new_annotation(.6, .9, prefix='manual_')
            manual.save(previous)
            original = previous.read_bytes()
            steps = [{'operation': 'quantify', 'settings': {'n_animals': 2, 'workers': 1}},
                     {'operation': 'detect_link', 'settings': {}}, {'operation': 'export', 'settings': {}}]
            with self.assertRaises(InterruptedError):
                workflow_job(dataset, steps=steps, cancelled=lambda: True)
            self.assertFalse(dataset.array('mask_areas').exists())
            updates = []
            workflow_job(dataset, steps=steps, progress=updates.append)
            dataset.reload()
            record = read_json(dataset.annotation)
            self.assertNotEqual(dataset.annotation, previous)
            self.assertEqual(previous.read_bytes(), original)
            self.assertEqual(record['header']['mode'], 'automatic')
            with h5py.File(dataset.export_path(record['export']['path'])) as file:
                self.assertTrue(file['preferred_valid'][:].all())
                self.assertEqual(file.attrs['annotation_mode'], 'automatic')
            self.assertEqual({item['stage'] for item in updates}, {'Mask analysis', 'ZoD detection and linking', 'Export'})

    def test_local_analysis_remains_available_with_remote_masks(self):
        with TemporaryDirectory() as local_root, TemporaryDirectory() as remote_root:
            local, remote = Dataset(local_root), Dataset(remote_root)
            local.settings['remote_locations'] = [{'host': 'workstation', 'path': remote_root}]
            local.save()
            remote.save()
            remote.masks.mkdir()
            for direction in ('forward', 'reverse'):
                with h5py.File(remote.mask(direction), 'w') as file:
                    file['masks'] = np.zeros((2, 3, 4), np.uint8)
            local.analysis.mkdir()
            for name in ('mask_intersections', 'mask_areas', 'mask_iou_match', 'mask_quality',
                         'mask_missing', 'mask_switch'):
                np.save(local.array(name), np.zeros(2))
            local.file('mask_quality_metadata').write_text('{"n_animals": 2}')
            profile = {'host': 'workstation', 'python': 'python', 'code': '/software/bivosia'}
            for remote_analysis in (False, True):
                snapshot = workflow_snapshot(remote)
                if remote_analysis:
                    snapshot['analysis'] = workflow_snapshot(local)['analysis']
                with self.subTest(remote_analysis=remote_analysis), patch(
                        'bivosia.processing.remote_inventory.Remote.call', return_value=snapshot):
                    info = inspect_locations(local, [profile], include_workflow=True)
                self.assertEqual(info['mask_remote'].host, 'workstation')
                self.assertIsNone(info['analysis_remote'])
                self.assertTrue(all(item['exists'] and not item['error'] for item in info['analysis'].values()))
                self.assertTrue(all(item['path'].parent == local.analysis for item in info['analysis'].values()))
                self.assertFalse(any(item.get('host') for item in info['analysis'].values()))


if __name__ == '__main__':
    unittest.main()
