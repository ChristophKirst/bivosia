"""Check equivalent mask sources, validity, identities and feature extraction."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

import h5py
import numpy as np

from bivosia.analysis.mask_linking import initial_draft
from bivosia.analysis.mask_features import frame_zernike_features, missing_mask_counts, validate_shape_metadata
from bivosia.analysis.zones_of_disagreement import ZodAnnotation, detection_parameters
from bivosia.dataset.masks import ConsensusMaskReader, consensus_metadata
from bivosia.dataset.io.json import atomic_json, read_json
from bivosia.dataset.layout import Dataset
from bivosia.dataset.paths import auxiliary_path
from bivosia.dataset.workflow.processing import step_settings
from bivosia.dataset.annotations.masks import RevisionSession
from bivosia.dataset.mask_features import append_zernike, read_embedding, read_zernike, shape_analysis_inputs, zernike_path
from bivosia.processing.analysis.mask_export import export_masks
from bivosia.processing.storage.consolidation import plan_input_sync
from bivosia.processing.remote.server import dispatch
from bivosia.processing.analysis.mask_shape import extract_zernike, shape_features_job


class ConsensusMaskTests(unittest.TestCase):
    def dataset(self, root, suffix='h5'):
        dataset = Dataset(root)
        dataset.save()
        dataset.masks.mkdir()
        dataset.analysis.mkdir()
        base = np.zeros((8, 12), dtype=np.uint8)
        base[1:4, 1:4], base[4:7, 8:11] = 1, 2
        swapped = np.where(base > 0, 3 - base, 0)
        forward = np.tile(base, (8, 1, 1))
        forward[3:] = swapped
        reverse = np.tile(swapped, (8, 1, 1))
        forward[6, forward[6] == 2] = 0
        for direction, masks in (('forward', forward), ('reverse', reverse)):
            path = dataset.mask(direction).with_suffix('.' + suffix)
            if suffix == 'h5':
                with h5py.File(path, 'w') as file:
                    file['masks'] = masks
            elif suffix == 'npy':
                np.save(path, masks)
            else:
                np.savez_compressed(path, masks=masks)
        areas = np.array([[(mask == np.arange(1, 3)[:, None, None]).sum(axis=(1, 2))
                           for mask in masks] for masks in (forward, reverse)], dtype=np.uint32)
        intersections = np.array([[[np.count_nonzero((f == i) & (r == j)) for j in (1, 2)]
                                    for i in (1, 2)] for f, r in zip(forward, reverse)], dtype=np.uint32)
        np.save(dataset.array('mask_areas'), areas)
        np.save(dataset.array('mask_intersections'), intersections)
        for name in ('mask_iou_match', 'mask_quality', 'mask_missing', 'mask_switch'):
            np.save(dataset.array(name), np.zeros(8))
        atomic_json(dataset.file('mask_quality_metadata'), {'n_animals': 2})
        zones = [ZodAnnotation('reverse', 2, 3, source='reverse', status='reviewed'),
                 ZodAnnotation('invalid', 4, 4, source='invalid', status='reviewed'),
                 ZodAnnotation('occluded', 6, 6, source='forward', status='automatic')]
        session = RevisionSession(dict(detection_parameters(.6, .9), mode='automatic'), (), zones,
                                  *missing_mask_counts(areas), 2)
        for job in session.jobs:
            left_swapped = job.left.source == 'reverse' or job.left.end >= 3
            right_swapped = job.right.source == 'reverse' or job.right.start >= 3
            mapping = (0, 1) if left_swapped == right_swapped else (1, 0)
            session.set_draft(job.id, dict(initial_draft(job), mapping=list(mapping), manual=True))
            session.approve(job.id, mapping)
        session.save(dataset.new_annotation(.6, .9))
        dataset.select_annotation(session.path)
        expected = np.tile(base, (8, 1, 1)).astype(np.uint16)
        expected[4] = 0
        expected[6, expected[6] == 1] = 0
        return dataset, session, expected

    def export(self, dataset, session):
        result = export_masks(session, dataset, dataset.linked_masks(session.path))
        session.export = result
        session.save()
        return dataset.export_path(result['path'])

    def test_frame_equivalence_for_supported_formats_without_recalculating_links(self):
        for suffix in ('h5', 'npy', 'npz'):
            with self.subTest(suffix=suffix), TemporaryDirectory() as root:
                dataset, session, expected = self.dataset(root, suffix)
                original = {path: path.read_bytes() for path in
                            (dataset.mask('forward'), dataset.mask('reverse'), session.path)}
                with patch('bivosia.analysis.mask_linking.match_masks', side_effect=AssertionError('Recomputed links')):
                    with ConsensusMaskReader(dataset) as direct:
                        self.assertEqual(direct.metadata['mask_source'], 'annotation')
                        np.testing.assert_array_equal(direct[:], expected)
                        np.testing.assert_array_equal(direct[-1], expected[-1])
                        self.assertEqual(direct[2:2].shape, (0, 8, 12))
                        valid = direct.valid.copy()
                    with self.assertRaisesRegex(ValueError, 'closed'):
                        direct[0]
                self.assertEqual(valid.tolist(), [True, True, True, True, False, True, True, True])
                self.assertTrue(all(path.read_bytes() == contents for path, contents in original.items()))
                self.export(dataset, session)
                with ConsensusMaskReader(dataset) as automatic, ConsensusMaskReader(dataset, source='annotation') as direct:
                    self.assertEqual(automatic.metadata['mask_source'], 'export')
                    np.testing.assert_array_equal(automatic[:], direct[:])
                    np.testing.assert_array_equal(automatic.valid, direct.valid)

    def test_selection_stale_exports_and_unavailable_explicit_sources(self):
        with TemporaryDirectory() as root:
            dataset, session, expected = self.dataset(root)
            with self.assertRaises(FileNotFoundError):
                ConsensusMaskReader(dataset, source='export')
            path = self.export(dataset, session)
            forward = dataset.mask('forward')
            moved = forward.with_suffix('.saved')
            forward.rename(moved)
            try:
                with ConsensusMaskReader(dataset) as reader:
                    np.testing.assert_array_equal(reader[:], expected)
                with self.assertRaises(FileNotFoundError):
                    ConsensusMaskReader(dataset, source='annotation')
            finally:
                moved.rename(forward)
            with h5py.File(path, 'r+') as file:
                file.attrs['annotation_fingerprint'] = 'outdated'
            with ConsensusMaskReader(dataset) as reader:
                self.assertEqual(reader.metadata['mask_source'], 'annotation')
            with self.assertRaisesRegex(ValueError, 'do not match'):
                ConsensusMaskReader(dataset, source='export')
            before = consensus_metadata(dataset, source='annotation')
            session.edit('reverse', 'source', source='forward')
            session.save()
            after = consensus_metadata(dataset, source='annotation')
            with self.assertRaisesRegex(ValueError, 'different masks or settings'):
                validate_shape_metadata(before, after)
            with self.assertRaisesRegex(ValueError, 'Cannot read consensus masks'):
                ConsensusMaskReader(dataset, source='annotation')

    def test_sampled_features_keep_source_indices_and_full_results(self):
        with TemporaryDirectory() as root:
            dataset, _, _ = self.dataset(root)
            full_path = extract_zernike(dataset, radius=20, degree=2)
            original = full_path.read_bytes()
            full = read_zernike(full_path, {})
            sampling = dict(start_frame=0, end_frame=6, frame_step=2)
            subset_path = extract_zernike(dataset, radius=20, degree=2, **sampling)
            subset = read_zernike(subset_path, {'sampling': sampling})
            self.assertEqual(subset.frames.tolist(), [0, 0, 2, 2, 6])
            selected = (full.frames <= 6) & (full.frames % 2 == 0)
            for name in ('values', 'frames', 'identities'):
                np.testing.assert_array_equal(getattr(subset, name), getattr(full, name)[selected])
            self.assertEqual(full_path.read_bytes(), original)
            downsampled = extract_zernike(dataset, radius=20, degree=2, downsample_factor=2, **sampling)
            self.assertNotEqual(downsampled, subset_path)
            smaller = read_zernike(downsampled, {'downsample_factor': 2})
            self.assertEqual(smaller.metadata['image_shape'], [4, 6])
            self.assertEqual(smaller.metadata['source_image_shape'], [8, 12])
            np.testing.assert_array_equal(smaller.frames, subset.frames)
            np.testing.assert_array_equal(smaller.identities, subset.identities)
            with ConsensusMaskReader(dataset) as masks:
                for frame in np.unique(smaller.frames):
                    identities, values, excluded = frame_zernike_features(masks[frame], 20, 2, downsample_factor=2)
                    self.assertEqual(excluded, [])
                    np.testing.assert_array_equal(smaller.identities[smaller.frames == frame], identities)
                    np.testing.assert_allclose(smaller.values[smaller.frames == frame], values)
            self.assertEqual(full_path.read_bytes(), original)

    def test_embedding_round_trip_retains_samples_scaling_and_parameters(self):
        with TemporaryDirectory() as root:
            dataset, _, _ = self.dataset(root)
            with h5py.File(dataset.mask('forward'), 'r+') as file:
                file['masks'][0, 1, 1] = 0
            path = extract_zernike(dataset, radius=20, degree=2)
            original = path.read_bytes()
            descriptors = read_zernike(path, {})
            for dimensions in (2, 3):
                coordinates = np.arange(len(descriptors.frames) * dimensions, dtype=float).reshape(-1, dimensions)
                parameters = {'n_components': dimensions, 'metric': 'euclidean'}
                model = Mock()
                model.fit_transform.return_value = coordinates
                model.get_params.return_value = parameters
                factory = Mock(return_value=model)
                settings = dict(radius=20, degree=2, n_components=dimensions, z_score=True)
                with patch.dict('sys.modules', {'umap': SimpleNamespace(UMAP=factory)}):
                    _, output = shape_features_job(dataset, **settings)
                    saved = read_embedding(output, {'n_components': dimensions})
                    np.testing.assert_array_equal(saved.embedding.coordinates, coordinates)
                    np.testing.assert_array_equal(saved.frames, descriptors.frames)
                    np.testing.assert_array_equal(saved.identities, descriptors.identities)
                    np.testing.assert_allclose(saved.embedding.mean, descriptors.values.mean(axis=0))
                    np.testing.assert_allclose(model.fit_transform.call_args.args[0],
                                              (descriptors.values - saved.embedding.mean) / saved.embedding.scale)
                    self.assertEqual(saved.embedding.parameters, parameters)
                    self.assertEqual(saved.metadata['feature_id'], descriptors.metadata['feature_id'])
                    inputs = shape_analysis_inputs(dataset, settings)
                    np.testing.assert_array_equal(inputs['coordinates'], coordinates)
                    np.testing.assert_array_equal(inputs['frames'], descriptors.frames)
                    np.testing.assert_array_equal(inputs['identities'], descriptors.identities)
                    shape_features_job(dataset, **settings)
                    factory.assert_called_once_with(n_components=dimensions)
            self.assertEqual(path.read_bytes(), original)

    def test_failed_and_cancelled_feature_writes_allow_retry(self):
        with TemporaryDirectory() as root:
            dataset, session, _ = self.dataset(root)
            originals = {path: path.read_bytes() for path in
                         (dataset.mask('forward'), dataset.mask('reverse'), session.path)}
            destination = zernike_path(dataset)
            partial = auxiliary_path(destination, 'partial')
            with self.assertLogs('bivosia.processing.analysis.mask_shape', level='WARNING'):
                with self.assertRaisesRegex(ValueError, 'At least four valid animal masks'):
                    extract_zernike(dataset, radius=1, degree=2)
            self.assertFalse(destination.exists())
            self.assertFalse(partial.exists())

            written = []
            def append(*args):
                append_zernike(*args)
                written.append(True)
            with patch('bivosia.processing.analysis.mask_shape.append_zernike', side_effect=append):
                with self.assertRaises(InterruptedError):
                    extract_zernike(dataset, radius=20, degree=2, block_size=1,
                                    cancelled=lambda: bool(written))
            self.assertTrue(written)
            self.assertFalse(destination.exists())
            self.assertFalse(partial.exists())

            extract_zernike(dataset, radius=20, degree=2)
            originals[destination] = destination.read_bytes()
            with self.assertRaises(FileExistsError):
                extract_zernike(dataset, radius=20, degree=2)
            for path, original in originals.items():
                self.assertEqual(path.read_bytes(), original)

    def test_radius_excludes_only_affected_animal_and_logs_warning(self):
        with TemporaryDirectory() as root:
            dataset, session, _ = self.dataset(root)
            with h5py.File(dataset.mask('forward'), 'r+') as file:
                file['masks'][0, 0, 11] = 1
            originals = {path: path.read_bytes() for path in
                         (dataset.mask('forward'), dataset.mask('reverse'), session.path)}
            with self.assertLogs('bivosia.processing.analysis.mask_shape', level='WARNING') as warnings:
                output = extract_zernike(dataset, radius=2, degree=2)
            descriptors = read_zernike(output, {})
            self.assertEqual(descriptors.identities[descriptors.frames == 0].tolist(), [2])
            self.assertEqual(descriptors.identities[descriptors.frames == 1].tolist(), [1, 2])
            self.assertEqual(len(warnings.records), 1)
            self.assertIn('Frame 0, animal 1: excluded', warnings.output[0])
            self.assertIn('Zernike radius 2 is too small', warnings.output[0])
            for path, original in originals.items():
                self.assertEqual(path.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, 'positive Zernike radius'):
                frame_zernike_features(np.ones((3, 3), dtype=np.uint8), radius=0)

    def test_features_match_and_sync_uses_selected_source(self):
        with TemporaryDirectory() as root:
            dataset, session, _ = self.dataset(root)
            settings = dict(step_settings(dataset, 'shape_features'), radius=20, degree=2)
            path = extract_zernike(dataset, radius=20, degree=2)
            direct = read_zernike(path, {'mask_source': 'annotation'})
            self.assertNotIn(4, direct.frames)
            self.assertEqual(direct.identities[direct.frames == 6].tolist(), [2])
            path.rename(path.with_suffix('.saved.h5'))
            self.export(dataset, session)
            path = extract_zernike(dataset, radius=20, degree=2)
            exported = read_zernike(path, {'mask_source': 'export'})
            for name in ('values', 'frames', 'identities'):
                np.testing.assert_array_equal(getattr(direct, name), getattr(exported, name))
            location = {'host': '', 'path': str(dataset.directory)}
            for source in ('auto', 'export', 'annotation'):
                with self.subTest(source=source), patch('bivosia.processing.storage.consolidation.plan_consolidation') as plan:
                    plan_input_sync(dataset, location, [], 'shape_features', settings=dict(settings, mask_source=source))
                    required = plan.call_args.kwargs['required']
                    has_export = dataset.relative(dataset.export_path(session.export['path'])) in required
                    self.assertEqual(has_export, source != 'annotation')
                    self.assertEqual(dataset.relative(dataset.mask('forward')) in required, source == 'annotation')
            saved = read_json(session.path)
            self.assertEqual(saved['export']['fingerprint'], session.fingerprint())
            other = dataset.new_annotation(.6, .9, prefix='other_')
            atomic_json(other, {**saved, 'export': None})
            dataset.select_annotation(other)
            result = dispatch({'action': 'analysis_inputs', 'dataset': root, 'operation': 'shape_features',
                               'settings': dict(settings, mask_source='export'),
                               'annotation': dataset.relative(session.path)})
            self.assertEqual(result['message'], 'Mask source: Exported masks')
            self.assertEqual(Dataset(root).annotation, other)
            with patch('bivosia.processing.storage.consolidation.plan_consolidation') as plan:
                plan_input_sync(dataset, location, [], 'shape_features', settings=settings,
                                annotation=dataset.relative(session.path))
                self.assertIn(dataset.relative(dataset.export_path(session.export['path'])),
                              plan.call_args.kwargs['required'])


if __name__ == '__main__':
    unittest.main()
