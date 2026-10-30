"""Focused checks for mask-format parity, validity and output preservation."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import os
import unittest

import numpy as np

from bivosia.dataset.arrays import array_output
from bivosia.dataset.containers import container_output
from bivosia.dataset.layout import Dataset
from bivosia.dataset.mask_output import mask_outputs, write_masks
from bivosia.dataset.masks import open_masks
from bivosia.dataset.numpy_masks import validity_path
from bivosia.dataset.paths import auxiliary_path
from bivosia.dataset.segmentation import vos_outputs
from bivosia.dataset.video import VideoSet
from bivosia.processing.quantify import quantify_many


class MaskIOTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.labels = np.tile(np.array([[[1, 1, 0, 2, 2]]], np.uint8), (3, 1, 1))

    def test_formats_preserve_labels_validity_and_existing_outputs(self):
        valid = np.array([True, False, True])
        for suffix in ('.h5', '.npy', '.npz'):
            with self.subTest(suffix=suffix):
                path = self.root / ('masks' + suffix)
                write_masks(path, self.labels, valid=valid)
                with open_masks(path, require_valid=False) as masks:
                    np.testing.assert_array_equal(masks[:], self.labels)
                    np.testing.assert_array_equal(masks.valid, valid)
                    self.assertEqual(isinstance(masks.masks, np.memmap), suffix == '.npy')
                before = path.read_bytes()
                with self.assertRaises(ValueError):
                    open_masks(path)
                with self.assertRaises(FileExistsError):
                    write_masks(path, self.labels)
                self.assertEqual(before, path.read_bytes())

    def test_numpy_sidecar_is_optional_and_protected(self):
        path = self.root / 'masks.npy'
        write_masks(path, self.labels)
        self.assertFalse(validity_path(path).exists())
        with open_masks(path) as masks:
            self.assertTrue(masks.valid.all())
        other = self.root / 'other.npy'
        np.save(validity_path(other), np.ones(3, dtype=bool))
        with self.assertRaises(FileExistsError):
            write_masks(other, self.labels)
        self.assertFalse(other.exists())

    def test_all_formats_feed_the_same_quantification_and_viewer(self):
        for suffix in ('.h5', '.npy', '.npz'):
            dataset = Dataset(self.root / suffix[1:])
            dataset.settings['masks'] = {name: name + suffix for name in ('forward', 'reverse')}
            with vos_outputs(dataset, ('forward', 'reverse'), 3, (1, 5), (0, 2)) as outputs:
                for masks, valid in outputs.values():
                    masks[:] = self.labels
                    valid[:] = True
            quantify_many([dataset], 2, workers=1)
            np.testing.assert_array_equal(np.load(dataset.array('mask_iou_match')), [[0, 1]] * 3)
            videos = VideoSet({name: dataset.mask(name) for name in ('forward', 'reverse')})
            try:
                np.testing.assert_array_equal(videos.read('forward', 1), self.labels[1])
            finally:
                videos.close()

    def test_incomplete_vos_outputs_are_not_published(self):
        for suffix in ('.h5', '.npy', '.npz'):
            dataset = Dataset(self.root / suffix[1:])
            dataset.settings['masks'] = {name: name + suffix for name in ('forward', 'reverse')}
            with self.assertRaises(ValueError):
                with vos_outputs(dataset, ('forward', 'reverse'), 3, (1, 5), (1, 2)) as outputs:
                    for masks, valid in outputs.values():
                        masks[:] = self.labels
                        valid[:] = False
            self.assertFalse(dataset.mask('forward').exists())
            self.assertFalse(dataset.mask('reverse').exists())
            self.assertFalse(list(dataset.masks.glob('*.lock')))
            self.assertFalse(list(dataset.masks.glob('*.partial*')))
            write_masks(dataset.mask('forward'), self.labels)
            with open_masks(dataset.mask('forward')) as masks:
                np.testing.assert_array_equal(masks[:], self.labels)

    def test_failed_container_and_array_writes_preserve_completed_results(self):
        for suffix in ('.h5', '.npy'):
            with self.subTest(suffix=suffix):
                path = self.root / ('result' + suffix)
                original = b'previous completed result'
                path.write_bytes(original)
                writer = container_output(path) if suffix == '.h5' else array_output(path, (3,), 'i8')
                with self.assertRaises(InterruptedError):
                    with writer as output:
                        if suffix == '.h5':
                            output.create_dataset('values', data=[1, 2, 3])
                        else:
                            output[:] = [1, 2, 3]
                        raise InterruptedError('Cancelled')
                self.assertEqual(path.read_bytes(), original)
                self.assertFalse(auxiliary_path(path, 'partial').exists())

        other = self.root / 'other.h5'
        partial = auxiliary_path(other, 'partial')
        partial.write_bytes(b'preexisting partial')
        with self.assertRaises(ValueError):
            with container_output(other, overwrite=False):
                self.fail('Existing partial was accepted')
        self.assertEqual(partial.read_bytes(), b'preexisting partial')

    def test_failed_mask_publication_removes_only_its_own_files(self):
        for suffix in ('.h5', '.npy', '.npz'):
            with self.subTest(suffix=suffix):
                paths = {name: self.root / (name + suffix) for name in ('forward', 'reverse')}
                replace = os.replace
                calls = []
                def publish(source, target):
                    calls.append(target)
                    if len(calls) == 2:
                        raise OSError('Publication failed')
                    replace(source, target)
                with patch('bivosia.dataset.mask_output.os.replace', side_effect=publish):
                    with self.assertRaisesRegex(OSError, 'Publication failed'):
                        with mask_outputs(paths, self.labels.shape) as outputs:
                            for masks, valid in outputs.values():
                                masks[:] = self.labels
                                valid[:] = [True, False, True]
                self.assertFalse(list(self.root.iterdir()))
                write_masks(paths['forward'], self.labels)
                saved = paths['forward'].read_bytes()
                with self.assertRaises(FileExistsError):
                    with mask_outputs(paths, self.labels.shape):
                        self.fail('Existing result was accepted')
                self.assertEqual(paths['forward'].read_bytes(), saved)
                paths['forward'].unlink()

    def test_discovery_rejects_ambiguous_formats(self):
        dataset = Dataset(self.root)
        first = dataset.masks / 'masks_forward.npy'
        write_masks(first, self.labels)
        self.assertEqual(dataset.mask('forward'), first)
        write_masks(first.with_suffix('.npz'), self.labels)
        with self.assertRaisesRegex(ValueError, 'Several forward masks'):
            dataset.mask('forward')
        dataset.settings['masks'] = {'forward': first.name}
        self.assertEqual(dataset.mask('forward'), first)

    def test_two_pass_numpy_array_selects_direction(self):
        path = self.root / 'both.npy'
        np.save(path, np.stack((self.labels, np.zeros_like(self.labels))))
        with open_masks(path, direction='reverse') as masks:
            np.testing.assert_array_equal(masks[:], np.zeros_like(self.labels))


if __name__ == '__main__':
    unittest.main()
