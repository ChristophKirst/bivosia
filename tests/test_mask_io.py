"""Focused checks for mask-format parity, validity and output preservation."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from bivosia.dataset.layout import Dataset
from bivosia.dataset.mask_output import write_masks
from bivosia.dataset.masks import open_masks
from bivosia.dataset.numpy_masks import validity_path
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
            self.assertTrue(list(dataset.masks.glob('*.partial*')))

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
