"""Check frame transport, format reuse and temporary analysis snapshots."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest

import h5py
import numpy as np

from bivosia.config import PACKAGE_DIRECTORY
from bivosia.dataset.layout import Dataset
from bivosia.dataset.storage.remote_reader import RemoteDatasetReader
from bivosia.processing.remote.client import BOOTSTRAP, Remote


class LocalTransport(Remote):
    def command(self):
        return [sys.executable, '-c', BOOTSTRAP]


class RemoteReaderTests(unittest.TestCase):
    def test_formats_validity_indexing_and_cleanup(self):
        values = np.arange(60, dtype=np.uint8).reshape(3, 4, 5)
        valid = np.array([True, False, True])
        for suffix in ('.h5', '.npy', '.npz'):
            with self.subTest(suffix=suffix), TemporaryDirectory() as tmp:
                dataset = Dataset(tmp)
                path = dataset.mask('forward').with_suffix(suffix)
                path.parent.mkdir()
                if suffix == '.h5':
                    with h5py.File(path, 'w') as file:
                        file['masks'], file['valid'] = values, valid
                elif suffix == '.npy':
                    np.save(path, values)
                    np.save(path.with_name(path.stem + '.valid.npy'), valid)
                else:
                    np.savez_compressed(path, masks=values, valid=valid)
                reader = RemoteDatasetReader(LocalTransport('local', sys.executable, str(PACKAGE_DIRECTORY), tmp))
                try:
                    self.assertEqual(reader.mask_paths(), {'forward': path})
                    masks = reader.open_masks('forward')
                    np.testing.assert_array_equal(masks.valid, valid)
                    np.testing.assert_array_equal(masks[-1], values[-1])
                    np.testing.assert_array_equal(masks[::-1], values[::-1])
                    self.assertEqual(masks[:0].shape, (0, 4, 5))
                    self.assertIs(reader.open_masks('forward'), masks)
                    with self.assertRaises(IndexError):
                        masks[3]
                    dataset.analysis.mkdir()
                    np.save(dataset.array('mask_areas'), values)
                    snapshot = reader.analysis_dataset()
                    np.testing.assert_array_equal(np.load(snapshot.array('mask_areas')), values)
                    snapshot_root = snapshot.directory
                finally:
                    reader.close()
                self.assertFalse(snapshot_root.exists())
                self.assertIsNone(reader.session.process)
                with self.assertRaises(ValueError):
                    masks[0]


if __name__ == '__main__':
    unittest.main()
