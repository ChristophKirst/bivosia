"""Check keypoint selection while protecting source files and existing results."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import h5py
import numpy as np

from bivosia.dataset.keypoints import select_keypoints
from bivosia.dataset.layout import Dataset


class KeypointImportTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'tracks.h5'
        with h5py.File(self.source, 'w') as file:
            file['tracks'] = np.zeros((2, 2, 1, 3), dtype=np.float32)
            file['node_names'] = np.array(['nose'], dtype=h5py.string_dtype())
            file['track_names'] = np.array(['a', 'b'], dtype=h5py.string_dtype())
        self.original = self.source.read_bytes()
        self.dataset = Dataset(self.root / 'dataset')
        self.dataset.settings['video'] = '/recordings/experiment.mp4'
        self.dataset.save()

    def test_copy_external_reuse_and_preserve_settings(self):
        external = self.root / 'external'
        external.mkdir()
        self.dataset.settings['storage_folders'] = [str(external)]
        self.dataset.save()
        editor = Dataset(self.dataset.directory)
        destination = select_keypoints(self.dataset, self.source, (480, 640))
        self.assertEqual(destination, external / 'keypoints/tracks.h5')
        self.assertEqual(destination.read_bytes(), self.original)
        self.assertEqual(self.source.read_bytes(), self.original)
        self.dataset.settings['keypoints']['invalid_frames'] = [1]
        self.dataset.save()
        # A file in a registered, non-default storage folder is reused in place.
        self.dataset.settings.pop('storage_folders')
        self.dataset.settings['storage_folders'] = [str(self.root / 'other'), str(external)]
        self.dataset.save()
        self.assertEqual(select_keypoints(self.dataset, destination, (960, 1280)), destination)
        editor.settings['animal_count'] = 2
        editor.save()
        saved = Dataset(self.dataset.directory)
        self.assertEqual(saved.settings['keypoints'], {
            'file': 'tracks.h5', 'format': 'sleap', 'image_shape': [960, 1280], 'invalid_frames': [1],
        })
        self.assertEqual(saved.settings['video'], '/recordings/experiment.mp4')
        self.assertEqual(saved.settings['animal_count'], 2)

    def test_invalid_input_and_conflict_preserve_files(self):
        description = self.dataset.description.read_bytes()
        for shape in ((0, 640), (480.5, 640)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                select_keypoints(self.dataset, self.source, shape)
        invalid = self.root / 'invalid.h5'
        with h5py.File(invalid, 'w') as file:
            file['tracks'] = np.zeros((3, 2))
        with self.assertRaisesRegex(ValueError, 'SLEAP'):
            select_keypoints(self.dataset, invalid, (480, 640))
        self.assertFalse(self.dataset.keypoints.exists())
        self.assertEqual(self.dataset.description.read_bytes(), description)
        destination = self.dataset.keypoints / self.source.name
        destination.parent.mkdir()
        destination.write_bytes(b'existing data')
        with self.assertRaises(FileExistsError):
            select_keypoints(self.dataset, self.source, (480, 640))
        self.assertEqual(destination.read_bytes(), b'existing data')
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual(self.dataset.description.read_bytes(), description)

    def test_existing_results_protect_source_selection(self):
        destination = select_keypoints(self.dataset, self.source, (480, 640))
        result = self.dataset.array('keypoint_alignment', write=True)
        result.parent.mkdir()
        result.write_bytes(b'existing alignment')
        self.assertEqual(select_keypoints(self.dataset, destination, (480, 640)), destination)
        description = self.dataset.description.read_bytes()
        for source, shape in ((destination, (960, 1280)), (self.source, (480, 640))):
            with self.subTest(source=source, shape=shape), self.assertRaisesRegex(ValueError, 'Existing keypoint results'):
                select_keypoints(self.dataset, source, shape)
        self.assertEqual(self.dataset.description.read_bytes(), description)
        self.assertEqual(result.read_bytes(), b'existing alignment')
