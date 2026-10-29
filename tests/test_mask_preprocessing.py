"""Check body preprocessing and preservation of consensus-mask analysis inputs."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from itertools import product
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from bivosia.analysis.mask_preprocessing import preprocess_mask, preprocessing_settings
from bivosia.analysis.revision import revision_fingerprint
from bivosia.analysis.shape_features import frame_zernike_features
from bivosia.config import ANALYSIS_DEFAULTS
from bivosia.dataset.containers import container_output
from bivosia.dataset.jsonio import atomic_json
from bivosia.dataset.layout import Dataset
from bivosia.dataset.masks import linked_mask_arrays
from bivosia.dataset.shape_features import (
    embedding_path, read_zernike, shape_analysis_inputs, zernike_path,
)
from bivosia.processing.shape_features import extract_zernike


class MaskPreprocessingTests(unittest.TestCase):
    def test_fragment_thresholds_and_opening_preserve_source(self):
        mask = np.zeros((40, 50), dtype=bool)
        mask[2:12, 2:12] = mask[2:8, 14:20] = mask[30, 40] = True
        original = mask.copy()
        distance = float(np.hypot(10, 2))
        parameters = dict(main_body_area=100, fragment_distance=distance, opening_radius=0)
        self.assertEqual(int(preprocess_mask(mask, **parameters).sum()), 136)
        self.assertEqual(int(preprocess_mask(mask, **{**parameters, 'main_body_area': 99}).sum()), 100)
        self.assertEqual(int(preprocess_mask(mask, **{**parameters, 'fragment_distance': distance - .01}).sum()), 100)
        np.testing.assert_array_equal(mask, original)
        body = np.zeros((40, 50), dtype=bool)
        body[8:30, 8:30] = True
        body[18, 30:45] = True
        body[16:20, 16:20] = False
        cleaned = preprocess_mask(body, opening_radius=3)
        self.assertFalse(cleaned[:, 32:].any())
        self.assertFalse(cleaned[16:20, 16:20].any())
        self.assertTrue(cleaned[12, 12])
        self.assertTrue(np.all(cleaned <= body))
        self.assertEqual(cleaned.shape, body.shape)
        self.assertFalse(preprocess_mask(np.zeros_like(body)).any())
        self.assertFalse(preprocess_mask(np.eye(5, dtype=bool), opening_radius=1).any())

    def test_steps_can_be_disabled_independently(self):
        mask = np.zeros((40, 50), dtype=bool)
        mask[2:12, 2:12] = mask[2:8, 14:20] = mask[30, 40] = True
        original = mask.copy()
        for area, distance, opening in product((False, True), repeat=3):
            with self.subTest(area=area, distance=distance, opening=opening):
                cleaned = preprocess_mask(mask, filter_by_area=area, main_body_area=99,
                                          filter_by_distance=distance, fragment_distance=20,
                                          apply_opening=opening, opening_radius=1)
                expected = (96 if area else 128) if opening else (100 if area else 136 if distance else 137)
                self.assertEqual(int(cleaned.sum()), expected)
                np.testing.assert_array_equal(mask, original)
                self.assertFalse(np.shares_memory(cleaned, mask))
        np.testing.assert_array_equal(preprocess_mask(mask, main_body_area=100,
                                      filter_by_distance=False, apply_opening=False), mask)
        self.assertEqual(int(preprocess_mask(mask, filter_by_area=False, fragment_distance=2,
                                            apply_opening=False).sum()), 100)

    def test_optional_extraction_keeps_results_and_sources_separate(self):
        with TemporaryDirectory() as directory:
            dataset = Dataset(directory)
            record = {'header': {'frame_count': 4}, 'zones': [], 'links': [], 'link_frames': []}
            fingerprint = revision_fingerprint(record['header'], [], [], [])
            annotation = dataset.annotations / 'review.json'
            export = dataset.masks / 'review.linked.h5'
            record['export'] = {'path': export.name, 'fingerprint': fingerprint}
            atomic_json(annotation, record)
            labels = np.zeros((4, 48, 64), dtype=np.uint16)
            labels[:, 3:15, 3:15] = 1
            labels[:, 25:40, 35:50] = 4
            labels[:, 8, 15:25] = 1
            labels[:, 45, 4] = 1
            labels[:, 45, 60] = 7
            with container_output(export, overwrite=False) as output:
                arrays = linked_mask_arrays(output, labels.shape, dataset.relative(annotation), fingerprint)
                arrays['preferred'][:] = labels
                arrays['forward'][:] = arrays['reverse'][:] = 0
                arrays['preferred_valid'][:] = [True, False, True, True]
                arrays['direction_valid'][:] = True
                arrays['source'][:] = 0
            originals = {path: path.read_bytes() for path in (annotation, export)}
            raw = extract_zernike(dataset, radius=70, degree=2, block_size=2)
            originals[raw] = raw.read_bytes()
            parameters = preprocessing_settings(True, main_body_area=100, fragment_distance=8, opening_radius=2)
            progress = []
            cleaned = extract_zernike(dataset, radius=70, degree=2, block_size=2,
                                      preprocess_masks=True, **parameters,
                                      progress=lambda done, total: progress.append((done, total)))
            values, frames, identities, metadata = read_zernike(cleaned, {'preprocessing': parameters})
            self.assertEqual(frames.tolist(), [0, 0, 2, 2, 3, 3])
            self.assertEqual(identities.tolist(), [1, 4, 1, 4, 1, 4])
            expected_ids, expected = frame_zernike_features(labels[0], 70, 2, preprocessing=parameters)
            np.testing.assert_allclose(values[:2], expected)
            np.testing.assert_array_equal(expected_ids, [1, 4])
            self.assertEqual(metadata['image_shape'], [48, 64])
            self.assertEqual(progress[-1], (4, 4))
            self.assertNotEqual(raw, cleaned)
            self.assertEqual(cleaned, zernike_path(dataset, preprocess_masks=True))
            for z_score in (False, True):
                self.assertNotEqual(embedding_path(dataset, z_score=z_score),
                                    embedding_path(dataset, z_score=z_score, preprocess_masks=True))
            settings = dict(ANALYSIS_DEFAULTS['shape_features'], **parameters, radius=70, degree=2)
            shape_analysis_inputs(dataset, settings)
            settings['preprocess_masks'] = True
            shape_analysis_inputs(dataset, settings)
            settings['opening_radius'] = 3
            with self.assertRaisesRegex(ValueError, 'different masks or settings'):
                shape_analysis_inputs(dataset, settings)
            settings['opening_radius'] = 2
            settings['apply_opening'] = False
            with self.assertRaisesRegex(ValueError, 'different masks or settings'):
                shape_analysis_inputs(dataset, settings)
            with self.assertRaises(FileExistsError):
                extract_zernike(dataset, radius=70, degree=2, preprocess_masks=True, **parameters)
            for path, before in originals.items():
                self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
