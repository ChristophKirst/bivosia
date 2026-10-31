"""Focused checks for category-aware formats and backend-owned settings."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path
from unittest.mock import patch
import unittest

from bivosia.config import DATA_FORMATS, KEY_FRAME_BACKENDS, VOS_BACKENDS
from bivosia.dataset.io.registry import format_handler, format_spec, format_suffixes
from bivosia.dataset.io.hdf5 import Hdf5MaskReader
from bivosia.dataset.io.sleap import SleapKeypointReader
from bivosia.segmentation.image import create_segmenter as image_segmenter
from bivosia.segmentation.video import create_segmenter as video_segmenter


class RegistryTests(unittest.TestCase):
    def test_formats_are_category_specific_and_report_capabilities(self):
        self.assertIs(format_handler('masks', 'labels.h5'), Hdf5MaskReader)
        self.assertIs(format_handler('keypoints', 'points.h5'), SleapKeypointReader)
        with self.assertRaisesRegex(ValueError, 'no writer'):
            format_handler('keypoints', 'points.h5', operation='writer')
        with patch.dict(DATA_FORMATS['masks'], {'other': DATA_FORMATS['masks']['hdf5']}):
            with self.assertRaisesRegex(ValueError, 'Ambiguous'):
                format_spec('masks', 'labels.h5')
            self.assertIs(format_handler('masks', 'labels.h5', format='hdf5'), Hdf5MaskReader)
            self.assertEqual(format_suffixes('masks').count('.h5'), 1)

    def test_models_take_independent_settings_from_their_own_backend(self):
        cutie = video_segmenter('cutie', parameters={'internal_size': 256})
        self.assertEqual(cutie.parameters['internal_size'], 256)
        self.assertEqual(VOS_BACKENDS['cutie']['defaults']['internal_size'], 480)
        self.assertEqual(cutie.checkpoint, VOS_BACKENDS['cutie']['checkpoint'])
        self.assertEqual(cutie.reference_modes, VOS_BACKENDS['cutie']['reference_modes'])
        with patch.dict(KEY_FRAME_BACKENDS['sam2'], {'checkpoint': Path('/tmp/custom-sam.pt'),
                                                    'defaults': {'config': 'custom.yaml'}}):
            sam2 = image_segmenter('sam2')
            self.assertEqual(sam2.checkpoint, Path('/tmp/custom-sam.pt'))
            self.assertEqual(sam2.parameters, {'config': 'custom.yaml'})
            self.assertEqual(sam2.metadata['config'], 'custom.yaml')
        self.assertEqual(image_segmenter('sam3').parameters, {})
        for model in (cutie, sam2):
            model.check_parameters()
            model.close()


if __name__ == '__main__':
    unittest.main()
