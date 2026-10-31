"""Check separate cleanup previews and preservation of original key-frame data."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import json
import unittest

import numpy as np
from PySide6.QtWidgets import QApplication

from bivosia.analysis.masks import clean_label_mask
from bivosia.config import ANALYSIS_DEFAULTS, DATASET_LAYOUT
from bivosia.dataset.io.containers import open_container
from bivosia.dataset.annotations.keyframes import KeyFrameSession, MaskDraft, save_cleaned_key_frame
from bivosia.dataset.layout import Dataset
from bivosia.ui.mask_editor import MaskEditor
from bivosia.ui.mask_cleanup_dialog import MaskCleanupDialog


class MaskCleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.dataset = Dataset(self.temporary.name)
        self.original = np.zeros((128, 160), dtype=np.uint8)
        self.original[10:50, 10:50] = 1
        self.original[10:50, 50:90] = 2
        self.original[70:110, 110:150] = 3
        self.original[20:25, 20:25] = 0
        self.original[120, 10] = 1
        self.session = KeyFrameSession(self.dataset, 2, self.original.shape, animals=3)
        for frame in (0, 1):
            self.session.label(frame, MaskDraft(self.original))
        self.session.save()
        self.original_files = {key: self.dataset.file(key).read_bytes()
                               for key in ('key_frames', 'key_frame_masks')}
        self.editor = MaskEditor(SimpleNamespace(label='SAM', device='cpu', metadata={'name': 'sam'}))
        self.editor.set_frame(self.session, 0, np.zeros((*self.original.shape, 3), dtype=np.uint8))
        self.addCleanup(self.close_editor)

    def close_editor(self):
        self.editor.discard_preview()
        self.editor._clear_preview()
        self.editor.draft.mark_saved()
        self.editor.close()

    def assert_originals_unchanged(self):
        for key, content in self.original_files.items():
            self.assertEqual(self.dataset.file(key).read_bytes(), content)
        for mask in self.session.masks.values():
            np.testing.assert_array_equal(mask, self.original)
        np.testing.assert_array_equal(self.editor.draft.mask, self.original)

    def test_settings_change_cleanup_without_changing_originals(self):
        values = dict(contour_min_area=0, contour_max_area=100000, overlap_kernel=1)
        def choose(dialog):
            dialog.parameters.set_values(values)
            dialog.accept()
            return dialog.result()
        with patch.object(MaskCleanupDialog, 'exec', choose):
            self.editor.cleanup_settings_button.click()
        self.assertEqual(self.editor.cleanup_options, values)
        self.assertIsNone(self.editor.cleanup_draft)
        self.assert_originals_unchanged()
        self.editor.clean_up()
        np.testing.assert_array_equal(self.editor.active_draft.mask, clean_label_mask(self.original, **values))
        self.assertEqual(self.editor.cleanup_parameters, values)
        self.assertTrue(self.editor.save())
        self.assert_originals_unchanged()
        with open_container(self.dataset.file('cleaned_key_frame_masks')) as source:
            self.assertEqual(json.loads(source.attrs['parameters'])['0'], values)
        for button in (self.editor.undo_button, self.editor.redo_button, self.editor.clear_points_button):
            self.assertFalse(button.icon().isNull())

    def test_preview_undo_and_separate_save_preserve_originals(self):
        cleaned = clean_label_mask(self.original)
        self.assertEqual(int(cleaned[120, 10]), 0)
        self.assertEqual(int(cleaned[22, 22]), 1)
        self.assertTrue(np.all(cleaned[10:50, 48:52] == 0))
        self.assertTrue(np.all(cleaned[70:110, 110:150] == 3))
        self.editor.clean_up()
        self.assertEqual(self.editor.save_button.text(), 'Save cleaned')
        self.assertFalse(self.dataset.file('cleaned_key_frame_masks').exists())
        self.assert_originals_unchanged()
        self.editor.undo_button.click()
        np.testing.assert_array_equal(self.editor.active_draft.mask, self.original)
        self.editor.redo_button.click()
        np.testing.assert_array_equal(self.editor.active_draft.mask, cleaned)
        self.assertTrue(self.editor.save())
        self.assert_originals_unchanged()
        self.editor.discard_preview()
        np.testing.assert_array_equal(self.editor.active_draft.mask, self.original)
        self.assertEqual(self.editor.save_button.text(), 'Save')

        self.editor.set_frame(self.session, 1, self.editor.image)
        self.editor.clean_up()
        self.assertTrue(self.editor.save())
        with open_container(self.dataset.file('cleaned_key_frame_masks')) as source:
            self.assertEqual(source['frames'][:].tolist(), [0, 1])
            np.testing.assert_array_equal(source['masks'][0], cleaned)
            np.testing.assert_array_equal(source['masks'][1], cleaned)
            self.assertEqual(json.loads(source.attrs['parameters'])['0'], ANALYSIS_DEFAULTS['masks']['cleanup'])
        self.assert_originals_unchanged()

    def test_cleanup_keeps_pending_predictions_separate(self):
        self.editor.predictions = {1: (self.original == 1, .9), 2: (self.original == 2, .8)}
        self.editor._update_preview()
        review = self.editor.preview_mask.copy()
        predictions = {key: (mask.copy(), score) for key, (mask, score) in self.editor.predictions.items()}
        self.editor.clean_up()
        self.editor.tools.set_mode('Erase')
        self.editor.begin_stroke(20, 20)
        self.editor.history(False)
        self.editor.approve()
        self.assertTrue(self.editor.save())
        self.assertIsNone(self.editor.pending_model)
        np.testing.assert_array_equal(self.editor.preview_mask, review)
        for key, (mask, score) in predictions.items():
            np.testing.assert_array_equal(self.editor.predictions[key][0], mask)
            self.assertEqual(self.editor.predictions[key][1], score)
        self.assert_originals_unchanged()
        self.editor.discard_preview()
        np.testing.assert_array_equal(self.editor.active_draft.mask, review)
        self.assertFalse(self.editor.save_button.isEnabled())

    def test_cleaned_output_cannot_replace_originals_or_other_video_results(self):
        settings = dict(frame_count=2, animals=3, parameters=ANALYSIS_DEFAULTS['masks']['cleanup'])
        path = save_cleaned_key_frame(self.dataset, 0, self.original, **settings)
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            save_cleaned_key_frame(self.dataset, 1, self.original, **{**settings, 'frame_count': 3})
        self.assertEqual(path.read_bytes(), before)
        files = DATASET_LAYOUT['folders']['masks']['folders']['key_frame']['files']
        with patch.dict(files['cleaned_key_frame_masks'], name=files['key_frame_masks']['name']):
            with self.assertRaisesRegex(ValueError, 'separately'):
                save_cleaned_key_frame(self.dataset, 0, self.original, **settings)
        self.assert_originals_unchanged()


if __name__ == '__main__':
    unittest.main()
