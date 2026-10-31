"""Dataset-relative paths, selection, relocation and structural checks."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import unittest

import numpy as np

from bivosia.dataset.annotations.masks import RevisionSession
from bivosia.analysis.mask_linking import initial_draft
from bivosia.analysis.zones_of_disagreement import ZodAnnotation
from bivosia.config import DATASET_LAYOUT
from bivosia.dataset.workflow.jobs import Job
from bivosia.dataset.annotations.keyframes import KeyFrameSession, MaskDraft
from bivosia.dataset.paths import folder_path
from bivosia.dataset.layout import Dataset


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "dataset"
        self.root.mkdir()
        self.data = Dataset(self.root)

    def session(self):
        return RevisionSession({"seed": .6, "recover": .9, "gap": 0}, ((3, 6),),
                               [ZodAnnotation("a", 3, 6)], np.zeros(10, bool), np.zeros(10, bool), 2)

    def test_defaults_follow_shared_configuration(self):
        folders = DATASET_LAYOUT['folders']
        with patch.dict(folders['recording'], name='video-data'), \
             patch.dict(folders['masks'], name='segmentations'), \
             patch.dict(folders['analysis'], name='results'), \
             patch.dict(folders['masks']['files']['forward'], name='f.h5'):
            self.assertEqual(self.data.mask("forward"), self.root / "segmentations/f.h5")
            self.assertEqual(self.data.recording, self.root / "video-data")
            self.assertEqual(self.data.cache, self.root / folder_path("cache"))

    def test_key_frames_and_jobs_follow_declared_folders(self):
        session = KeyFrameSession(self.data, 3, (2, 2))
        session.label(0, MaskDraft(np.ones((2, 2), np.uint8)))
        session.save()
        self.assertEqual(self.data.file('key_frames'), self.data.masks / 'key_frame/key_frames.json')
        self.assertEqual(self.data.file('key_frame_masks'), self.data.masks / 'key_frame/key_frames.h5')
        self.assertIsNone(self.data.annotation)
        restored = KeyFrameSession(self.data, 3, (2, 2))
        np.testing.assert_array_equal(restored.masks[0], session.masks[0])
        analysis = DATASET_LAYOUT['folders']['analysis']
        with patch.dict(analysis, name='results', storage='external'), \
             patch.dict(analysis['folders']['jobs'], name='runs'), \
             patch.dict(analysis['files']['mask_features']['companions'], metadata='{stem}.meta.json'):
            external = self.root.parent / 'external'
            external.mkdir()
            self.data.settings['storage_folders'] = [str(external)]
            job = Job.create(self.data, 'vos', {}, job_id='abc')
            self.assertEqual(job.path('settings'), external / 'results/runs/abc/job.json')
            self.assertEqual(job.status['state'], 'queued')
            self.assertEqual(self.data.file('mask_features', companion='metadata'),
                             external / 'results/mask_features.meta.json')

    def test_video_and_annotation_auto_selection(self):
        self.data.recording.mkdir()
        (self.data.recording / "movie.mp4").touch()
        self.assertEqual(self.data.video.name, "movie.mp4")
        self.assertIsNone(self.data.annotation)
        session = self.session()
        first = self.data.new_annotation(.6, .9)
        session.save(first)
        self.data.select_annotation(first)
        self.assertNotIn("annotation", self.data.settings)
        self.assertEqual(self.data.annotation, first)
        second = self.data.new_annotation(.6, .9, "second_")
        self.session().save(second)
        with self.assertRaisesRegex(ValueError, "Several annotations"):
            _ = Dataset(self.root).annotation
        self.data.select_annotation(second)
        self.assertEqual(Dataset(self.root).annotation, second)
        self.assertFalse(Path(self.data.settings["annotation"]).is_absolute())

    def test_external_storage_and_relative_locations(self):
        relative = Dataset(self.root, "../external", "movie.mp4")
        self.assertEqual(relative.video, self.root.parent / "external/recording/movie.mp4")
        relative.save()
        self.assertEqual(Dataset(self.root).mask("forward"), self.root.parent / "external/masks/masks_forward.h5")
        external = Dataset(self.root, self.root.parent / "external", "movie.mp4")
        external.save()
        self.assertEqual(Dataset(self.root).recording, self.root.parent / "external/recording")

    def test_lookup_all_storage_folders_and_configured_writes(self):
        external = self.root.parent / "external"
        other = self.root.parent / "other"
        for root in (self.root, external, other):
            (root / "masks").mkdir(parents=True)
            (root / "analysis").mkdir(parents=True)
        self.data.settings["storage_folders"] = ["../external", "../other"]
        local_mask = self.root / "masks/masks_forward.h5"
        other_mask = other / "masks/masks_reverse.h5"
        local_mask.touch()
        other_mask.touch()
        self.assertEqual(self.data.mask("forward"), local_mask)
        self.assertEqual(self.data.mask("reverse"), other_mask)
        annotation = self.data.new_annotation(.6, .9)
        self.assertEqual(annotation.parent, self.root / "annotations")
        self.assertEqual(self.data.linked_masks(annotation).parent, external / "masks")
        self.assertEqual(self.data.array("mask_quality", write=True), self.root / "analysis/mask_quality.npy")
        (other / "analysis/mask_quality.npy").touch()
        self.assertEqual(self.data.array("mask_quality"), other / "analysis/mask_quality.npy")
        with self.assertRaisesRegex(ValueError, "outside its write location"):
            self.data.array("mask_quality", write=True)
        with patch.dict(DATASET_LAYOUT["folders"]["analysis"], storage="external"):
            self.assertEqual(self.data.array("mask_missing", write=True), external / "analysis/mask_missing.npy")
        (external / "masks/masks_forward.h5").touch()
        with self.assertRaisesRegex(ValueError, "multiple storage folders"):
            self.data.mask("forward")
        self.data.settings["storage_folders"] = ["../unavailable"]
        with self.assertRaisesRegex(FileNotFoundError, "Storage folder is unavailable"):
            self.data.linked_masks(annotation)

    def test_relocation_preserves_annotations_history_and_export_path(self):
        session = self.session()
        session.edit("a", "source", source="reverse")
        state, history = session._state(), session.undo_stack
        path = self.data.new_annotation(.6, .9)
        session.save(path)
        self.data.select_annotation(path)
        payload = json.loads(path.read_text())
        self.assertEqual(set(payload["header"]), {"seed", "recover", "gap"})
        moved = self.root.with_name("renamed")
        self.root.rename(moved)
        dataset = Dataset(moved)
        restored = RevisionSession.load(dataset.annotation, np.zeros(10, bool), np.zeros(10, bool), 2)
        self.assertEqual(restored._state(), state)
        self.assertEqual(restored.undo_stack, history)
        self.assertEqual(restored.fingerprint(), session.fingerprint())
        self.assertEqual(dataset.export_path("x.h5"), moved / "masks/x.h5")

    def test_annotation_and_video_selections_stay_in_their_folders(self):
        with self.assertRaises(ValueError):
            self.data.annotation_path("outside.json")
        with self.assertRaises(ValueError):
            Dataset(self.root, video="../outside.mp4")

    def test_annotation_frame_and_label_checks(self):
        session = self.session()
        job = session.jobs[0]
        session.set_draft(job.id, {**initial_draft(job), "mapping": [0, 1]})
        path = self.data.new_annotation(.6, .9)
        session.save(path)
        with self.assertRaises(ValueError):
            RevisionSession.load(path, np.zeros(4, bool), np.zeros(4, bool), 2)
        with self.assertRaises(ValueError):
            RevisionSession.load(path, np.zeros(10, bool), np.zeros(10, bool), 3)


if __name__ == "__main__":
    unittest.main()
