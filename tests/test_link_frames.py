"""Independent link boundaries preserve zones and drive identity propagation."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import h5py
import numpy as np

from bivosia.processing.analysis.mask_export import export_masks
from bivosia.analysis.mask_linking import initial_draft, link_segments, match_masks, prepare_links, validation_errors
from bivosia.dataset.annotations.masks import RevisionSession
from bivosia.analysis.zones_of_disagreement import ZodAnnotation
from bivosia.dataset.layout import Dataset


class LinkFrameTests(unittest.TestCase):
    def session(self, zones=(), count=30):
        return RevisionSession({"seed": .6, "recover": .9, "gap": 0}, (), list(zones),
                               np.zeros(count, bool), np.zeros(count, bool), 2)

    def approve(self, session, job, mapping=(0, 1)):
        session.set_draft(job.id, {**initial_draft(job), "mapping": list(mapping), "manual": True})
        session.approve(job.id, mapping)

    def test_inside_zone_and_gap_preserve_geometry_sources_and_existing_links(self):
        zones = [ZodAnnotation("z", 10, 19, "reverse", True, "reviewed")]
        s = self.session(zones)
        for job in s.jobs:
            self.approve(s, job)
        existing = deepcopy(s.links)
        key = s.set_link_frame(12, True)
        self.assertEqual(key, "frame:12")
        self.assertEqual(s.zones, zones)
        for old_key, record in existing.items():
            self.assertEqual(s.links[old_key], record)
        job = next(j for j in s.jobs if j.id == key)
        self.assertEqual((job.left.end, job.right.start, job.left.source, job.right.source), (12, 13, "reverse", "reverse"))
        outside = s.set_link_frame(5, True)
        job = next(j for j in s.jobs if j.id == outside)
        self.assertEqual((job.left.source, job.right.source), ("forward", "forward"))
        self.assertEqual(s.zones, zones)
        parts = link_segments(s.zones, s.frame_count, s.link_frames)
        self.assertEqual([t for p in parts for t in range(p.start, p.end + 1)], list(range(30)))

    def test_coincident_zone_boundary_reuses_existing_link(self):
        s = self.session([ZodAnnotation("z", 10, 19, "reverse")])
        job = next(j for j in s.jobs if j.left.end == 9)
        self.approve(s, job)
        before = s._state()
        self.assertEqual(s.set_link_frame(9, True), job.id)
        self.assertEqual(s._state(), before)

    def test_undo_redo_delete_and_save_roundtrip(self):
        s = self.session()
        s.set_link_frame(5, True)
        self.assertEqual(s.link_frames, {5})
        s.undo()
        self.assertEqual(s.link_frames, set())
        s.redo()
        self.assertEqual(s.link_frames, {5})
        self.approve(s, s.jobs[0], (1, 0))
        state = s._state()
        s.set_link_frame(5, False)
        self.assertEqual(s.link_frames, set())
        s.undo()
        self.assertEqual(s._state(), state)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "seed_0.60_recover_0.90.json"
            s.save(path)
            loaded = RevisionSession.load(path, s.missing_forward, s.missing_reverse, 2)
            self.assertEqual(loaded._state(), s._state())
            loaded.undo()
            self.assertEqual(loaded.links['frame:5']['status'], 'draft')

    def test_invalid_frames_reject_without_changing_state(self):
        s = self.session([ZodAnnotation("invalid", 10, 19, "invalid")])
        before = s._state()
        for t in (-1, 9, 12, 19, 29):
            with self.subTest(t=t), self.assertRaises(ValueError):
                s.set_link_frame(t, True)
            self.assertEqual(s._state(), before)

    def test_event_survives_zone_changes_and_undo(self):
        s = self.session([ZodAnnotation("z", 10, 19, "forward", True, "reviewed")])
        s.set_link_frame(12, True)
        job = next(j for j in s.jobs if j.id == "frame:12")
        self.approve(s, job, (1, 0))
        before = s._state()
        s.edit("z", "source", source="reverse")
        self.assertEqual(s.links[job.id]['status'], 'needs_review')
        self.assertEqual(s.link_frames, {12})
        s.undo()
        self.assertEqual(s._state(), before)
        s.edit("z", "source", source="invalid")
        self.assertNotIn(job.id, s.links)
        self.assertEqual(s.link_frames, {12})
        s.undo()
        self.assertEqual(s._state(), before)

    def test_new_event_does_not_change_an_existing_event_approval(self):
        s = self.session()
        s.set_link_frame(8, True)
        self.approve(s, s.jobs[0], (1, 0))
        before = deepcopy(s.links['frame:8'])
        s.set_link_frame(3, True)
        s.set_link_frame(20, True)
        self.assertEqual(s.links['frame:8'], before)

    def test_automatic_identity_and_export_apply_single_frame_transition_without_writes(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            masks = np.tile(np.array([[1, 2], [1, 2]], dtype=np.uint8), (6, 1, 1))
            masks[3:] = 3 - masks[3:]
            dataset = Dataset(tmp)
            dataset.masks.mkdir(parents=True)
            for direction in ("forward", "reverse"):
                path = dataset.mask(direction)
                with h5py.File(path, "w") as file:
                    file.create_dataset("masks", data=masks)
            s = self.session(count=6)
            s.set_link_frame(2, True)
            state = deepcopy(s._state())
            links = prepare_links(s, lambda job: match_masks(masks[job.left.end], masks[job.right.start], 2))
            self.assertEqual(validation_errors(s, links), [])
            result = links.resolve()
            before, after = link_segments(s.zones, 6, s.link_frames)
            self.assertEqual((result.maps[before.id], result.maps[after.id]), ((0, 1), (1, 0)))
            self.assertEqual(result.automatic[s.jobs[0].id].quality, 1.)
            self.assertEqual(len(set(result.components.values())), 1)
            self.assertFalse(result.errors)
            result = export_masks(s, dataset, dataset.masks / "result.h5")
            with h5py.File(dataset.export_path(result['path']), "r") as output:
                expected = np.tile(masks[0], (6, 1, 1))
                np.testing.assert_array_equal(output['preferred'][:], expected)
                np.testing.assert_array_equal(output['forward'][:], expected)
                np.testing.assert_array_equal(output['reverse'][:], expected)
                self.assertTrue(output['preferred_valid'][:].all())
            self.assertEqual(s._state(), state)


if __name__ == "__main__":
    unittest.main()
