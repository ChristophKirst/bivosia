"""Tests for current-link and global identity colors."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
import unittest

import numpy as np

from bivosia.ui.link_colors import LinkColors, current_link_labels, global_labels, global_label_colors
from bivosia.ui.colors import bgr_colors, component_colors, MASK_COLORS_BGR
from bivosia.ui.keypoints import overlay_keypoints
from bivosia.processing.revision import RevisionData
from bivosia.dataset.session import RevisionSession
from bivosia.analysis.linking import Match, automatic_link, initial_draft, match_masks, prepare_links
from bivosia.analysis.zones_of_disagreement import RevisionZone


class LinkColorTests(unittest.TestCase):
    def session(self, zones=None):
        return RevisionSession({}, (),
                               [RevisionZone("a", 10, 19), RevisionZone("b", 30, 39)] if zones is None else zones,
                               np.zeros(50, bool), np.zeros(50, bool), 2)

    def approve(self, session, job, mapping):
        session.set_draft(job.id, {**initial_draft(job), "mapping": list(mapping)})
        session.approve(job.id, mapping)

    def test_current_link_uses_inverse_mapping_on_right(self):
        maps = current_link_labels((2, 0, 1), 3)
        self.assertEqual(maps, {"left": (0, 1, 2), "right": (1, 2, 0)})
        for i, j in enumerate((2, 0, 1)):
            self.assertEqual(maps["left"][i], maps["right"][j])

    def test_saved_automatic_and_approved_links_remain_distinct(self):
        session = self.session()
        session.header['mode'] = 'automatic'
        masks = np.array([[1, 1, 0, 2, 2]], dtype=np.uint8)
        for job in session.jobs:
            session.links[job.id] = automatic_link(job, lambda source, frame: masks, session.n)
        approved, invalidated = session.jobs[:2]
        self.approve(session, approved, (1, 0))
        session.links[invalidated.id]['context'] = 'changed boundary'
        before = deepcopy(session.links)
        matcher = Mock(side_effect=AssertionError('Opening colors must not calculate links'))
        colors = LinkColors(SimpleNamespace(linking=SimpleNamespace(boundary_match=matcher)))
        links = colors.links(session)
        self.assertEqual(links.mappings[approved.id], (1, 0))
        self.assertNotIn(approved.id, links.automatic)
        self.assertNotIn(invalidated.id, links.mappings)
        self.assertEqual(len(links.automatic), len(session.jobs) - 2)
        self.assertEqual(session.links, before)
        matcher.assert_not_called()

    def test_approved_overrides_default_and_live_overrides_approved_without_writes(self):
        session = self.session()
        job = session.jobs[0]
        self.approve(session, job, (1, 0))
        before = deepcopy(session.__dict__)
        default = Mock(return_value=Match((0, 1), (), .8))
        links = prepare_links(session, default)
        preview = links.resolve()
        self.assertEqual(default.call_count, len(session.jobs) - 1)
        self.assertEqual(preview.maps[job.right.id], (1, 0))
        self.assertEqual(preview.regions, ((0, 49, 0),))
        self.assertNotIn(job.id, preview.automatic)
        self.assertEqual(len(preview.automatic), len(session.jobs) - 1)
        live = links.resolve(job.id, (0, 1))
        self.assertEqual(live.maps[job.right.id], (0, 1))
        self.assertEqual(session.links, before["links"])
        self.assertEqual(session.revision, before["revision"])
        self.assertEqual(session.undo_stack, before["undo_stack"])
        self.assertEqual(session.dirty, before["dirty"])

    def test_ambiguous_endpoint_stays_separate_until_reviewed(self):
        session = self.session()
        job = session.jobs[0]
        default = lambda boundary: Match((0, 1), ('Tied assignments',) if boundary == job else (), .8)
        links = prepare_links(session, default)
        self.assertEqual(links.resolve().regions, ((0, 9, 0), (10, 49, 1)))
        live = links.resolve(session.jobs[0].id, (1, 0))
        self.assertEqual(live.regions, ((0, 49, 0),))
        self.assertEqual(live.maps["a"], (1, 0))
        self.approve(session, job, (0, 1))
        self.assertEqual(prepare_links(session, default).resolve().regions, ((0, 49, 0),))

    def test_interior_missing_masks_do_not_block_clean_boundary_endpoints(self):
        session = self.session()
        session.missing_forward[12] = True
        session.missing_reverse[16] = True
        preview = prepare_links(session, lambda job: Match((0, 1), (), .8)).resolve()
        self.assertEqual(preview.regions, ((0, 49, 0),))

    def test_invalid_bridge_has_no_default_and_no_colored_invalid_frames(self):
        session = self.session([RevisionZone("bad", 10, 19, "invalid")])
        default = Mock(return_value=Match((0, 1), (), .8))
        links = prepare_links(session, default)
        default.assert_not_called()
        self.assertEqual(links.resolve().regions, ((0, 9, 0), (20, 49, 1)))
        self.assertEqual(links.resolve(session.jobs[0].id, (0, 1)).regions, ((0, 9, 0), (20, 49, 0)))

    def test_manual_draft_is_preserved_and_invalid_approval_is_reported(self):
        session = self.session()
        job = session.jobs[0]
        session.set_draft(job.id, {**initial_draft(job), "mapping": [1, 0], "manual": True})
        default = Mock(return_value=Match((0, 1), (), .8))
        preview = prepare_links(session, default).resolve()
        self.assertNotIn(job.id, preview.automatic)
        self.assertEqual(default.call_count, len(session.jobs) - 1)
        self.assertEqual(len(set(preview.components.values())), 2)
        session.links[job.id].update(status="approved", approved={"canonical_mapping": [0, 0]})
        preview = prepare_links(session, default).resolve()
        self.assertTrue(preview.errors)

    def test_link_quality_uses_margins_of_selected_pairs(self):
        # IoU = [[1/7, 3/5], [3/5, 1/7]], so both selected margins are 16/35.
        left = np.array([[1, 1, 1, 1, 2, 2, 2, 2]])
        right = np.array([[1, 2, 2, 2, 1, 1, 1, 2]])
        match = match_masks(left, right, 2)
        self.assertEqual(match.mapping, (1, 0))
        self.assertEqual(match.warnings, ())
        self.assertAlmostEqual(match.quality, 16 / 35)
        self.assertTrue(match_masks(left, np.zeros_like(right), 2).warnings)
        self.assertTrue(match_masks(left, np.tile([[1, 1, 2, 2]], (1, 2)), 2).warnings)

    def test_source_alignment_and_preview_cache_track_live_mapping(self):
        session = self.session()
        data = object.__new__(RevisionData)
        data.n, data.count = 2, 50
        data.evidence = SimpleNamespace(switch_frames=np.zeros(50, bool))
        data.linking = SimpleNamespace(n=2)
        data.linking.boundary_match = Mock(return_value=Match((0, 1), (), .8))
        data.linking.canonical_mapping = Mock(side_effect=lambda job, draft, seed: (draft["mapping"], []))
        data.linking.conversion = Mock(return_value=((1, 0), []))
        colors = LinkColors(data)
        job = session.jobs[0]
        draft = {**initial_draft(job), "mapping": [1, 0]}
        before = session.fingerprint()
        first = colors.preview(session, job, draft, .6)
        self.assertIs(colors.preview(session, job, draft, .6), first)
        data.linking.boundary_match.assert_not_called()
        self.assertEqual(data.linking.canonical_mapping.call_count, 1)
        labels, component = global_labels(data.linking, first, "reverse", 12, .6)
        self.assertEqual((labels, component), ((0, 1), 0))
        second = colors.preview(session, job, {**draft, "mapping": [0, 1]}, .6)
        self.assertEqual(second.maps["a"], (0, 1))
        data.linking.boundary_match.assert_not_called()
        self.assertEqual(session.fingerprint(), before)

    def test_masks_and_assigned_points_share_global_colors(self):
        session = self.session()
        self.approve(session, session.jobs[0], (1, 0))
        preview = prepare_links(session, lambda job: Match((0, 1), (), .8)).resolve()
        linker = SimpleNamespace(n=2, conversion=Mock(return_value=((0, 1), [])))
        colors = bgr_colors(global_label_colors(linker, preview, 'forward', 12, .6))
        np.testing.assert_array_equal(colors, bgr_colors(component_colors(2, 0))[::-1])
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        points = np.array([[[10., 10.]], [[25., 25.]]])
        assigned = overlay_keypoints(image, points, [0, -1], (40, 40), 'assigned', colors)
        np.testing.assert_array_equal(assigned[10, 10], colors[0])
        self.assertFalse(assigned[25, 25].any())
        both = overlay_keypoints(image, points, [0, 1], (40, 40), 'both', colors)
        np.testing.assert_array_equal(both[10, 10], MASK_COLORS_BGR[0])
        local = overlay_keypoints(image, points, [0, 1], (40, 40), 'both')
        self.assertFalse(np.array_equal(both[10, 13], local[10, 13]))
        np.testing.assert_array_equal(both[25, 25], MASK_COLORS_BGR[1])


if __name__ == "__main__":
    unittest.main()
