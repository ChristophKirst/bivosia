"""Focused checks for partial matching, ties, fallback and invalid frames."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest

import h5py
import numpy as np

from bivosia.analysis.alignment import (
    align_scores, align_points_to_masks, point_mask_scores, point_mask_overlaps, point_overlap_scores,
)
from bivosia.analysis.keypoints import keypoint_identity_events
from bivosia.analysis.zones_of_disagreement import RevisionZone
from bivosia.dataset.keypoints import load_keypoints
from bivosia.dataset.frame_data import PointData
from bivosia.dataset.layout import Dataset
from bivosia.dataset.mask_output import write_masks
from bivosia.processing.alignment import calculate_keypoint_alignment


class ArrayPoints(PointData):
    """A serializable non-SLEAP reader used to exercise generic processing."""

    def __init__(self):
        super().__init__((1, 5))
        self.points = np.tile(np.array([[[[0., 0.]], [[4., 0.]]]]), (2, 1, 1, 1))
        self.shape, self.dtype = self.points.shape, self.points.dtype
        self.valid = np.ones(2, dtype=bool)
        self.node_names, self.track_names = ['point'], ['a', 'b']

    def _read(self, frames):
        return self.points[frames]


class AlignmentTests(unittest.TestCase):
    def test_processing_accepts_a_non_sleap_point_reader(self):
        with TemporaryDirectory() as directory:
            dataset = Dataset(directory)
            dataset.annotations.mkdir()
            (dataset.annotations / 'hand.json').write_text(json.dumps({'zones': []}))
            labels = np.tile(np.array([[[1, 1, 0, 2, 2]]], np.uint8), (2, 1, 1))
            for direction in ('forward', 'reverse'):
                write_masks(dataset.masks / f'masks_{direction}.npy', labels)
            result = calculate_keypoint_alignment(dataset, 2, points=ArrayPoints(),
                                                   workers=2, block_size=1, dilation_kernel=1)
            self.assertEqual(result.assignment.tolist(), [[0, 1], [0, 1]])
            self.assertFalse(dataset.keypoint_file().exists())

    def test_recording_alignment_excludes_named_keypoints(self):
        with TemporaryDirectory() as directory:
            dataset = Dataset(directory)
            dataset.settings['keypoints'] = {'image_shape': [1, 5]}
            dataset.save()
            dataset.annotations.mkdir()
            (dataset.annotations / 'hand.json').write_text(json.dumps({'zones': []}))
            dataset.masks.mkdir()
            for direction in ('forward', 'reverse'):
                with h5py.File(dataset.mask(direction), 'w') as file:
                    file['masks'] = np.array([[[1, 1, 0, 2, 2]]], dtype=np.uint8)
            dataset.keypoints.mkdir()
            points = np.array([[[[0., 0.], [4., 0.], [4., 0.]],
                                [[4., 0.], [0., 0.], [0., 0.]]]])
            with h5py.File(dataset.keypoint_file(), 'w') as file:
                file['tracks'] = points.transpose(1, 3, 2, 0)
                file['node_names'] = ['nose', 'tailBase', 'tailEnd']
                file['track_names'] = ['a', 'b']
            for excluded, expected, scores in (
                ((), [[1, 0]], [[[1 / 3, 4 / 3], [4 / 3, 1 / 3]]]),
                (('tailBase', 'tailEnd'), [[0, 1]], [[[1, 0], [0, 1]]]),
            ):
                result = calculate_keypoint_alignment(dataset, 2, exclude_keypoints=excluded,
                                                       dilation_kernel=1)
                self.assertEqual(result.assignment.tolist(), expected)
                loaded = load_keypoints(dataset)
                self.assertEqual(loaded.excluded_keypoints, excluded)
                self.assertEqual(loaded.selected_node_ids, [0] if excluded else [0, 1, 2])
                np.testing.assert_allclose(loaded.scores, scores)
                np.testing.assert_equal(loaded[:], points)
                path = dataset.array('keypoint_alignment')
                with h5py.File(path, 'r') as file:
                    self.assertEqual(json.loads(file.attrs['excluded_keypoints']), list(excluded))
                    self.assertEqual(json.loads(file.attrs['node_names']), ['nose', 'tailBase', 'tailEnd'])
                    self.assertEqual(file['overlaps'].shape, (1, 2, 3, 2))
                    # Excluded tails remain available to reproduce the all-node scores.
                    np.testing.assert_allclose(point_overlap_scores(file['overlaps'][:], file['point_usable'][:]),
                                               [[[1 / 3, 4 / 3], [4 / 3, 1 / 3]]])
                path.unlink()
            with self.assertRaisesRegex(ValueError, 'Unknown keypoint to exclude'):
                calculate_keypoint_alignment(dataset, 2, exclude_keypoints=['typo'])
            self.assertFalse(dataset.array('keypoint_alignment').exists())

    def test_keypoint_events_follow_linked_masks_and_resolved_frames(self):
        scores = np.array([
            [[2, 0], [0, 2], [0, 0]],
            [[0, 2], [2, 0], [0, 0]],
            [[2, 0], [0, 2], [0, 0]],
            [[1, 1], [1, 1], [1, 1]],
            [[0, 2], [2, 0], [0, 0]],
            [[0, 2], [2, 0], [0, 0]],
            [[2, 0], [0, 0], [0, 0]],
            [[10, 1], [1, 0], [0, 0]],
            [[0, 0], [0, 0], [0, 0]],
            [[2, 0], [0, 0], [0, 0]],
            [[8, 0], [0, 8], [3, 0]],
            [[8, 0], [0, 0], [0, 8]],
            [[0, 8], [0, 0], [8, 0]],
        ], dtype=float)
        valid = np.ones(13, dtype=bool)
        valid[8] = False
        present = np.ones((13, 2), dtype=bool)
        present[8] = False
        present[9, 1] = False
        links = SimpleNamespace(
            partition=(RevisionZone('a', 0, 0, 'forward'), RevisionZone('b', 1, 4, 'reverse'),
                       RevisionZone('c', 5, 7, 'reverse'), RevisionZone('bad', 8, 8, 'invalid'),
                       RevisionZone('d', 9, 12, 'reverse')),
            maps={'a': (0, 1), 'b': (1, 0), 'c': (0, 1), 'd': (0, 1)},
            components={'a': 0, 'b': 0, 'c': 1, 'd': 2}, errors=())
        counts = np.full((13, 3), 11)
        counts[:, 2] = 0  # An extra empty track does not make a skeleton missing.
        counts[3, 2] = 11
        counts[6, 1] = 0
        counts[8] = 0  # Invalid frames remain unmarked.
        counts[10:, 2] = 11  # A third skeleton competes, then replaces a disappearing track.
        counts[11:, 1] = 0
        events = keypoint_identity_events(align_scores(scores, valid),
            np.array([0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]), scores, present, links, point_counts=counts)
        expected = {'swaps': [2, 4, 12], 'swap frames': [1, 2, 3, 4, 11, 12],
                    'no overlap': [9], 'ties': [], 'unassigned': [7, 10],
                    'ties + unassigned': [3], 'missing skeleton': [6]}
        self.assertEqual({name: np.flatnonzero(events[name]).tolist() for name in expected}, expected)
        self.assertEqual(events['swap zones'].tolist(), [[1, 2], [2, 4], [11, 12]])
        self.assertEqual(events['swaps over invalid zone'].tolist(), [False, False, False])

    def test_swap_zones_bridge_unresolved_and_linked_invalid_frames(self):
        states = [-1, 0, -1, 1, 0, -2, 1, -1, 1, -3, 0, -1]
        choices = {0: [[2, 0], [0, 2]], 1: [[0, 2], [2, 0]],
                   -1: [[1, 1], [1, 1]], -2: [[0, 0], [0, 0]],
                   -3: [[2, 0], [0, 0]]}
        scores = np.array([choices[state] for state in states], dtype=float)
        valid = np.array(states) != -2
        present = np.repeat(valid[:, None], 2, axis=1)
        links = SimpleNamespace(
            partition=(RevisionZone('a', 0, 4), RevisionZone('bad', 5, 5, 'invalid'),
                       RevisionZone('b', 6, 11)),
            maps={'a': (0, 1), 'b': (0, 1)}, components={'a': 0, 'b': 0}, errors=())
        counts = np.ones((12, 2), dtype=int)
        counts[9, 1] = 0
        events = keypoint_identity_events(align_scores(scores, valid),
            np.where(valid, 0, 2), scores, present, links, point_counts=counts)
        self.assertEqual(events['swap zones'].tolist(), [[1, 3], [3, 4], [4, 6], [8, 10]])
        self.assertEqual(np.flatnonzero(events['swaps']).tolist(), [3, 4, 6, 10])
        self.assertEqual(events['swaps over invalid zone'].tolist(), [False, False, True, False])
        self.assertEqual(np.flatnonzero(events['swap frames']).tolist(), [1, 2, 3, 4, 6, 8, 9, 10])

    def test_partial_swaps_follow_each_mask_across_gaps_and_keep_full_swaps(self):
        scores = np.array([
            [[2, 0], [0, 2]], [[2, 0], [8, 0]], [[0, 0], [0, 0]],
            [[8, 0], [2, 0]], [[0, 2], [2, 0]], [[0, 0], [0, 0]],
            [[8, 0], [2, 0]], [[1, 0], [1, 0]], [[2, 0], [8, 0]],
            [[2, 0], [0, 2]], [[0, 2], [2, 0]],
        ], dtype=float)
        valid = np.ones(len(scores), bool)
        valid[5] = False
        links = SimpleNamespace(
            partition=(RevisionZone('a', 0, 4), RevisionZone('bad', 5, 5, 'invalid'),
                       RevisionZone('b', 6, 10)),
            maps={'a': (0, 1), 'b': (0, 1)}, components={'a': 0, 'b': 0}, errors=())
        events = keypoint_identity_events(align_scores(scores, valid), np.where(valid, 0, 2),
            scores, np.repeat(valid[:, None], 2, axis=1), links,
            point_counts=np.ones((len(scores), 2), int))
        self.assertEqual(events['swap zones'].tolist(), [[0, 4], [4, 9], [9, 10]])
        self.assertEqual(events['partial resolved swap zones'].tolist(),
                         [[0, 1], [1, 3], [3, 4], [4, 6], [6, 8], [8, 9]])
        self.assertEqual(events['partial swaps over invalid zone'].tolist(),
                         [False, False, False, True, False, False])
        self.assertEqual(np.flatnonzero(events['partial swap frames']).tolist(),
                         [0, 1, 2, 3, 4, 6, 7, 8, 9])

    def test_partial_swaps_exclude_tied_fallback_and_count_changed_masks_once(self):
        scores = np.array([
            [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            [[1, 1, 0], [1, 1, 0], [0, 0, 2]],
            [[0, 2, 0], [2, 0, 0], [0, 0, 2]],
            [[0, 0, 2], [1, 1, 0], [1, 1, 0]],
            [[1, 1, 0], [0, 0, 2], [1, 1, 0]],
            [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            [[0, 2, 0], [2, 0, 0], [0, 0, 0]],
        ], dtype=float)
        valid = np.ones(len(scores), bool)
        links = SimpleNamespace(partition=(RevisionZone('a', 0, 4), RevisionZone('b', 5, 6)),
            maps={'a': (0, 1, 2), 'b': (0, 1, 2)}, components={'a': 0, 'b': 1}, errors=())
        result = align_scores(scores, valid, fallback_previous=True)
        before = result.assignment.copy()
        events = keypoint_identity_events(result, np.zeros(len(scores), int), scores,
            np.ones((len(scores), 3), bool), links, point_counts=np.ones((len(scores), 3), int))
        self.assertEqual(events['swap zones'].tolist(), [[0, 2]])
        self.assertEqual(events['partial resolved swap zones'].tolist(), [[2, 3], [3, 4], [5, 6]])
        np.testing.assert_array_equal(result.assignment, before)

    def test_rectangular_matching_maximizes_total_score(self):
        for scores, expected in (
            ([[10, 1], [1, 0]], [0, -1]),
            ([[10, 9], [9, 0]], [1, 0]),
            ([[4, 0], [0, 0], [0, 3]], [0, -1, 1]),
            ([[0, 1, 0], [0, 0, 3]], [1, 2]),
            (np.zeros((3, 0)), [-1, -1, -1]),
            (np.zeros((0, 2)), []),
        ):
            result = align_scores(np.array(scores)[None], [True])
            self.assertEqual(result.assignment[0].tolist(), expected)
            self.assertFalse(result.tied[0])

    def test_ties_include_unmatched_competitors_and_keep_unique_pairs(self):
        scores = np.array([[[1, 0], [1, 0], [1, 0], [0, 1]]])
        result = align_scores(scores, [True])
        self.assertEqual(result.assignment.tolist(), [[-1, -1, -1, 1]])
        self.assertTrue(result.tied[0])
        # Equal row preferences can still have a unique global assignment.
        unique = align_scores([[[1, 1], [1, 0]]], [True])
        self.assertEqual(unique.assignment.tolist(), [[1, 0]])
        self.assertFalse(unique.tied[0])
        # Equal total scores remain tied even when the match counts differ.
        tied = align_scores([[[2, 1], [1, 0]]], [True])
        self.assertEqual(tied.assignment.tolist(), [[-1, -1]])
        self.assertTrue(tied.tied[0])

    def test_forward_backward_fallback_and_invalid_gaps(self):
        scores = np.array([
            [[1, 1], [1, 1]],
            [[np.nan, np.nan], [np.nan, np.nan]],
            [[1, 0], [0, 1]],
            [[1, 1], [1, 1]],
            [[1, 0], [1, 0]],
        ])
        valid = [True, False, True, True, True]
        before = scores.copy()
        result = align_scores(scores, valid, fallback_previous=True)
        self.assertEqual(result.assignment.tolist(),
                         [[0, 1], [-1, -1], [0, 1], [0, 1], [0, -1]])
        self.assertEqual(result.tied.tolist(), [True, False, False, True, True])
        self.assertEqual(result.valid.tolist(), valid)
        np.testing.assert_equal(scores, before)
        all_tied = align_scores(np.ones((3, 2, 2)), [True] * 3, fallback_previous=True)
        self.assertTrue(all_tied.tied.all())
        self.assertTrue((all_tied.assignment == -1).all())

    def test_fallback_keeps_only_pairs_compatible_with_current_optimum(self):
        scores = np.array([[[1, 0], [0, 1], [0, 0]],
                           [[1, 2], [0, 0], [1, 2]]])
        result = align_scores(scores, [True, True], fallback_previous=True)
        self.assertEqual(result.assignment[1].tolist(), [0, -1, -1])
        # A previous positive pair can be strictly worse than every optimum.
        scores = np.array([[[1, 0, 0], [0, 1, 0]],
                           [[1, 2, 2], [2, 0, 0]]])
        result = align_scores(scores, [True, True], fallback_previous=True)
        self.assertEqual(result.assignment[1].tolist(), [-1, 0])
        self.assertTrue(result.tied[1])

    def test_scoring_and_excluded_frames(self):
        mask = np.array([[1, 1, 0, 2, 2]], dtype=np.uint8)
        points = np.array([[[0, 0], [1, 0], [2, 0], [np.nan, 0], [99, 0]],
                           [[3, 0], [4, 0], [np.nan, np.nan], [0, np.inf], [-1, 0]]])
        np.testing.assert_allclose(point_mask_scores(points, mask, 2, dilation_kernel=1),
                                   [[4 / 3, 0], [0, 2]])
        np.testing.assert_equal(point_mask_scores([[2, 0]], mask, 2, dilation_kernel=3),
                                [[1, 1]])
        overlaps, usable = point_mask_overlaps(points, mask, 2, dilation_kernel=1)
        np.testing.assert_equal(usable, [[True, True, True, False, False],
                                         [True, True, False, False, False]])
        np.testing.assert_allclose(point_overlap_scores(overlaps, usable), [[4 / 3, 0], [0, 2]])
        overlaps, usable = point_mask_overlaps([[2, 0]], mask, 2, dilation_kernel=3)
        self.assertEqual(overlaps.tolist(), [[[True, True]]])
        masks = np.repeat(mask[None], 3, axis=0)
        masks[1:] = 99  # Invalid frames must never be scored or label-validated.
        sequence = np.repeat(points[None], 3, axis=0)
        result = align_points_to_masks(sequence, masks, 2, mask_valid=[True, False, True],
                                      invalid_point_frames=[2], dilation_kernel=1)
        self.assertEqual(result.assignment.tolist(), [[0, 1], [-1, -1], [-1, -1]])
        self.assertEqual(result.valid.tolist(), [True, False, False])
        self.assertFalse(result.tied.any())

    def test_nearest_integer_rounding_and_image_edges(self):
        mask = np.array([[1, 2, 0]], dtype=np.uint8)
        points = np.array([[0.6, 0], [0.4, 0], [2.6, 0], [np.nan, 0]])
        np.testing.assert_equal(point_mask_scores(points, mask, 2, dilation_kernel=1),
                                [[0, 1], [1, 0], [0, 0], [0, 0]])


if __name__ == '__main__':
    unittest.main()
