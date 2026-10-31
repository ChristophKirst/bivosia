"""Exact overlap counts, block transitions and derived recording evidence."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import h5py
import numpy as np

from bivosia.analysis.zones_of_disagreement import prepare_zod_evidence
from bivosia.analysis.mask_quality import frame_quality_measures, iou_from_intersections, optimal_assignments
from bivosia.dataset.mask_quality import load_quality_evidence, MaskOverlapReader, quality_cache_available
from bivosia.dataset.io.arrays import read_array
from bivosia.dataset.layout import Dataset
from bivosia.processing.analysis.mask_quality import calculate_mask_overlaps, calculate_quality_measures, quantify_many


def write_masks(dataset, forward, reverse):
    dataset.masks.mkdir(parents=True)
    for name, values in (("forward", forward), ("reverse", reverse)):
        with h5py.File(dataset.mask(name), "w") as file:
            file.create_dataset("masks", data=values)


class MaskQuantificationTests(unittest.TestCase):
    def test_counts_temporal_boundaries_and_quality(self):
        with TemporaryDirectory() as temporary:
            for n, frames in ((2, 9), (6, 1)):
                dataset = Dataset(Path(temporary) / str(n))
                base = (np.arange(64).reshape(8, 8) % (n + 1)).astype(np.uint16)
                forward = np.repeat(base[None], frames, axis=0)
                reverse = forward.copy()
                if frames > 1:
                    reverse[4:] = np.where(reverse[4:] > 0, 3 - reverse[4:], 0)
                    forward[6] = 0
                    reverse[7] = 0
                    forward[8] = reverse[8] = 0
                write_masks(dataset, forward, reverse)
                calculate_mask_overlaps(dataset.mask("forward"), dataset.mask("reverse"),
                    dataset.array("mask_intersections"), n, workers=2, batch_size=3,
                    temporal_intersections=True)
                areas = np.array([[[np.count_nonzero(frame == i) for i in range(1, n + 1)]
                                   for frame in masks] for masks in (forward, reverse)], dtype=np.int64)

                def intersections(first, second):
                    return np.array([[np.count_nonzero((first == i) & (second == j))
                                     for j in range(1, n + 1)] for i in range(1, n + 1)], dtype=np.int64)

                expected = np.array([intersections(f, r) for f, r in zip(forward, reverse)])
                temporal = np.empty((2, frames - 1, n, n), dtype=np.int64)
                for direction, masks in enumerate((forward, reverse)):
                    for t in range(frames - 1):
                        temporal[direction, t] = intersections(masks[t], masks[t + 1])
                np.testing.assert_array_equal(read_array(dataset.array("mask_areas")), areas)
                np.testing.assert_array_equal(read_array(dataset.array("mask_temporal_intersections")), temporal)
                quality = read_array(dataset.array('mask_temporal_quality'))
                self.assertEqual(quality.shape, (2, frames - 1))
                for direction in range(2):
                    matrices = iou_from_intersections(temporal[direction], areas[direction, :-1], areas[direction, 1:])
                    expected_quality = frame_quality_measures(matrices, optimal_assignments(matrices))[0]
                    np.testing.assert_array_equal(quality[direction], expected_quality)
                for name in ("mask_areas", "mask_intersections", "mask_temporal_intersections"):
                    self.assertEqual(read_array(dataset.array(name)).dtype, np.dtype("uint32"))
                with closing(MaskOverlapReader(dataset.array("mask_intersections"), dataset.array("mask_areas"))) as counts:
                    self.assertIsInstance(counts.intersections, np.memmap)
                    np.testing.assert_array_equal(counts.intersections, expected)
                    expected_iou = iou_from_intersections(expected, *areas)
                    np.testing.assert_array_equal(counts.iou(slice(0, 1)), expected_iou[:1])
                paths = calculate_quality_measures(dataset.array("mask_intersections"),
                    dataset.array("mask_areas"), dataset.analysis, batch_size=4)
                evidence = prepare_zod_evidence(expected_iou)
                for name, values in (("mask_iou_match", evidence.assignment), ("mask_quality", evidence.quality),
                                     ("mask_missing", evidence.missing), ("mask_switch", evidence.switch_frames)):
                    self.assertEqual(paths[name].name, name + '.npy')
                    np.testing.assert_array_equal(read_array(paths[name]), values)
                self.assertEqual(paths['metadata'].name, 'mask_quality_metadata.json')
                self.assertTrue(quality_cache_available(dataset.analysis_file))
                loaded = load_quality_evidence(dataset.analysis_file, paths['mask_iou_match'])
                np.testing.assert_array_equal(loaded.assignment, evidence.assignment)
                np.testing.assert_array_equal(loaded.switch_frames, evidence.switch_frames)
                if frames > 1:
                    self.assertEqual(np.flatnonzero(loaded.switch_frames).tolist(), [3, 4])
                self.assertFalse((dataset.analysis / "iou_matrices.npy").exists())

    def test_collection_without_temporal_counts(self):
        with TemporaryDirectory() as temporary:
            datasets = [Dataset(Path(temporary) / str(i)) for i in range(2)]
            masks = np.array([[[0, 1, 2], [1, 1, 2]]], dtype=np.uint16)
            masks = np.repeat(masks, 3, axis=0)
            for dataset in datasets:
                write_masks(dataset, masks, masks)
            results = quantify_many(datasets, 2, workers=2)
            self.assertEqual(len(results), 2)
            for dataset in datasets:
                self.assertTrue(dataset.array("mask_intersections").is_file())
                self.assertFalse(dataset.array("mask_temporal_intersections").exists())
                np.testing.assert_array_equal(read_array(dataset.array('mask_temporal_quality')), np.ones((2, 2)))
                self.assertFalse(list(dataset.analysis.glob("*.partial")))

    def test_uint32_limits(self):
        intersection = np.array([[2_000_000_000]], dtype=np.uint32)
        area = np.array([3_000_000_000], dtype=np.uint32)
        np.testing.assert_array_equal(iou_from_intersections(intersection, area, area), [[0.5]])
        with patch("bivosia.processing.analysis.mask_quality.mask_shape", return_value=(1, 65536, 65536)):
            with self.assertRaisesRegex(ValueError, "uint32 capacity"):
                calculate_mask_overlaps("forward.h5", "reverse.h5", "counts.npy", 2)


if __name__ == "__main__":
    unittest.main()
