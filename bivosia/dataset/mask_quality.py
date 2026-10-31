"""Load mask-overlap counts and persisted framewise quality evidence."""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import closing
from pathlib import Path

import numpy as np

from bivosia.config import ANALYSIS_DEFAULTS
from bivosia.analysis.mask_features import missing_mask_counts
from bivosia.analysis.mask_quality import iou_from_intersections, permutation_change_measures
from bivosia.analysis.zones_of_disagreement import ZodEvidence
from bivosia.dataset.io.arrays import read_array
from bivosia.dataset.paths import file_name


MASK_IOU_MATCH = file_name("mask_iou_match")
MASK_QUALITY = file_name("mask_quality")
MASK_MISSING = file_name("mask_missing")
MASK_SWITCH = file_name("mask_switch")
MASK_QUALITY_METADATA = file_name("mask_quality_metadata")

QUALITY_PRODUCTS = (MASK_QUALITY, MASK_MISSING, MASK_SWITCH, MASK_QUALITY_METADATA)


def quality_cache_available(resolve) -> bool:
    """Check quality products using an analysis-file resolver."""
    return all(resolve(filename).is_file() for filename in QUALITY_PRODUCTS)


class MaskOverlapReader:
    """Memory-mapped counts with IoU derived for a requested frame or block."""

    def __init__(self, intersections, areas):
        self.intersections = read_array(intersections)
        self.areas = None
        try:
            self.areas = read_array(areas)
            shape = self.intersections.shape
            if len(shape) != 3 or shape[0] < 1 or shape[1] < 1 or shape[1] != shape[2]:
                raise ValueError("Expected intersections with shape (frames, animals, animals)")
            if self.areas.shape != (2, shape[0], shape[1]):
                raise ValueError("Area and intersection dimensions differ")
            count_dtype = np.dtype(ANALYSIS_DEFAULTS['mask_quality']['count_dtype'])
            if self.intersections.dtype != count_dtype or self.areas.dtype != count_dtype:
                raise ValueError(f"Intersections and areas must contain {count_dtype} pixel counts")
        except BaseException:
            self.close()
            raise

    @property
    def shape(self):
        return self.intersections.shape

    def iou(self, frames=slice(None)):
        return iou_from_intersections(self.intersections[frames],
                                      self.areas[0, frames], self.areas[1, frames])

    def close(self):
        for array in (self.intersections, self.areas):
            if array is not None:
                array._mmap.close()


def load_mask_summary(dataset, *, frame_count=None, with_iou=False):
    """Read frame/animal counts, missing masks for each pass and optional IoU matrices."""
    with closing(MaskOverlapReader(dataset.array('mask_intersections'), dataset.array('mask_areas'))) as overlaps:
        count, animals, _ = overlaps.shape
        if frame_count is not None and count != frame_count:
            raise ValueError('Mask analysis and recording frame counts differ')
        missing_forward, missing_reverse = missing_mask_counts(overlaps.areas)
        matrices = overlaps.iou() if with_iou else None
    return count, animals, missing_forward, missing_reverse, matrices


def read_quality_array(dataset, name, shape):
    """Read an optional saved quality array, checking its expected dimensions."""
    path = dataset.array(name)
    if not path.is_file():
        return None
    values = read_array(path)
    if values.shape != shape:
        values._mmap.close()
        raise ValueError(f'{name} and recording dimensions differ')
    return values


def load_quality_evidence(resolve, iou_match_path: str | Path, *, shape=None):
    """Load persisted arrays as a ``ZodEvidence`` without recomputing quality."""
    arrays = []
    try:
        paths = [Path(iou_match_path).expanduser().resolve(),
                 *(resolve(name) for name in (MASK_QUALITY, MASK_MISSING, MASK_SWITCH))]
        for path in paths:
            arrays.append(read_array(path))
        assignment, quality, missing, switches = arrays
        if shape is not None and (assignment.ndim != len(shape) or any(
                expected is not None and actual != expected
                for actual, expected in zip(assignment.shape, shape))):
            raise ValueError('Cached assignments and recording dimensions differ')
        if quality.shape != (assignment.shape[0],) or missing.shape != quality.shape or switches.shape != quality.shape:
            raise ValueError("Persisted quality arrays have incompatible frame counts")
        changes, expected_switches = permutation_change_measures(assignment, missing)
        if not np.array_equal(switches, expected_switches):
            raise ValueError("Persisted mask_switch array does not match mask_iou_match and mask_missing")
        return ZodEvidence(quality, missing, assignment, changes, switches)
    except BaseException:
        for array in arrays:
            array._mmap.close()
        raise
