"""Read, write and locate reusable framewise analysis arrays."""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'


import numpy as np

from bivosia.analysis.masks import iou_from_intersections
from bivosia.config import ANALYSIS_DEFAULTS
from bivosia.dataset.paths import file_name
from bivosia.dataset.formats import format_handler


MASK_IOU_MATCH = file_name("mask_iou_match")
MASK_QUALITY = file_name("mask_quality")
MASK_MISSING = file_name("mask_missing")
MASK_SWITCH = file_name("mask_switch")
MASK_QUALITY_METADATA = file_name("mask_quality_metadata")

QUALITY_PRODUCTS = (MASK_QUALITY, MASK_MISSING, MASK_SWITCH, MASK_QUALITY_METADATA)


def quality_cache_available(resolve) -> bool:
    """Check quality products using an analysis-file resolver."""
    return all(resolve(filename).is_file() for filename in QUALITY_PRODUCTS)


def read_array(path):
    """Open a read-only, memory-mapped NumPy array."""
    return format_handler('arrays', path)(path)


class MaskOverlaps:
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
            count_dtype = np.dtype(ANALYSIS_DEFAULTS['masks']['count_dtype'])
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


def array_output(path, shape, dtype, *, fill=None):
    """Create an array through its registered writer, retaining failed partials."""
    return format_handler('arrays', path, operation='writer')(path, shape, dtype, fill=fill)


def write_array(path, values):
    with array_output(path, values.shape, values.dtype) as output:
        output[:] = values
