"""Validate key-frame coverage and generated segmentation outputs.

Examples
--------
>>> from bivosia.analysis.segmentation import key_frame_intervals, output_shape
>>> key_frame_intervals([0, 5, 10], [0, 10])
[(0, 5), (5, 10)]
>>> output_shape((100, 200), 0.5)
(50, 100)
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import cv2
import numpy as np

from bivosia.analysis.key_frames import validate_mask


def key_frame_intervals(frames, crop):
    frames = sorted(set(f for f in frames if crop[0] <= f <= crop[1]))
    if not frames or frames[0] != crop[0] or frames[-1] != crop[1]:
        raise ValueError('Label and save both crop endpoints before running VOS')
    return list(zip(frames[:-1], frames[1:])) or [(frames[0], frames[0])]


def output_shape(shape, scale):
    if not np.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError('Output scale must be greater than zero and at most one')
    return tuple(max(1, round(size * scale)) for size in shape)


def output_mask(mask, source_shape, shape, animals):
    """Validate IDs and resize integer labels using nearest-neighbor sampling."""
    mask = validate_mask(mask, source_shape, animals)
    return mask if tuple(shape) == tuple(source_shape) else cv2.resize(
        mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST_EXACT)


def validate_coverage(valid, crop):
    valid = np.asarray(valid, dtype=bool)
    first, last = crop
    if not valid[first:last + 1].all() or valid[:first].any() or valid[last + 1:].any():
        raise ValueError('Generated masks do not cover exactly the selected crop')
