"""Contour-area filtering and overlap removal for animal masks.

Examples
--------
>>> import numpy as np
>>> from bivosia.analysis.mask_cleanup import clean_masks
>>> masks = np.zeros((3, 8, 16), dtype=bool)
>>> masks[0, 1:4, 1:4] = masks[1, 1:4, 4:7] = True
>>> masks[2, 1:4, 12:15] = masks[0, 6, 1] = True
>>> cleaned = clean_masks(masks, contour_min_area=4, contour_max_area=9, overlap_kernel=3)
>>> cleaned.sum(axis=(1, 2)).tolist()
[6, 6, 9]
>>> bool(masks[0, 6, 1]), bool(cleaned[0, 6, 1])
(True, False)
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import operator

import numpy as np
import cv2

from bivosia.config import ANALYSIS_DEFAULTS
from bivosia.analysis.key_frames import validate_mask


def clean_masks(masks, *,
                contour_min_area=ANALYSIS_DEFAULTS['mask_cleanup']['contour_min_area'],
                contour_max_area=ANALYSIS_DEFAULTS['mask_cleanup']['contour_max_area'],
                overlap_kernel=ANALYSIS_DEFAULTS['mask_cleanup']['overlap_kernel']):
    """Return cleaned binary masks shaped (animals, height, width).

    Contour areas are measured by OpenCV in squared input-image pixels, with
    both limits inclusive. Each accepted contour is filled independently,
    including contours around holes. Pixels covered by the square dilation
    of two or more accepted masks are cleared from every mask. The dilation
    only determines exclusions; it does not enlarge the returned masks.

    ``overlap_kernel`` is a positive odd side length in input-image pixels.
    A side length of one excludes only actual overlaps. Inputs are unchanged.
    """
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3 or min(masks.shape[1:]) < 1:
        raise ValueError('Masks must have shape (animals, height, width) with nonempty images')
    if not (np.isfinite(contour_min_area) and np.isfinite(contour_max_area)
            and 0 <= contour_min_area <= contour_max_area):
        raise ValueError('Contour area limits must be finite and satisfy 0 <= minimum <= maximum')
    try:
        size = operator.index(overlap_kernel)
    except TypeError:
        raise ValueError('Overlap kernel must be a positive odd integer') from None
    if isinstance(overlap_kernel, (bool, np.bool_)) or size < 1 or size % 2 != 1:
        raise ValueError('Overlap kernel must be a positive odd integer')

    cleaned = np.zeros(masks.shape, dtype=bool)
    covered = np.zeros(masks.shape[1:], dtype=bool)
    excluded = np.zeros_like(covered)
    kernel = np.ones((size, size), dtype=np.uint8)
    for animal, mask in enumerate(masks):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        accepted = np.zeros(mask.shape, dtype=np.uint8)
        for contour in contours:
            if contour_min_area <= cv2.contourArea(contour) <= contour_max_area:
                cv2.drawContours(accepted, [contour], -1, 1, -1)
        cleaned[animal] = accepted != 0
        dilated = cv2.dilate(accepted, kernel) != 0
        excluded |= covered & dilated
        covered |= dilated
    cleaned[:, excluded] = False
    return cleaned


def clean_label_mask(mask, **parameters):
    """Clean a full-resolution label image while preserving animal IDs and input pixels.

    The returned image is a separate array. Background is zero; positive IDs
    are cleaned jointly so overlap exclusions account for every animal.

    >>> import numpy as np
    >>> from bivosia.analysis.mask_cleanup import clean_label_mask
    >>> labels = np.zeros((6, 6), dtype=np.uint8)
    >>> labels[1:4, 1:4], labels[5, 5] = 7, 7
    >>> cleaned = clean_label_mask(labels, contour_min_area=4, contour_max_area=9, overlap_kernel=1)
    >>> np.unique(cleaned).tolist(), int(np.count_nonzero(cleaned))
    ([0, 7], 9)
    >>> int(labels[5, 5])
    7
    """
    mask = np.asarray(mask)
    if mask.ndim != 2 or not all(mask.shape):
        raise ValueError('Expected a nonempty two-dimensional label image')
    mask = validate_mask(mask, mask.shape, 255)
    animals = np.unique(mask[mask > 0])
    regions = clean_masks(mask == animals[:, None, None], **parameters)
    result = np.zeros_like(mask)
    for animal, region in zip(animals, regions):
        result[region] = animal
    return result


def validate_cleaned_key_frames(frames, masks, frame_count, animals):
    """Check source frame indices, image dimensions and cleaned animal labels."""
    frames, masks = np.asarray(frames), np.asarray(masks)
    if (frames.ndim != 1 or not np.issubdtype(frames.dtype, np.integer)
            or frame_count < 1 or np.any(frames < 0) or np.any(frames >= frame_count)
            or np.any(np.diff(frames) <= 0)):
        raise ValueError('Cleaned key frames must have unique increasing indices inside the video')
    if masks.ndim != 3 or masks.shape[0] != len(frames) or not all(masks.shape):
        raise ValueError('Cleaned masks must have shape (key frames, height, width)')
    for mask in masks:
        validate_mask(mask, masks.shape[1:], animals)
