"""Validate, relabel, clean and preprocess mask images.

Examples
--------
>>> import numpy as np
>>> from bivosia.analysis.masks import relabel_mask
>>> labels = np.array([[0, 1], [2, 2]])
>>> relabel_mask(labels, (1, 0)).tolist()
[[0, 2], [1, 1]]
"""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import operator

import numpy as np
import cv2

from bivosia.config import ANALYSIS_DEFAULTS, LINKED_MASK_STREAMS
from bivosia.analysis.keyframes import output_shape, validate_mask


# Mask validation


def validate_mask_data(masks, valid=None):
    """Validate integer sequence dimensions and per-frame validity without reading pixels."""
    if (len(masks.shape) != 3 or not all(masks.shape)
            or not np.issubdtype(masks.dtype, np.integer)):
        raise ValueError('Expected integer masks with shape (frames, height, width)')
    if valid is not None and (valid.shape != (masks.shape[0],) or valid.dtype != np.bool_):
        raise ValueError('Mask validity must be a boolean vector covering every frame')


def validate_linked_mask_data(arrays):
    """Validate linked-export array headers without reading mask pixels or flags."""
    required = (*LINKED_MASK_STREAMS, 'preferred_valid', 'direction_valid', 'source')
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError('Linked masks are missing arrays: ' + ', '.join(missing))
    for name in required:
        item = arrays[name]
        if (not hasattr(item, 'shape') or item.shape is None
                or not hasattr(item, 'dtype')):
            raise ValueError(f'Linked-mask entry {name} must be an array')
    shape = arrays['preferred'].shape
    for name in LINKED_MASK_STREAMS:
        validate_mask_data(arrays[name])
        if arrays[name].shape != shape:
            raise ValueError('Linked-mask streams must have matching dimensions')
    validate_mask_data(arrays['preferred'], arrays['preferred_valid'])
    valid = arrays['direction_valid']
    if valid.shape != (2, shape[0]) or valid.dtype != np.bool_:
        raise ValueError('Direction validity must be boolean with shape (2, frames)')
    source = arrays['source']
    if source.shape != (shape[0],) or not np.issubdtype(source.dtype, np.integer):
        raise ValueError('Linked-mask source must be an integer vector covering every frame')


def _mask_labels(mask, n):
    mask = np.asarray(mask)
    if n < 1 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("Masks require integer labels and a positive animal count")
    if mask.size and (mask.min() < 0 or mask.max() > n):
        raise ValueError(f"Mask labels must be between 0 and {n}")
    return mask.reshape(-1).astype(np.int64, copy=False)


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


# Mask manipulation


def validate_downsample_factor(value):
    """Return a positive integer spatial downsampling factor."""
    try:
        factor = operator.index(value)
    except TypeError:
        raise ValueError('Downsample factor must be a positive integer') from None
    if isinstance(value, (bool, np.bool_)) or factor < 1:
        raise ValueError('Downsample factor must be a positive integer')
    return factor


def downsample_mask(mask, factor):
    """Reduce a boolean mask with nearest-neighbor sampling; never alter its source.

    Factor 1 keeps the original resolution. Dimensions are rounded to the
    nearest integer, with a minimum of one pixel, as for VOS output resizing.

    >>> import numpy as np
    >>> from bivosia.analysis.masks import downsample_mask
    >>> mask = np.zeros((8, 12), dtype=bool)
    >>> mask[2:6, 4:8] = True
    >>> smaller = downsample_mask(mask, 2)
    >>> smaller.shape, int(smaller.sum()), int(mask.sum())
    ((4, 6), 4, 16)
    """
    factor = validate_downsample_factor(factor)
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != np.bool_ or not all(mask.shape):
        raise ValueError('Expected a nonempty two-dimensional boolean image')
    if factor == 1:
        return mask
    shape = output_shape(mask.shape, 1 / factor)
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]),
                      interpolation=cv2.INTER_NEAREST_EXACT) != 0


def relabel_mask(mask, mapping):
    n = len(mapping)
    if np.any(mask < 0) or np.any(mask > n):
        raise ValueError("Raw mask contains a label outside the declared animal count")
    lookup = np.asarray([0, *(int(i) + 1 for i in mapping)], dtype=np.uint16)
    return lookup[mask]


def clean_masks(masks, *,
                contour_min_area=ANALYSIS_DEFAULTS['masks']['cleanup']['contour_min_area'],
                contour_max_area=ANALYSIS_DEFAULTS['masks']['cleanup']['contour_max_area'],
                overlap_kernel=ANALYSIS_DEFAULTS['masks']['cleanup']['overlap_kernel']):
    """Return cleaned binary masks shaped (animals, height, width).

    Contour areas are measured by OpenCV in squared input-image pixels, with
    both limits inclusive. Each accepted contour is filled independently,
    including contours around holes. Pixels covered by the square dilation
    of two or more accepted masks are cleared from every mask. The dilation
    only determines exclusions; it does not enlarge the returned masks.

    ``overlap_kernel`` is a positive odd side length in input-image pixels.
    A side length of one excludes only actual overlaps. Inputs are unchanged.

    Examples
    --------
    >>> import numpy as np
    >>> from bivosia.analysis.masks import clean_masks
    >>> masks = np.zeros((3, 8, 16), dtype=bool)
    >>> masks[0, 1:4, 1:4] = masks[1, 1:4, 4:7] = True
    >>> masks[2, 1:4, 12:15] = masks[0, 6, 1] = True
    >>> cleaned = clean_masks(masks, contour_min_area=4, contour_max_area=9, overlap_kernel=3)
    >>> cleaned.sum(axis=(1, 2)).tolist()
    [6, 6, 9]
    >>> bool(masks[0, 6, 1]), bool(cleaned[0, 6, 1])
    (True, False)
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
    >>> from bivosia.analysis.masks import clean_label_mask
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


def preprocessing_settings(enabled, **parameters):
    """Return validated settings, or None when preprocessing is disabled."""
    if not enabled:
        return None
    settings = dict(ANALYSIS_DEFAULTS['masks']['preprocessing'])
    if parameters.keys() - settings.keys():
        raise ValueError('Unknown mask preprocessing parameter')
    settings.update(parameters)
    for name in ('filter_by_area', 'filter_by_distance', 'apply_opening'):
        if not isinstance(settings[name], (bool, np.bool_)):
            raise ValueError(f'{name} must be boolean')
        settings[name] = bool(settings[name])
    for name in ('main_body_area', 'fragment_distance'):
        if not np.isfinite(settings[name]) or settings[name] < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    try:
        radius = operator.index(settings['opening_radius'])
    except TypeError:
        raise ValueError('Opening radius must be a nonnegative integer') from None
    if isinstance(settings['opening_radius'], (bool, np.bool_)) or radius < 0:
        raise ValueError('Opening radius must be a nonnegative integer')
    settings['opening_radius'] = radius
    return settings


def preprocess_mask(mask, *,
                    filter_by_area=ANALYSIS_DEFAULTS['masks']['preprocessing']['filter_by_area'],
                    main_body_area=ANALYSIS_DEFAULTS['masks']['preprocessing']['main_body_area'],
                    filter_by_distance=ANALYSIS_DEFAULTS['masks']['preprocessing']['filter_by_distance'],
                    fragment_distance=ANALYSIS_DEFAULTS['masks']['preprocessing']['fragment_distance'],
                    apply_opening=ANALYSIS_DEFAULTS['masks']['preprocessing']['apply_opening'],
                    opening_radius=ANALYSIS_DEFAULTS['masks']['preprocessing']['opening_radius']):
    """Return a separate body mask at the supplied image resolution.

    Area filtering retains only the largest 8-connected component when its
    foreground pixel count exceeds ``main_body_area``. Distance filtering
    retains the two largest remaining components when their centroid distance
    is at most ``fragment_distance``; otherwise it retains only the largest.
    Equal areas retain component scan order. Each disabled filter leaves all
    current components intact. Opening applies erosion followed by dilation
    with a disk of ``opening_radius`` pixels (zero skips opening). Pixels
    outside the image are background. Each step can be disabled independently.
    Holes remain unless altered by opening; source pixels are never modified.

    Examples
    --------
    >>> import numpy as np
    >>> from bivosia.analysis.masks import preprocess_mask
    >>> mask = np.zeros((12, 24), dtype=bool)
    >>> mask[2:6, 2:6] = mask[2:5, 8:11] = mask[9, 21] = True
    >>> cleaned = preprocess_mask(mask, main_body_area=16, fragment_distance=8, opening_radius=0)
    >>> int(cleaned.sum()), int(mask.sum())
    (25, 26)
    >>> int(preprocess_mask(mask, main_body_area=15, fragment_distance=8, opening_radius=0).sum())
    16
    >>> int(preprocess_mask(mask, main_body_area=16, fragment_distance=2, opening_radius=0).sum())
    16
    """
    settings = preprocessing_settings(True, filter_by_area=filter_by_area, main_body_area=main_body_area,
                                      filter_by_distance=filter_by_distance, fragment_distance=fragment_distance,
                                      apply_opening=apply_opening, opening_radius=opening_radius)
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != np.bool_ or not all(mask.shape):
        raise ValueError('Expected a nonempty two-dimensional boolean image')
    result = mask.copy()
    if settings['filter_by_area'] or settings['filter_by_distance']:
        count, labels, statistics, centers = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8)
        if count == 1:
            return result
        areas = statistics[1:, cv2.CC_STAT_AREA]
        keep = np.argsort(-areas, kind='stable') + 1
        if settings['filter_by_area'] and areas[keep[0] - 1] > settings['main_body_area']:
            keep = keep[:1]
        if settings['filter_by_distance'] and len(keep) > 1:
            nearby = np.linalg.norm(centers[keep[0]] - centers[keep[1]]) <= settings['fragment_distance']
            keep = keep[:2] if nearby else keep[:1]
        result = np.isin(labels, keep)
    radius = settings['opening_radius']
    if settings['apply_opening'] and radius:
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        disk = (x * x + y * y <= radius * radius).astype(np.uint8)
        result = cv2.morphologyEx(result.astype(np.uint8), cv2.MORPH_OPEN, disk,
                                  borderType=cv2.BORDER_CONSTANT, borderValue=0) != 0
    return result
