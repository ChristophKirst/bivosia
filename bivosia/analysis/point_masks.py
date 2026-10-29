"""Containment scores and alignment of points or keypoint sets to labeled masks.

Examples
--------
>>> import numpy as np
>>> from bivosia.analysis.point_masks import align_points_to_masks, point_mask_scores
>>> mask = np.array([[1, 1, 0, 2, 2]], dtype=np.uint8)
>>> points = np.array([[0., 0.], [4., 0.], [2., 0.]])
>>> point_mask_scores(points, mask, 2, dilation_kernel=1).tolist()
[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
>>> result = align_points_to_masks(points[None], mask[None], 2,
...                                mask_valid=[True], dilation_kernel=1)
>>> result.assignment.tolist()
[[0, 1, -1]]
"""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import numpy as np
import cv2

from bivosia.config import ANALYSIS_DEFAULTS
from bivosia.analysis.alignment import align_scores, alignment_valid_frames
from bivosia.analysis.masks import mask_areas


def point_mask_overlaps(points, mask, n_masks, *,
                        dilation_kernel=ANALYSIS_DEFAULTS['point_masks']['dilation_kernel']):
    """Return per-node containment and finite, in-image point flags.

    Points have shape (tracks, keypoints, 2), or (tracks, 2) for single
    points, with coordinates ordered x,y. Mask labels are 0 for background
    and 1..n_masks for objects; overlap column c refers to label c+1.
    The boolean outputs have shapes (tracks, nodes, masks) and (tracks, nodes).
    Usable points have both
    coordinates finite and inside the image after rounding points
    to the nearest integer (half-integers round to the nearest even integer).
    Each mask is dilated independently, so one point can overlap several masks.
    Missing or out-of-image points have false overlap and usability flags.

    The dilation kernel is a positive odd square width in mask pixels.
    The default 5 gives a two-pixel tolerance. Supplied points must already
    use mask pixels; readers apply the dataset's coordinate annotation.
    Coordinates and input masks are not changed.
    """
    points = np.asarray(points, dtype=np.float64)
    mask = np.asarray(mask)
    if points.ndim == 2 and points.shape[-1] == 2:
        points = points[:, None, :]
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError('Points must have shape (tracks, keypoints, 2) or (tracks, 2)')
    if mask.ndim != 2 or not all(mask.shape):
        raise ValueError('Mask must be a nonempty two-dimensional label image')
    if not isinstance(n_masks, (int, np.integer)) or n_masks < 0:
        raise ValueError('n_masks must be a nonnegative integer')
    if (not isinstance(dilation_kernel, (int, np.integer))
            or dilation_kernel < 1 or dilation_kernel % 2 != 1):
        raise ValueError('Dilation kernel must be a positive odd integer')
    areas = mask_areas(mask, max(1, n_masks))
    if n_masks == 0 and np.any(mask):
        raise ValueError('A mask with no object slots must contain only background')

    rounded = np.rint(points)
    height, width = mask.shape
    usable = (np.isfinite(rounded).all(axis=-1)
              & (rounded[..., 0] >= 0) & (rounded[..., 0] < width)
              & (rounded[..., 1] >= 0) & (rounded[..., 1] < height))
    tracks, nodes = np.nonzero(usable)
    coordinates = rounded[tracks, nodes].astype(np.intp)
    overlaps = np.zeros((*points.shape[:2], n_masks), dtype=bool)
    kernel = np.ones((dilation_kernel, dilation_kernel), dtype=np.uint8)
    for c in np.flatnonzero(areas[:n_masks]):
        dilated = cv2.dilate((mask == c + 1).astype(np.uint8), kernel)
        contained = dilated[coordinates[:, 1], coordinates[:, 0]] > 0
        overlaps[tracks, nodes, c] = contained
    return overlaps, usable


def point_overlap_scores(overlaps, usable):
    """Calculate C**2/K from selected nodes, retaining any leading frame axes.

    Overlaps end in (nodes, masks); usability ends in (nodes). C counts
    contained nodes per mask and K counts usable nodes, including points
    outside every mask. A skeleton with K=0 receives zero scores.
    """
    overlaps, usable = np.asarray(overlaps), np.asarray(usable)
    if (overlaps.ndim < 3 or overlaps.shape[:-1] != usable.shape
            or overlaps.dtype != np.bool_ or usable.dtype != np.bool_):
        raise ValueError('Expected boolean per-node overlaps and matching usability flags')
    if np.any(overlaps & ~usable[..., None]):
        raise ValueError('Unusable points cannot overlap a mask')
    contained = overlaps.sum(axis=-2).astype(np.float64)
    counts = usable.sum(axis=-1)[..., None]
    return np.divide(contained ** 2, counts, out=np.zeros_like(contained), where=counts > 0)


def point_mask_scores(points, mask, n_masks, *,
                      dilation_kernel=ANALYSIS_DEFAULTS['point_masks']['dilation_kernel']):
    """Return C**2/K scores for all supplied nodes, with shape (tracks, masks)."""
    return point_overlap_scores(*point_mask_overlaps(points, mask, n_masks,
                                                     dilation_kernel=dilation_kernel))


def validate_point_overlaps(overlaps, usable, scores, mask_present, valid, node_names, excluded):
    """Check saved per-node evidence against frame validity and selected-node scores."""
    if overlaps.shape != (*scores.shape[:2], len(node_names), scores.shape[2]):
        raise ValueError('Point overlaps must cover every frame, track, node and mask')
    if (usable.shape != overlaps.shape[:-1] or np.any(usable[~valid])
            or np.any(overlaps & ~mask_present[:, None, None, :])):
        raise ValueError('Point overlaps and usability must respect frame and mask validity')
    selected = [index for index, name in enumerate(node_names) if name not in excluded]
    point_overlap_scores(overlaps, usable)
    if not np.array_equal(scores, point_overlap_scores(overlaps[:, :, selected], usable[:, :, selected])):
        raise ValueError('Saved overlaps must reproduce the alignment scores')


def align_points_to_masks(points, masks, n_masks, *, mask_valid,
                          invalid_point_frames=(),
                          dilation_kernel=ANALYSIS_DEFAULTS['point_masks']['dilation_kernel'],
                          fallback_previous=ANALYSIS_DEFAULTS['alignment']['fallback_previous']):
    """Align supplied point and mask sequences using explicit frame validity.

    Arrays or frame-indexed readers supply points with shape (frames, tracks,
    keypoints, 2) or (frames, tracks, 2), and masks with shape (frames, height,
    width). mask_valid is a boolean vector
    supplied by the dataset; invalid_point_frames optionally excludes further
    frame indices relative to these arrays. Excluded frames are not scored.
    Track slots and mask labels must refer to consistent identities across
    frames for previous-assignment fallback to be meaningful.

    Return an Alignment with zero-based mask indices (-1 for unmatched),
    combined frame validity and tie flags that persist after fallback. Data
    must already share frame indices; the function estimates no time offset.
    """
    points = points if hasattr(points, 'shape') else np.asarray(points)
    masks = masks if hasattr(masks, 'shape') else np.asarray(masks)
    valid = alignment_valid_frames(mask_valid, invalid_point_frames)
    if (len(points.shape) not in (3, 4) or points.shape[-1] != 2
            or len(masks.shape) != 3 or points.shape[0] != masks.shape[0] or points.shape[0] != len(valid)):
        raise ValueError('Point, mask and validity arrays must have matching frame dimensions')
    if not isinstance(n_masks, (int, np.integer)) or n_masks < 0:
        raise ValueError('n_masks must be a nonnegative integer')
    scores = np.zeros((len(valid), points.shape[1], n_masks), dtype=np.float64)
    for t in np.flatnonzero(valid):
        scores[t] = point_mask_scores(points[t], masks[t], n_masks,
            dilation_kernel=dilation_kernel)
    return align_scores(scores, valid, fallback_previous=fallback_previous)
