"""Mask overlap, assignment quality, missing identities and pixel relabeling.

Examples
--------
>>> from bivosia.analysis.masks import frame_iou_matrix, mask_intersections, mask_presence, missing_mask_counts, relabel_mask
>>> import numpy as np
>>> labels = np.array([[0, 1], [2, 2]])
>>> intersections = mask_intersections(labels, labels, 2)
>>> intersections.tolist()
[[1, 0], [0, 2]]
>>> iou, forward_area, reverse_area = frame_iou_matrix(labels, labels, 2)
>>> iou.tolist()
[[1.0, 0.0], [0.0, 1.0]]
>>> forward_area.tolist(), reverse_area.tolist()
([1, 2], [1, 2])
>>> cached, _, _ = frame_iou_matrix(labels, labels, 2, areas=(forward_area, reverse_area))
>>> bool(np.array_equal(cached, iou))
True
>>> iou, forward_area, reverse_area = frame_iou_matrix(labels, np.zeros_like(labels), 2)
>>> bool(np.isnan(iou).all()), reverse_area.tolist()
(True, [0, 0])
>>> missing_mask_counts(np.stack((forward_area, reverse_area))).tolist()
[0, 2]
>>> relabel_mask(labels, (1, 0)).tolist()
[[0, 2], [1, 1]]
>>> mask_presence(labels, 3).tolist()
[True, True, False]
"""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import numpy as np
from scipy.optimize import linear_sum_assignment

from bivosia.config import LINKED_MASK_STREAMS


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


def mask_areas(mask: np.ndarray, n: int) -> np.ndarray:
    """Count foreground pixels for raw object IDs 1..n, excluding background."""
    labels = _mask_labels(mask, n)
    return np.bincount(labels, minlength=n + 1)[1:n + 1].astype(np.int64, copy=False)


def _mask_labels(mask, n):
    mask = np.asarray(mask)
    if n < 1 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("Masks require integer labels and a positive animal count")
    if mask.size and (mask.min() < 0 or mask.max() > n):
        raise ValueError(f"Mask labels must be between 0 and {n}")
    return mask.reshape(-1).astype(np.int64, copy=False)


def mask_intersections(first: np.ndarray, second: np.ndarray, n: int) -> np.ndarray:
    """Count pairwise foreground intersections; rows index first, columns second."""
    if first.shape != second.shape:
        raise ValueError("Masks must use the same image coordinates")
    first, second = _mask_labels(first, n), _mask_labels(second, n)
    valid = (first > 0) & (second > 0)
    pairs = (first[valid] - 1) * n + second[valid] - 1
    return np.bincount(pairs, minlength=n * n).reshape(n, n).astype(np.int64, copy=False)


def iou_from_intersections(intersections, first_area, second_area):
    """Derive float32 IoU from integer counts, with NaN for either absent mask.

    Accept one matrix or batches: (..., N, M), (..., N), (..., M).
    Areas include mask pixels overlapping the other image's background.
    Counts are promoted to int64 before arithmetic to avoid unsigned overflow.
    """
    intersections, first_area, second_area = map(np.asarray, (intersections, first_area, second_area))
    if (intersections.ndim < 2 or first_area.shape != intersections.shape[:-1]
            or second_area.shape != intersections.shape[:-2] + intersections.shape[-1:]):
        raise ValueError("Intersection and area dimensions differ")
    if any(not np.issubdtype(values.dtype, np.integer) for values in (intersections, first_area, second_area)):
        raise ValueError("Intersections and areas must contain integer pixel counts")
    intersections, first_area, second_area = (
        values.astype(np.int64, copy=False) for values in (intersections, first_area, second_area))
    if any(np.any(values < 0) for values in (intersections, first_area, second_area)):
        raise ValueError("Pixel counts must be nonnegative")
    # Each label image partitions its foreground into disjoint masks.
    if (np.any(intersections.sum(axis=-1) > first_area)
            or np.any(intersections.sum(axis=-2) > second_area)):
        raise ValueError("Intersections exceed the corresponding mask areas")
    union = first_area[..., :, None] + second_area[..., None, :] - intersections
    present = (first_area[..., :, None] > 0) & (second_area[..., None, :] > 0)
    return np.divide(intersections, union,
                     out=np.full(intersections.shape, np.nan, dtype=np.float32), where=present)


def frame_iou_matrix(
    first: np.ndarray, second: np.ndarray, n: int, *,
    areas: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pairwise IoU and both area vectors for object IDs 1..n.

    Areas count foreground pixels, excluding background ID 0. Missing masks
    have area zero; their IoU pairs are NaN. Supply precomputed areas to reuse
    them when comparing the same masks with another direction or time frame.
    Rows index the first mask image; columns index the second.
    """
    first_area, second_area = (mask_areas(first, n), mask_areas(second, n)) if areas is None else areas
    result = iou_from_intersections(mask_intersections(first, second, n), first_area, second_area)
    return result, first_area, second_area


def optimal_assignments(iou_matrices: np.ndarray) -> np.ndarray:
    """Return the independent per-frame Hungarian permutation."""
    matrices = np.asarray(iou_matrices)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError(
            "IoU matrices must have shape (frames, animals, animals); "
            f"got {matrices.shape}"
        )
    frame_count, n_animals, _ = matrices.shape
    work = np.nan_to_num(matrices, nan=0.0, posinf=0.0, neginf=0.0)
    assignments = np.empty((frame_count, n_animals), dtype=np.int16)
    if n_animals == 2:
        identity = work[:, 0, 0] + work[:, 1, 1]
        crossed = work[:, 0, 1] + work[:, 1, 0]
        use_identity = identity >= crossed
        assignments[:, 0] = np.where(use_identity, 0, 1)
        assignments[:, 1] = np.where(use_identity, 1, 0)
    else:
        for frame in range(frame_count):
            rows, columns = linear_sum_assignment(work[frame], maximize=True)
            assignments[frame, rows] = columns
    return assignments


def frame_quality_measures(
    iou_matrices: np.ndarray, assignment: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return signed quality, missing flags, assigned IoUs and local margins."""
    matrices = np.asarray(iou_matrices)
    assignments = np.asarray(assignment)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("IoU matrices must have shape (frames, animals, animals)")
    frame_count, n_animals, _ = matrices.shape
    if n_animals < 2:
        raise ValueError("Local assignment margins require at least two animals")
    if assignments.shape != (frame_count, n_animals):
        raise ValueError("IoU match array has an incompatible shape")
    missing = np.isnan(matrices).any(axis=(1, 2))
    work = np.nan_to_num(matrices, nan=0.0, posinf=0.0, neginf=0.0)
    frame_indices = np.arange(frame_count)[:, None]
    animal_indices = np.arange(n_animals)[None, :]
    assigned = work[frame_indices, animal_indices, assignments]

    # Compare each assigned pair with the strongest unassigned row/column pair.
    alternatives = work.copy()
    alternatives[frame_indices, animal_indices, assignments] = -np.inf
    best_row_alternative = np.max(alternatives, axis=2)
    best_column_alternatives = np.max(alternatives, axis=1)
    best_column_alternative = best_column_alternatives[frame_indices, assignments]
    row_margin = assigned - best_row_alternative
    column_margin = assigned - best_column_alternative
    local_margin = np.minimum(row_margin, column_margin).astype(np.float32, copy=False)
    quality = np.min(local_margin, axis=1).astype(np.float32, copy=False)
    assigned = assigned.astype(np.float32, copy=False)
    if np.any(missing):
        # Assignment quality is undefined when a mask is missing.
        quality = quality.copy()
        assigned = assigned.copy()
        local_margin = local_margin.copy()
        quality[missing] = np.nan
        assigned[missing] = np.nan
        local_margin[missing] = np.nan
    return quality, missing, assigned, local_margin


def permutation_change_measures(
    assignment: np.ndarray, missing: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return valid adjacent permutation changes and their incident frames."""
    assignments = np.asarray(assignment)
    missing_frames = np.asarray(missing, dtype=bool)
    frame_count = assignments.shape[0]
    if assignments.ndim != 2 or missing_frames.shape != (frame_count,):
        raise ValueError("Assignment and missing arrays have incompatible shapes")
    changes = np.zeros(max(0, frame_count - 1), dtype=bool)
    if frame_count > 1:
        changes = np.any(assignments[1:] != assignments[:-1], axis=1)
        changes &= ~missing_frames[:-1] & ~missing_frames[1:]
    switch_frames = np.zeros(frame_count, dtype=bool)
    if changes.size:
        switch_frames[:-1] |= changes
        switch_frames[1:] |= changes
    return changes, switch_frames


def missing_mask_counts(present: np.ndarray) -> np.ndarray:
    """Count zeros along the identity axis of Boolean presence or pixel areas."""
    present = np.asarray(present, dtype=bool)
    return present.shape[-1] - np.count_nonzero(present, axis=-1)


def mask_presence(mask, n):
    """Return whether each object ID 1..n occurs in the mask."""
    return np.isin(np.arange(1, n + 1), np.unique(mask))


def mask_centers(masks, n):
    ys, xs = np.nonzero((masks > 0) & (masks <= n))
    labels = masks[ys, xs].astype(np.intp)
    counts = np.bincount(labels, minlength=n + 1)
    sum_x = np.bincount(labels, weights=xs, minlength=n + 1)
    sum_y = np.bincount(labels, weights=ys, minlength=n + 1)
    return tuple((float(sum_x[i] / counts[i]), float(sum_y[i] / counts[i])) if counts[i] else None
                 for i in range(1, n + 1))


def relabel_mask(mask, mapping):
    n = len(mapping)
    if np.any(mask < 0) or np.any(mask > n):
        raise ValueError("Raw mask contains a label outside the declared animal count")
    lookup = np.asarray([0, *(int(i) + 1 for i in mapping)], dtype=np.uint16)
    return lookup[mask]
