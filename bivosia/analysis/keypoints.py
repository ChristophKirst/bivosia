"""Keypoint validation, displacement and identity events relative to linked masks.

Examples
--------
>>> import numpy as np
>>> from bivosia.analysis.keypoints import validate_keypoints
>>> validate_keypoints(np.zeros((2, 1, 3, 2)), [True, False], (20, 30))
(2, 1, 3)
"""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import numpy as np

from bivosia.analysis.alignment import align_scores
from bivosia.analysis.mask_linking import permutation


def keypoint_counts(points):
    """Count finite xy pairs in each supplied skeleton, independent of masks.

    >>> import numpy as np
    >>> from bivosia.analysis.keypoints import keypoint_counts
    >>> points = np.array([[[[1., 2.], [np.nan, 3.]],
    ...                     [[np.nan, np.nan], [np.inf, 0.]]]])
    >>> keypoint_counts(points).tolist()
    [[1, 0]]
    """
    return np.isfinite(points).all(axis=-1).sum(axis=-1)


def keypoint_displacement(points):
    """Return per-track means and their maximum for each adjacent frame pair.

    Points have shape (frames, tracks, nodes, xy). A shared node must have
    finite x and y at both endpoints. An empty intersection gives infinity.
    Output index t describes the transition from frame t to frame t+1.
    All supplied nodes contribute, independently of point-to-mask alignment.

    Examples
    --------
    >>> import numpy as np
    >>> from bivosia.analysis.keypoints import keypoint_displacement
    >>> points = np.array([[[[0., 0.], [np.nan, np.nan]]],
    ...                    [[[3., 4.], [1., 1.]]],
    ...                    [[[np.nan, np.nan], [np.nan, np.nan]]]])
    >>> individual, maximal = keypoint_displacement(points)
    >>> individual[:, 0].tolist(), maximal.tolist()
    ([5.0, inf], [5.0, inf])
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 4 or points.shape[-1] != 2 or points.shape[1] == 0:
        raise ValueError('Expected points with shape (frames, tracks, nodes, 2) and at least one track')
    finite = np.isfinite(points).all(axis=-1)
    shared = finite[:-1] & finite[1:]
    delta = np.zeros_like(points[1:])
    np.subtract(points[1:], points[:-1], out=delta, where=shared[..., None])
    distances = np.linalg.norm(delta, axis=-1)
    counts = shared.sum(axis=-1)
    individual = np.divide(distances.sum(axis=-1), counts,
                           out=np.full(counts.shape, np.inf), where=counts > 0)
    return individual, individual.max(axis=1)


def validate_keypoint_displacement(individual, maximal, frame_count, track_count):
    """Check transition dimensions, nonnegative scores and track maxima."""
    individual, maximal = np.asarray(individual), np.asarray(maximal)
    if individual.shape != (max(0, frame_count - 1), track_count) or maximal.shape != individual.shape[:1]:
        raise ValueError('Displacement arrays must cover every consecutive frame pair and track')
    if np.any(np.isnan(individual)) or np.any(individual < 0):
        raise ValueError('Displacements must be nonnegative or positive infinity')
    if not np.array_equal(maximal, individual.max(axis=1)):
        raise ValueError('Maximal displacement differs from the per-track maximum')


def keypoint_identity_events(result, source, scores, mask_present, links, *, point_counts):
    """Find swaps and assignment errors from each keypoint skeleton's perspective.

    Apply mask identity links to each track's assignment. Compare consecutive
    resolved frames, skipping unresolved frames between them. Resolved frames
    are valid, untied and have an assignment for every mask animal. A changed
    track must be assigned in both endpoints, in the same connected component.
    Appearing or disappearing tracks, local mask relabeling and forward/reverse
    changes alone do not mark full swaps.

    ``swap zones`` contains inclusive (previous, next) resolved-frame pairs,
    without merging touching zones. ``swaps`` marks each zone's later endpoint
    once, independently of its duration or number of changed tracks. The
    per-zone ``swaps over invalid zone`` flags distinguish spans containing
    invalid frames. ``swap frames`` marks the valid frames covered by zones
    for frame-error counts; the plotted intervals include invalid frames too.

    Partial resolved swaps compare successive unambiguous assignments for each
    linked mask, including across gaps. Its assigned keypoint track must change,
    and at least one endpoint must be unresolved as a whole frame. Equal
    endpoint pairs count once even when several masks change. Different pairs
    remain separate events, and overlapping full and partial swaps are retained.
    Tied frames contribute only pairs supported without temporal fallback.
    Partial zones, invalid-span flags and valid-frame coverage are returned
    separately from full swaps.

    No overlap marks a detected skeleton with zero scores for every mask.
    Unassigned marks a detected skeleton with positive overlap but no mask
    assignment. Ties retain the matcher's equal-optimum flags, including when
    temporal fallback supplies assignments. These flags impose no additional
    overlap threshold or restriction on the number of skeletons.
    Ties, unassigned and their intersection are returned as separate, mutually
    exclusive flags for the combined uncertainty strip.
    Missing skeleton marks fewer detected skeletons than expected mask animals.
    A skeleton is detected when point_counts contains at least one finite xy
    pair among the same selected nodes used for alignment and displacement.
    """
    n_masks = mask_present.shape[1]
    validate_keypoint_alignment(result, source, scores, mask_present, n_masks, result.assignment.shape)
    if links.errors:
        raise ValueError('Cannot compare keypoint identities with inconsistent mask links')
    identities = np.full(result.assignment.shape, -1, dtype=np.intp)
    components = np.full(len(result.valid), -1, dtype=np.intp)
    assigned = result.assignment >= 0
    clear = result.assignment.copy()
    tied_frames = np.flatnonzero(result.valid & result.tied)
    if len(tied_frames):
        clear[tied_frames] = align_scores(scores[tied_frames], np.ones(len(tied_frames), bool)).assignment
    mask_tracks = np.full((len(result.valid), n_masks), -1, dtype=np.intp)
    for part in links.partition:
        if part.source == 'invalid':
            continue
        frames = slice(part.start, part.end + 1)
        if np.any(result.valid[frames] & (source[frames] != ('forward', 'reverse').index(part.source))):
            raise ValueError('Keypoint assignments and mask links use different selected sources')
        mapping = np.asarray(permutation(links.maps[part.id], n_masks), dtype=np.intp)
        labels = result.assignment[frames]
        times, tracks = np.nonzero(labels >= 0)
        identities[part.start + times, tracks] = mapping[labels[times, tracks]]
        components[frames] = links.components[part.id]
        labels = clear[frames]
        times, tracks = np.nonzero(labels >= 0)
        mask_tracks[part.start + times, mapping[labels[times, tracks]]] = tracks
    valid = result.valid & (components >= 0)
    resolved = valid & ~result.tied & (assigned.sum(axis=1) == n_masks)
    anchors = np.flatnonzero(resolved)
    previous, following = anchors[:-1], anchors[1:]
    changed = ((components[previous] == components[following])
               & np.any(assigned[previous] & assigned[following]
                        & (identities[previous] != identities[following]), axis=1))
    zones = np.column_stack((previous[changed], following[changed]))
    invalid_count = np.r_[0, np.cumsum(~valid)]
    over_invalid = invalid_count[zones[:, 1] + 1] > invalid_count[zones[:, 0]]
    swaps = np.zeros_like(valid)
    swaps[zones[:, 1]] = True
    swap_frames = np.zeros_like(valid)
    for start, end in zones:
        swap_frames[start:end + 1] = valid[start:end + 1]
    partial_zones = []
    for tracks in mask_tracks.T:
        anchors = np.flatnonzero(valid & (tracks >= 0))
        previous, following = anchors[:-1], anchors[1:]
        changed = ((components[previous] == components[following])
                   & (tracks[previous] != tracks[following])
                   & ~(resolved[previous] & resolved[following]))
        partial_zones.extend(zip(previous[changed], following[changed]))
    partial_zones = np.asarray(sorted(set(partial_zones)), dtype=np.intp).reshape(-1, 2)
    partial_invalid = invalid_count[partial_zones[:, 1] + 1] > invalid_count[partial_zones[:, 0]]
    partial_frames = np.zeros_like(valid)
    for start, end in partial_zones:
        partial_frames[start:end + 1] = valid[start:end + 1]
    return {
        'swaps': swaps,
        'swap zones': zones,
        'swaps over invalid zone': over_invalid,
        'swap frames': swap_frames,
        'partial resolved swap zones': partial_zones,
        'partial swaps over invalid zone': partial_invalid,
        'partial swap frames': partial_frames,
        **keypoint_detection_events(result, scores, point_counts, n_masks, valid),
    }


def keypoint_detection_events(result, scores, point_counts, n_masks, valid):
    """Mark selected-node absence, non-overlap and assignment uncertainty.

    Counts include finite xy pairs only among analysis-selected nodes. A track
    containing only excluded nodes is absent, rather than detected without overlap.

    >>> import numpy as np
    >>> from bivosia.analysis.keypoints import keypoint_detection_events
    >>> from bivosia.analysis.alignment import AlignmentResult
    >>> result = AlignmentResult(np.array([[-1, 0]]), np.array([True]), np.array([False]))
    >>> events = keypoint_detection_events(result, np.array([[[0, 0], [1, 0]]]),
    ...                                    np.array([[0, 1]]), 2, np.array([True]))
    >>> events['missing skeleton'].tolist(), events['no overlap'].tolist()
    ([True], [False])
    """
    point_counts = np.asarray(point_counts)
    if (point_counts.shape != result.assignment.shape
            or not np.issubdtype(point_counts.dtype, np.integer) or np.any(point_counts < 0)):
        raise ValueError('Keypoint counts must be nonnegative integers for every frame and track')
    detected = point_counts > 0
    overlap = np.any(scores > 0, axis=2)
    ties = valid & result.tied
    unassigned = valid & np.any(detected & overlap & (result.assignment < 0), axis=1)
    return {
        'no overlap': valid & np.any(detected & ~overlap, axis=1),
        'ties': ties & ~unassigned,
        'unassigned': unassigned & ~ties,
        'ties + unassigned': ties & unassigned,
        'missing skeleton': valid & (detected.sum(axis=1) < n_masks),
    }


def point_coordinate_scale(source_shape, target_shape):
    """Derive a uniform coordinate scale from source and target image dimensions.

    >>> from bivosia.analysis.keypoints import point_coordinate_scale
    >>> point_coordinate_scale((1536, 2048), (384, 512))
    0.25
    """
    source, target = np.asarray(source_shape), np.asarray(target_shape)
    for shape in (source, target):
        if shape.shape != (2,) or not np.issubdtype(shape.dtype, np.integer) or np.any(shape <= 0):
            raise ValueError('Image dimensions must contain positive integer height and width')
    scales = target / source
    if not np.isclose(scales[0], scales[1], rtol=1e-12, atol=0):
        raise ValueError('Point and target images must have the same aspect ratio')
    return float(scales[0])


def validate_keypoints(points, valid, image_shape):
    """Validate (frames, tracks, nodes, xy) arrays in source-image coordinates."""
    if (len(points.shape) != 4 or points.shape[-1] != 2
            or not all(points.shape) or not np.issubdtype(points.dtype, np.floating)):
        raise ValueError('Keypoints require floating coordinates with shape (frames, tracks, nodes, 2)')
    valid = np.asarray(valid)
    if valid.shape != (points.shape[0],) or valid.dtype != np.bool_:
        raise ValueError('Keypoint validity must be a boolean vector covering every frame')
    shape = np.asarray(image_shape)
    if shape.shape != (2,) or not np.issubdtype(shape.dtype, np.integer) or np.any(shape <= 0):
        raise ValueError('Keypoint image_shape must contain positive integer height and width')
    return points.shape[:3]


def validate_keypoint_alignment(result, source, scores, mask_present, n_masks, shape):
    """Validate assignments, overlap scores, frame flags and selected mask sources."""
    frames, tracks = shape
    if (result.assignment.shape != (frames, tracks)
            or not np.issubdtype(result.assignment.dtype, np.integer)
            or np.any(result.assignment < -1) or np.any(result.assignment >= n_masks)):
        raise ValueError('Keypoint assignments must contain -1 or zero-based mask indices')
    for flags in (result.valid, result.tied):
        if flags.shape != (frames,) or flags.dtype != np.bool_:
            raise ValueError('Alignment validity and tie flags must cover every frame')
    if (source.shape != (frames,) or not np.issubdtype(source.dtype, np.integer)
            or np.any(source < 0) or np.any(source > 2)):
        raise ValueError('Alignment sources must be 0=forward, 1=reverse or 2=invalid')
    if (np.any(result.assignment[~result.valid] != -1) or np.any(result.tied[~result.valid])
            or np.any(result.valid & (source == 2))):
        raise ValueError('Invalid frames cannot contain assignments or ties')
    ordered = np.sort(result.assignment, axis=1)
    if tracks > 1 and np.any((ordered[:, 1:] >= 0) & (ordered[:, 1:] == ordered[:, :-1])):
        raise ValueError('Keypoint assignments must be one-to-one within each frame')
    if (scores.shape != (frames, tracks, n_masks) or not np.isfinite(scores).all()
            or np.any(scores < 0)):
        raise ValueError('Keypoint scores must be finite nonnegative track-to-mask arrays')
    if mask_present.shape != (frames, n_masks) or mask_present.dtype != np.bool_:
        raise ValueError('Mask presence must be a boolean array covering frames and masks')
    if np.any(scores[~result.valid]) or np.any(mask_present[~result.valid]):
        raise ValueError('Invalid frames cannot contain scored masks')
    if np.any((scores > 0) & ~mask_present[:, None, :]):
        raise ValueError('Absent masks cannot have positive keypoint scores')
    times, animals = np.nonzero(result.assignment >= 0)
    if np.any(scores[times, animals, result.assignment[times, animals]] <= 0):
        raise ValueError('Assigned keypoint tracks must have positive overlap with their mask')
