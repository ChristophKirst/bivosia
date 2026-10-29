"""Partial point-to-mask assignments with explicit ties and temporal fallback.

Examples
--------
>>> import numpy as np
>>> from bivosia.analysis.alignment import align_scores
>>> scores = np.array([[[1, 0], [1, 0]], [[1, 0], [0, 1]],
...                    [[1, 1], [1, 1]]], dtype=float)
>>> result = align_scores(scores, [True, True, True])
>>> result.assignment.tolist()
[[-1, -1], [0, 1], [-1, -1]]
>>> result.tied.tolist()
[True, False, True]
>>> result = align_scores(scores, [True, True, True], fallback_previous=True)
>>> result.assignment.tolist()
[[0, -1], [0, 1], [0, 1]]
>>> result.tied.tolist()
[True, False, True]
"""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from bivosia.config import ANALYSIS_DEFAULTS


@dataclass(frozen=True)
class Alignment:
    """Assignments and frame flags, retaining ties after temporal fallback.

    ``assignment[t, a]`` is the zero-based mask index for point track a,
    or -1 when unmatched. ``valid`` marks frames eligible for alignment;
    ``tied`` marks valid frames with multiple optimal positive-overlap
    assignments. Invalid frames have only -1 assignments and tied=False.
    """

    assignment: np.ndarray
    valid: np.ndarray
    tied: np.ndarray


def alignment_valid_frames(mask_valid, invalid_point_frames=()):
    """Combine dataset mask validity with explicit invalid point-frame indices.

    Examples
    --------
    >>> from bivosia.analysis.alignment import alignment_valid_frames
    >>> alignment_valid_frames([True, False, True], [2]).tolist()
    [True, False, False]
    """
    valid = np.asarray(mask_valid)
    if valid.ndim != 1 or valid.dtype != np.bool_:
        raise ValueError('Mask validity must be a one-dimensional boolean array')
    invalid = np.asarray(invalid_point_frames)
    if invalid.ndim != 1 or (invalid.size and not np.issubdtype(invalid.dtype, np.integer)):
        raise ValueError('Invalid point frames must be a one-dimensional list of integer indices')
    if invalid.size and (invalid.min() < 0 or invalid.max() >= len(valid)):
        raise ValueError('Invalid point frame outside the supplied sequence')
    valid = valid.copy()
    valid[invalid.astype(np.intp)] = False
    return valid


def _solve(weights, scores):
    """Return real assignments and their total score."""
    rows, columns = linear_sum_assignment(weights, maximize=True)
    real = columns < scores.shape[1]
    assignment = np.full(scores.shape[0], -1, dtype=np.intp)
    assignment[rows[real]] = columns[real]
    total = math.fsum(float(scores[a, c]) for a, c in zip(rows[real], columns[real]))
    return assignment, total


def _same_objective(first, second):
    """Compare total assignment scores using the configured tie tolerances."""
    parameters = ANALYSIS_DEFAULTS['alignment']
    return math.isclose(first, second,
                        rel_tol=parameters['tie_relative_tolerance'],
                        abs_tol=parameters['tie_absolute_tolerance'])


def _match_frame(scores, previous=None):
    """Keep pairs shared by all optima, optionally restoring previous tied pairs."""
    if not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError('Valid-frame scores must be finite and nonnegative')
    n_tracks, n_masks = scores.shape
    unmatched = np.full(n_tracks, -1, dtype=np.intp)
    if not np.any(scores > 0):
        return unmatched, False

    # Zero-valued dummy columns permit unmatched rows without changing the
    # total-score objective. Zero-overlap real edges are forbidden.
    weights = np.zeros((n_tracks, n_masks + n_tracks), dtype=np.float64)
    weights[:, :n_masks] = np.where(scores > 0, scores, -np.inf)
    best, objective = _solve(weights, scores)
    ambiguous = np.zeros(n_tracks, dtype=bool)
    for a, c in enumerate(best):
        if c < 0 and not np.any(scores[a] > 0):
            continue
        alternative = weights.copy()
        if c >= 0:
            alternative[a, c] = -np.inf
        else:
            # An unmatched row is ambiguous if some optimum matches it.
            # Permutations among dummy columns do not represent different labels.
            alternative[a, n_masks:] = -np.inf
        _, other_objective = _solve(alternative, scores)
        ambiguous[a] = _same_objective(objective, other_objective)

    result = best.copy()
    result[ambiguous] = -1
    if previous is not None:
        for a in np.flatnonzero(ambiguous):
            c = previous[a]
            if c < 0 or scores[a, c] <= 0:
                continue
            # Retain earlier labels only if they can coexist in a current
            # optimum. Constrain accepted pairs before considering the next one.
            candidate = weights.copy()
            candidate[a, :] = -np.inf
            candidate[:, c] = -np.inf
            candidate[a, c] = weights[a, c]
            try:
                _, candidate_objective = _solve(candidate, scores)
            except ValueError:
                continue
            if _same_objective(objective, candidate_objective):
                result[a] = c
                weights = candidate
    return result, bool(ambiguous.any())


def align_scores(scores, valid_frames, *, fallback_previous=ANALYSIS_DEFAULTS['alignment']['fallback_previous']):
    """Align a supplied (frames, point tracks, masks) score array.

    Maximize the total score, leaving zero-overlap pairs unmatched.
    Unequal track/mask counts and empty sides are allowed. A track is left
    unmatched when equally optimal assignments disagree about its mask or
    whether it is matched. Other unambiguous pairs remain assigned.

    With fallback_previous=True, tied frames reuse compatible pairs from the
    preceding valid frame's assignment. Invalid frames are skipped entirely.
    Initial ties are processed backwards from the first valid non-tied frame.
    If every valid frame is tied, there is no temporal anchor. All tied flags
    remain True even when fallback supplies some or all labels. Frames with
    no positive scores are non-tied and have an entirely unmatched assignment.
    Score equality uses the configured relative and absolute tie tolerances.
    """
    scores = np.asarray(scores, dtype=np.float64)
    valid = alignment_valid_frames(valid_frames)
    if scores.ndim != 3 or scores.shape[0] != len(valid):
        raise ValueError('Scores must have shape (frames, point tracks, masks) matching validity')
    assignment = np.full(scores.shape[:2], -1, dtype=np.intp)
    tied = np.zeros(len(valid), dtype=bool)
    for t, labels, tie in iter_align_scores(scores, valid, fallback_previous=fallback_previous):
        assignment[t], tied[t] = labels, tie
    return Alignment(assignment, valid, tied)


def iter_align_scores(scores, valid_frames, *, fallback_previous=ANALYSIS_DEFAULTS['alignment']['fallback_previous']):
    """Yield (frame, assignment, tied), followed by any initial-tie corrections.

    Frames are first yielded in chronological order, including invalid frames
    with unmatched labels. Backward fallback yields updated leading tied frames
    afterwards. This permits recording processing to report progress and cancel
    between calculations while keeping temporal assignment rules here.
    """
    scores = np.asarray(scores, dtype=np.float64)
    valid = alignment_valid_frames(valid_frames)
    if scores.ndim != 3 or scores.shape[0] != len(valid):
        raise ValueError('Scores must have shape (frames, point tracks, masks) matching validity')
    previous = anchor = None
    leading_ties = []
    for t in range(len(valid)):
        if not valid[t]:
            yield t, np.full(scores.shape[1], -1, dtype=np.intp), False
            continue
        labels, tied = _match_frame(scores[t], previous if fallback_previous else None)
        previous = labels
        if anchor is None:
            if tied:
                leading_ties.append(t)
            else:
                anchor = labels
        yield t, labels, tied
    if fallback_previous and anchor is not None:
        previous = anchor
        for t in reversed(leading_ties):
            labels, _ = _match_frame(scores[t], previous)
            previous = labels
            yield t, labels, True
