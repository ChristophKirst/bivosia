"""General n-animal Zone-of-Disagreement detection from IoU matrices.

Framewise assignments and quality are computed once, then thresholded into zones.

Examples
--------
>>> from bivosia.analysis.zone_detection import detect_zods, prepare_zod_evidence
>>> import numpy as np
>>> matrices = np.array([np.eye(2), 0.4 * np.eye(2), np.eye(2)])
>>> evidence = prepare_zod_evidence(matrices)
>>> detect_zods(evidence, theta_quality=0.6, theta_recovery=0.9).final_zods
((1, 1),)
"""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from dataclasses import dataclass

import numpy as np

from bivosia.config import ANALYSIS_DEFAULTS
from bivosia.analysis.masks import frame_quality_measures, optimal_assignments, permutation_change_measures


Interval = tuple[int, int]


@dataclass(frozen=True)
class ZodEvidence:
    """Framewise evidence that does not depend on detector thresholds.

    A switch is a change in the forward-to-reverse assignment between adjacent
    valid frames. ``switch_frames`` marks both frames incident to each change;
    it does not assert an identity swap within either mask direction.
    """

    quality: np.ndarray
    missing: np.ndarray
    assignment: np.ndarray
    permutation_change: np.ndarray
    switch_frames: np.ndarray


@dataclass(frozen=True)
class ZodResult:
    """Thresholded seeds and final proposed ZoDs (all intervals inclusive)."""

    missing_seeds: tuple[Interval, ...]
    quality_seeds: tuple[Interval, ...]
    switch_seeds: tuple[Interval, ...]
    initial_zods: tuple[Interval, ...]
    final_zods: tuple[Interval, ...]




def prepare_zod_evidence(
    iou_matrices: np.ndarray, assignment: np.ndarray | None = None
) -> ZodEvidence:
    """Calculate assignments, signed margin quality and switch evidence.

    ``iou_matrices[t, i, j]`` is the unweighted IoU of forward animal ``i``
    and reverse animal ``j``. Missing rows/columns must contain NaN. A
    zero-filled working copy is used for a deterministic best-effort assignment
    even at missing-mask frames; such frames cannot generate switch evidence.
    """
    matrices = np.asarray(iou_matrices)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError(
            "IoU matrices must have shape (frames, animals, animals); "
            f"got {matrices.shape}"
        )
    frame_count, n_animals, _ = matrices.shape
    if n_animals < 2:
        raise ValueError("Permutation disagreement requires at least two animals")

    if assignment is None:
        assignments = optimal_assignments(matrices)
    else:
        assignments = np.asarray(assignment)
        if assignments.shape != (frame_count, n_animals):
            raise ValueError(
                "IoU match array must have shape (frames, animals); "
                f"got {assignments.shape}, expected {(frame_count, n_animals)}"
            )
        if not np.issubdtype(assignments.dtype, np.integer):
            raise ValueError("IoU match array must contain integer column indices")
        if np.any(assignments < 0) or np.any(assignments >= n_animals):
            raise ValueError("IoU match array contains an out-of-range identity")
        expected = np.arange(n_animals)
        if np.any(np.sort(assignments, axis=1) != expected):
            raise ValueError("Every row of the IoU match array must be a permutation")
        assignments = assignments.astype(np.int16, copy=False)

    quality, missing, _assigned, _local_margin = frame_quality_measures(
        matrices, assignments
    )

    changes, switch_frames = permutation_change_measures(assignments, missing)
    return ZodEvidence(quality, missing, assignments, changes, switch_frames)


def boolean_runs(indicator: np.ndarray) -> tuple[Interval, ...]:
    """Convert a Boolean frame indicator into maximal inclusive intervals."""
    values = np.asarray(indicator, dtype=bool)
    if values.ndim != 1:
        raise ValueError("Frame indicators must be one-dimensional")
    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return tuple((int(start), int(end)) for start, end in zip(starts, ends))


def minimum_frame_value(values: np.ndarray, frames) -> float | None:
    """Return the minimum finite value at selected frames.

    >>> import numpy as np
    >>> from bivosia.analysis.zone_detection import minimum_frame_value
    >>> minimum_frame_value(np.array([0.8, np.nan, 0.4]), (0, 1, 2))
    0.4
    >>> minimum_frame_value(np.array([np.nan]), (0,)) is None
    True
    """
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("Frame values must be one-dimensional")
    indices = np.asarray(tuple(frames), dtype=np.int64)
    if not indices.size:
        return None
    if np.any(indices < 0) or np.any(indices >= len(array)):
        raise ValueError("Selected frame is outside the value array")
    selected = array[indices]
    finite = selected[np.isfinite(selected)]
    return None if not finite.size else float(np.min(finite))




def mask_count_seed_signal(
    evidence: ZodEvidence,
    forward_missing_masks: np.ndarray | None = None,
    reverse_missing_masks: np.ndarray | None = None,
) -> np.ndarray:
    """Return frames with missing masks or unequal forward/reverse mask counts.

    The IoU-derived missing indicator is always retained. When dense missing
    counts for both passes are available, a frame is additionally selected if
    either pass is incomplete, either count is non-finite, or the pass counts
    disagree.
    """
    signal = np.asarray(evidence.missing, dtype=bool).copy()
    if forward_missing_masks is None and reverse_missing_masks is None:
        return signal
    if forward_missing_masks is None or reverse_missing_masks is None:
        raise ValueError("Forward and reverse missing-mask counts must be supplied together")
    forward = np.asarray(forward_missing_masks)
    reverse = np.asarray(reverse_missing_masks)
    if forward.shape != signal.shape or reverse.shape != signal.shape:
        raise ValueError("Missing-mask counts have an incompatible frame count")
    signal |= (
        ~np.isfinite(forward)
        | ~np.isfinite(reverse)
        | (forward > 0)
        | (reverse > 0)
        | (forward != reverse)
    )
    return signal




def _expand_to_recovery(
    intervals: tuple[Interval, ...], recovered: np.ndarray
) -> tuple[Interval, ...]:
    """Extend each interval up to the nearest recovered frame on either side."""
    frame_count = len(recovered)
    if not intervals:
        return ()

    # Locate the nearest recovery frames with binary searches.
    recovery_frames = np.flatnonzero(np.asarray(recovered, dtype=bool))
    starts = np.fromiter((start for start, _end in intervals), dtype=np.int64)
    ends = np.fromiter((end for _start, end in intervals), dtype=np.int64)
    left_positions = np.searchsorted(recovery_frames, starts, side="left") - 1
    right_positions = np.searchsorted(recovery_frames, ends, side="right")
    expanded_starts = np.zeros(len(intervals), dtype=np.int64)
    expanded_ends = np.full(len(intervals), frame_count - 1, dtype=np.int64)
    has_left = left_positions >= 0
    has_right = right_positions < len(recovery_frames)
    expanded_starts[has_left] = recovery_frames[left_positions[has_left]] + 1
    expanded_ends[has_right] = recovery_frames[right_positions[has_right]] - 1
    return tuple(
        (int(start), int(end))
        for start, end in zip(expanded_starts, expanded_ends)
    )


def merge_intervals(
    intervals: tuple[Interval, ...], maximum_gap: int
) -> tuple[Interval, ...]:
    if maximum_gap < 0:
        raise ValueError("theta_gap must be non-negative")
    if not intervals:
        return ()
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if not merged or start - merged[-1][1] - 1 > maximum_gap:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return tuple((start, end) for start, end in merged)


def detect_zods(
    evidence: ZodEvidence,
    theta_quality: float = ANALYSIS_DEFAULTS['zone_detection']['theta_quality'],
    *,
    expand_to_baseline: bool = ANALYSIS_DEFAULTS['zone_detection']['expand_to_baseline'],
    theta_recovery: float = ANALYSIS_DEFAULTS['zone_detection']['theta_recovery'],
    theta_gap: int = ANALYSIS_DEFAULTS['zone_detection']['theta_gap'],
    quality_signal: np.ndarray | None = None,
    recovery_signal: np.ndarray | None = None,
    missing_signal: np.ndarray | None = None,
    quality_higher_is_better: bool = True,
    recovery_higher_is_better: bool = True,
) -> ZodResult:
    """Threshold prepared evidence and return typed seeds and final ZoDs.

    ``quality_signal`` selects the framewise measure used for poor-quality
    seeds. ``recovery_signal`` independently selects the measure used to stop
    recovery expansion. Both default to local-margin quality.
    ``missing_signal`` can augment or replace the IoU-derived missing-mask
    indicator, for example with a forward/reverse mask-count mismatch signal.
    Direction flags support both quality measures (higher is better) and
    ambiguity/error measures (higher is worse).
    """
    if not np.isfinite(theta_quality) or not np.isfinite(theta_recovery):
        raise ValueError("Quality and recovery thresholds must be finite")

    seed_quality = np.asarray(
        evidence.quality if quality_signal is None else quality_signal
    )
    same_quality_measure = recovery_signal is None or (
        quality_signal is not None and recovery_signal is quality_signal
    )
    recovery_quality = np.asarray(
        seed_quality if recovery_signal is None else recovery_signal
    )
    missing = np.asarray(
        evidence.missing if missing_signal is None else missing_signal,
        dtype=bool,
    )
    expected_shape = evidence.quality.shape
    if seed_quality.shape != expected_shape:
        raise ValueError("The selected quality signal has an incompatible frame count")
    if recovery_quality.shape != expected_shape:
        raise ValueError("The selected recovery signal has an incompatible frame count")
    if missing.shape != expected_shape:
        raise ValueError("The selected missing-mask signal has an incompatible frame count")
    if (
        expand_to_baseline
        and same_quality_measure
        and quality_higher_is_better == recovery_higher_is_better
        and (
            (quality_higher_is_better and theta_recovery < theta_quality)
            or (not quality_higher_is_better and theta_recovery > theta_quality)
        )
    ):
        raise ValueError(
            "The recovery threshold must be farther into the recovered direction than the seed "
            "threshold when one measure drives both stages"
        )

    # Missing frames contribute missing-mask seeds, not quality seeds.
    quality_thresholded = (
        seed_quality < theta_quality
        if quality_higher_is_better
        else seed_quality > theta_quality
    )
    quality_poor = (~missing) & np.isfinite(seed_quality) & quality_thresholded
    combined = missing | quality_poor | evidence.switch_frames
    initial = boolean_runs(combined)
    candidates = initial
    if expand_to_baseline:
        recovery_thresholded = (
            recovery_quality >= theta_recovery
            if recovery_higher_is_better
            else recovery_quality <= theta_recovery
        )
        recovered = (~missing) & np.isfinite(recovery_quality) & recovery_thresholded
        candidates = _expand_to_recovery(initial, recovered)
    final = merge_intervals(candidates, theta_gap)
    return ZodResult(
        missing_seeds=boolean_runs(missing),
        quality_seeds=boolean_runs(quality_poor),
        switch_seeds=boolean_runs(evidence.switch_frames),
        initial_zods=initial,
        final_zods=final,
    )
