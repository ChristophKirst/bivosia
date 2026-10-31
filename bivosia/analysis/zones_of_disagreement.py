"""Zones of Disagreement: evidence, detection, summaries and interval editing.

Missing masks, adjacent assignment switches and low assignment quality seed
inclusive intervals. Recovery expansion and gap merging produce the final
zones. Frame-event classification is shared by zone summaries and timelines.

Examples
--------
>>> from bivosia.analysis.zones_of_disagreement import detect_zods, prepare_zod_evidence
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

from dataclasses import asdict, dataclass, replace
from uuid import uuid4
import hashlib
import json

import numpy as np

from bivosia.config import ANALYSIS_DEFAULTS, AUTO_DIRECTION_METHODS
from bivosia.analysis.mask_quality import frame_quality_measures, optimal_assignments, permutation_change_measures


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
    >>> from bivosia.analysis.zones_of_disagreement import minimum_frame_value
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


def frame_event_signals(missing_signal=None, forward_missing_signal=None,
                        reverse_missing_signal=None, missing_direction_known=None,
                        switch_frames=None):
    """Classify directional missing masks and incident switch frames once.

    Supplied missing counts or Boolean indicators are positive for missing
    masks. An explicit direction-known indicator takes precedence; otherwise
    both directional inputs must be present and finite.

    >>> from bivosia.analysis.zones_of_disagreement import frame_event_signals
    >>> events = frame_event_signals([True, True], [1, 0], [0, 1])
    >>> events['Missing forward only'].tolist()
    [True, False]
    >>> events['Missing reverse only'].tolist()
    [False, True]
    """
    signals = [value for value in (
        missing_signal, forward_missing_signal, reverse_missing_signal,
        missing_direction_known, switch_frames,
    ) if value is not None]
    if not signals:
        return {}
    frame_count = len(signals[0])
    if any(np.asarray(value).shape != (frame_count,) for value in signals):
        raise ValueError('Zone frame signals must all have the same length')

    def flags(value):
        return np.zeros(frame_count, dtype=bool) if value is None else np.asarray(value) > 0

    forward = flags(forward_missing_signal)
    reverse = flags(reverse_missing_signal)
    missing = flags(missing_signal) if missing_signal is not None else forward | reverse
    switches = flags(switch_frames)
    if missing_direction_known is not None:
        known = flags(missing_direction_known)
    elif forward_missing_signal is not None and reverse_missing_signal is not None:
        known = np.isfinite(forward_missing_signal) & np.isfinite(reverse_missing_signal)
    else:
        known = np.zeros(frame_count, dtype=bool)
    return {
        'Missing forward only': missing & known & forward & ~reverse,
        'Missing reverse only': missing & known & reverse & ~forward,
        'Missing both': missing & known & forward & reverse,
        'Missing direction unknown': missing & (~known | ~(forward | reverse)),
        'Switch frames': switches,
        'Missing mask + switch': missing & switches,
    }


def frame_event_intervals(missing_signal=None, forward_missing_signal=None,
                          reverse_missing_signal=None, missing_direction_known=None,
                          switch_frames=None):
    """Return inclusive runs for each missing-mask and switch classification."""
    events = frame_event_signals(missing_signal, forward_missing_signal,
                                reverse_missing_signal, missing_direction_known, switch_frames)
    return {label: list(boolean_runs(values)) for label, values in events.items()}


def zone_frame_summary(frames, *, events=None, quality=None):
    """Summarize supplied frame events and finite quality within one zone.

    Directional flags include missing masks anywhere in the zone, so both can
    be true even when the two directions are missing in different frames.

    >>> from bivosia.analysis.zones_of_disagreement import frame_event_signals, zone_frame_summary
    >>> events = frame_event_signals([True, True], [1, 0], [0, 1], switch_frames=[0, 1])
    >>> summary = zone_frame_summary((0, 1), events=events, quality=[0.3, 0.8])
    >>> summary
    {'missing_in_forward': True, 'missing_in_reverse': True, 'missing_direction_unknown': False, 'has_switch': True, 'minimum_quality': 0.3}
    """
    frames = tuple(frames)
    indices = np.asarray(frames, dtype=np.int64)
    events = events or {}

    def present(*names):
        return any(np.any(events[name][indices]) for name in names if name in events)

    return {
        'missing_in_forward': present('Missing forward only', 'Missing both'),
        'missing_in_reverse': present('Missing reverse only', 'Missing both'),
        'missing_direction_unknown': present('Missing direction unknown'),
        'has_switch': present('Switch frames'),
        'minimum_quality': None if quality is None else minimum_frame_value(quality, frames),
    }


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
    theta_quality: float = ANALYSIS_DEFAULTS['zones_of_disagreement']['theta_quality'],
    *,
    expand_to_baseline: bool = ANALYSIS_DEFAULTS['zones_of_disagreement']['expand_to_baseline'],
    theta_recovery: float = ANALYSIS_DEFAULTS['zones_of_disagreement']['theta_recovery'],
    theta_gap: int = ANALYSIS_DEFAULTS['zones_of_disagreement']['theta_gap'],
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


SOURCES = ("forward", "reverse", "invalid")


def revision_fingerprint(header, zones, links, link_frames):
    """Identify the annotation state used to validate a linked-mask export."""
    payload = {"header": header, "zones": {zone['id']: zone for zone in zones},
               "links": links, "link_frames": {str(frame): frame for frame in link_frames}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_consensus_export(annotation, fingerprint):
    """Require an export to match the current saved annotation."""
    current = revision_fingerprint(annotation['header'], annotation['zones'],
                                   annotation['links'], annotation.get('link_frames', []))
    if annotation['export']['fingerprint'] != current or fingerprint != current:
        raise ValueError('Consensus masks do not match the current annotation; export again')


@dataclass(frozen=True)
class ZodAnnotation:
    """A ZoD interval with its mask choice, review status and directional judgments.

    Validity is true, false or unknown. Swap values are event counts;
    omitted counts mean zero. Counts do not identify event frames or approve
    identity links.
    """

    id: str
    start: int
    end: int
    source: str = "forward"
    source_manual: bool = False
    status: str = "unseen"
    forward_valid: bool | None = None
    reverse_valid: bool | None = None
    forward_swap: int | None = 0
    reverse_swap: int | None = 0

    def record(self):
        return asdict(self)


def validate_zones(zones, frame_count):
    ordered = sorted(zones, key=lambda z: (z.start, z.end))
    if len({z.id for z in ordered}) != len(ordered):
        raise ValueError("Duplicate zone IDs")
    previous_end = -1
    for zone in ordered:
        if not isinstance(zone.start, int) or not isinstance(zone.end, int):
            raise ValueError("Frame endpoints must be integers")
        if not 0 <= zone.start <= zone.end < frame_count or zone.start <= previous_end:
            raise ValueError("Zones must be nonempty, inside the video, and non-overlapping")
        if zone.source not in SOURCES or zone.status not in {"unseen", "draft", "reviewed", "needs_review", "automatic"}:
            raise ValueError("Invalid zone source or review status")
        for value in (zone.forward_valid, zone.reverse_valid):
            if value is not None and not isinstance(value, bool):
                raise ValueError('Hand validity must be true, false or unknown')
        for value in (zone.forward_swap, zone.reverse_swap):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError('Hand swap counts must be nonnegative integers or omitted')
        previous_end = zone.end
    return ordered


def zone_has_missing_masks(zone, missing_forward, missing_reverse, *, automatic=False):
    """Check whether the chosen direction exceeds its permitted missing-mask count."""
    if zone.source == 'invalid':
        return False
    missing = missing_forward if zone.source == 'forward' else missing_reverse
    allowed_missing = int(automatic and zone.status == 'automatic')
    return bool((np.asarray(missing)[zone.start:zone.end + 1] > allowed_missing).any())


def critical_zones(zones, missing_forward, missing_reverse, *, automatic=False):
    """Select unresolved choices, invalid zones and incomplete chosen directions.

    An ordinary automatic choice does not require manual approval to be skipped.
    Invalid zones remain available for inspection even after review.

    >>> from bivosia.analysis.zones_of_disagreement import ZodAnnotation, critical_zones
    >>> zones = [ZodAnnotation('auto', 0, 1, status='automatic'),
    ...          ZodAnnotation('invalid', 2, 3, source='invalid', status='reviewed'),
    ...          ZodAnnotation('edited', 4, 5, status='needs_review')]
    >>> [zone.id for zone in critical_zones(zones, [0] * 6, [0] * 6)]
    ['invalid', 'edited']
    """
    return tuple(zone for zone in zones
                 if zone.status not in {'automatic', 'reviewed'} or zone.source == 'invalid'
                 or zone_has_missing_masks(zone, missing_forward, missing_reverse, automatic=automatic))


def default_source(start, end, missing_forward, missing_reverse, *, temporal_quality=None,
                   direction_method=ANALYSIS_DEFAULTS['zones_of_disagreement']['direction_method']):
    """Propose a mask direction using flags, counts or temporal quality.

    ``missing_forward`` and ``missing_reverse`` contain per-frame counts.
    The flags rule chooses a complete direction, defaults to forward if both
    are complete, and marks both-incomplete zones invalid. The count rules
    reject any frame missing more than one mask in either pass, then prefer
    the smaller sum; equal nonzero sums are invalid. Without missing masks,
    the count-only rule chooses forward and the temporal rule compares the
    minimum finite assignment quality in each pass; ties are invalid.

    ``temporal_quality[d, t]`` compares frames t and t+1. A zone [start, end]
    uses transitions starting in that interval, excluding the recording's
    final frame, which has no successor. No finite quality means invalid.

    >>> from bivosia.analysis.zones_of_disagreement import default_source
    >>> default_source(0, 2, [1, 0, 0], [0, 1, 1], direction_method='missing_masks')
    'invalid'
    >>> default_source(0, 2, [1, 0, 0], [0, 1, 1], direction_method='missing_mask_counts')
    'forward'
    >>> default_source(0, 1, [2, 0], [0, 0])
    'invalid'
    >>> default_source(0, 1, [1, 0], [0, 1])
    'invalid'
    >>> default_source(0, 1, [0, 0], [0, 0], temporal_quality=[[0.4], [0.8]])
    'reverse'
    >>> default_source(0, 1, [0, 0], [0, 0], temporal_quality=[[0.8], [0.8]])
    'invalid'
    """
    if direction_method not in AUTO_DIRECTION_METHODS:
        raise ValueError(f'Unknown automatic direction rule: {direction_method}')
    forward, reverse = np.asarray(missing_forward), np.asarray(missing_reverse)
    if (forward.ndim != 1 or reverse.shape != forward.shape
            or not 0 <= start <= end < len(forward)):
        raise ValueError('Zone bounds and missing-mask counts are incompatible')
    counts = np.array([forward[start:end + 1], reverse[start:end + 1]])
    if np.any(~np.isfinite(counts)) or np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError('Missing-mask counts must be nonnegative integers')
    if direction_method == 'missing_masks':
        events = frame_event_signals(forward_missing_signal=counts[0], reverse_missing_signal=counts[1])
        summary = zone_frame_summary(range(end - start + 1), events=events)
        forward_missing, reverse_missing = summary['missing_in_forward'], summary['missing_in_reverse']
        return 'invalid' if forward_missing and reverse_missing else 'reverse' if forward_missing else 'forward'
    if np.any(counts > 1):
        return 'invalid'
    totals = counts.sum(axis=1)
    if np.any(totals):
        return 'invalid' if totals[0] == totals[1] else 'forward' if totals[0] < totals[1] else 'reverse'
    if direction_method == 'missing_mask_counts':
        return 'forward'
    if temporal_quality is None:
        raise ValueError('Temporal quality is unavailable. Run Workflow → Open Masks → Run '
                         'before using the temporal-quality direction rule.')
    quality = np.asarray(temporal_quality)
    if quality.shape != (2, len(forward) - 1):
        raise ValueError('Temporal quality must have shape (2, frames - 1)')
    frames = range(start, min(end + 1, quality.shape[1]))
    forward_quality, reverse_quality = (minimum_frame_value(values, frames) for values in quality)
    if forward_quality is None or reverse_quality is None or forward_quality == reverse_quality:
        return 'invalid'
    return 'forward' if forward_quality > reverse_quality else 'reverse'


def new_zone(start, end, missing_forward, missing_reverse, *, temporal_quality=None,
             direction_method=ANALYSIS_DEFAULTS['zones_of_disagreement']['direction_method']):
    return ZodAnnotation(uuid4().hex, int(start), int(end),
                        default_source(start, end, missing_forward, missing_reverse,
                                       temporal_quality=temporal_quality, direction_method=direction_method))


def edit_zones(zones, zone_id, operation, *, frame_count, missing_forward, missing_reverse,
               temporal_quality=None,
               direction_method=ANALYSIS_DEFAULTS['zones_of_disagreement']['direction_method'],
               start=None, end=None, frame=None, source=None, side="right"):
    """Apply an interval or source edit and validate the resulting zones."""
    result = validate_zones(zones, frame_count)
    if operation == "add":
        result.append(new_zone(start, end, missing_forward, missing_reverse,
                               temporal_quality=temporal_quality, direction_method=direction_method))
        return validate_zones(result, frame_count)
    index = next((i for i, z in enumerate(result) if z.id == zone_id), None)
    if index is None:
        raise ValueError("Select a zone first")
    zone = result[index]

    def changed(z, **fields):
        if fields.get('start', z.start) != z.start or fields.get('end', z.end) != z.end:
            # Whole-zone judgments and counts cannot be assigned to a new interval.
            fields.update(forward_valid=None, reverse_valid=None, forward_swap=0, reverse_swap=0)
        z = replace(z, **fields, status="needs_review" if z.status in {"reviewed", "automatic"} else "draft")
        if not z.source_manual:
            z = replace(z, source=default_source(z.start, z.end, missing_forward, missing_reverse,
                                                temporal_quality=temporal_quality, direction_method=direction_method))
        return z

    if operation == "delete":
        result.pop(index)
    elif operation == "bounds":
        start, end = int(start), int(end)
        # Move the edges of adjacent zones together.
        if index and result[index - 1].end == zone.start - 1 and start != zone.start:
            result[index - 1] = changed(result[index - 1], end=start - 1)
        if index + 1 < len(result) and result[index + 1].start == zone.end + 1 and end != zone.end:
            result[index + 1] = changed(result[index + 1], start=end + 1)
        result[index] = changed(zone, start=start, end=end)
    elif operation == "split":
        if not zone.start <= int(frame) < zone.end:
            raise ValueError("Split frame must be inside the zone, before its last frame")
        result[index:index + 1] = [changed(zone, end=int(frame)),
                                  changed(zone, id=uuid4().hex, start=int(frame) + 1)]
    elif operation == "join":
        if side not in ("left", "right"):
            raise ValueError("Choose the left or right neighbor")
        neighbor = index + (-1 if side == "left" else 1)
        if not 0 <= neighbor < len(result):
            raise ValueError(f"There is no {side} zone to join")
        if source not in SOURCES:
            raise ValueError("Choose the merged zone's source explicitly")
        first, last = sorted((index, neighbor))
        result[first:last + 1] = [changed(zone, start=result[first].start, end=result[last].end,
                                        source=source, source_manual=True)]
    elif operation == "source":
        if source not in SOURCES:
            raise ValueError("Choose forward, reverse, or invalid")
        result[index] = changed(zone, source=source, source_manual=True)
    elif operation == "review":
        result[index] = replace(zone, status="reviewed")
    else:
        raise ValueError(f"Unknown zone operation: {operation}")
    return validate_zones(result, frame_count)


@dataclass(frozen=True)
class Segment:
    id: str
    start: int
    end: int
    source: str


def segments(zones, frame_count):
    """Partition the entire video; outside zones always select forward masks."""
    result, start, previous = [], 0, "start"
    for z in validate_zones(zones, frame_count):
        if start < z.start:
            result.append(Segment(f"gap:{previous}:{z.id}", start, z.start - 1, "forward"))
        result.append(Segment(z.id, z.start, z.end, z.source))
        start, previous = z.end + 1, z.id
    if start < frame_count:
        result.append(Segment(f"gap:{previous}:end", start, frame_count - 1, "forward"))
    return result


def preferred_sources(zones, frame_count):
    """Return per-frame source codes: 0=forward, 1=reverse, 2=invalid."""
    sources = np.empty(frame_count, dtype=np.uint8)
    for segment in segments(zones, frame_count):
        sources[segment.start:segment.end + 1] = SOURCES.index(segment.source)
    return sources


def detection_parameters(seed, recover, gap=ANALYSIS_DEFAULTS['zones_of_disagreement']['theta_gap'], *,
                         direction_method=ANALYSIS_DEFAULTS['zones_of_disagreement']['direction_method']):
    """Describe the detector parameters saved with a revision annotation.

    >>> from bivosia.analysis.zones_of_disagreement import detection_parameters
    >>> detection_parameters(0.6, 0.9)['seed']
    0.6
    """
    if direction_method not in AUTO_DIRECTION_METHODS:
        raise ValueError(f'Unknown automatic direction rule: {direction_method}')
    return {"seed": float(seed), "recover": float(recover), "gap": int(gap),
            "detector": "local-margin-hungarian-v1", "direction_method": direction_method}
