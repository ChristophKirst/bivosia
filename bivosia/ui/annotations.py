"""Annotation display records and missing-mask stripes for the shared viewer."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Zone:
    name: str
    frames: tuple[int, ...]
    details: str = ""
    preferred_pass: str = ""
    swap_in_forward: bool = False
    swap_in_reverse: bool = False
    missing_in_forward: bool = False
    missing_in_reverse: bool = False
    missing_direction_unknown: bool = False
    swap_direction_unknown: bool = False
    forward_valid: bool | None = None
    reverse_valid: bool | None = None


def revision_zone(zone):
    """Translate a saved zone's directional hand annotations for display."""
    preferred = {'forward': 'FORWARD', 'reverse': 'REVERSE', 'invalid': 'NONE'}[zone.source]
    def valid_label(value):
        return 'none' if value is None else 'yes' if value else 'no'
    details = (f'Valid F: {valid_label(zone.forward_valid)}, R: {valid_label(zone.reverse_valid)} · '
               f'Preferred: {preferred.lower()} · '
               f'Swaps F: {zone.forward_swap or 0}, R: {zone.reverse_swap or 0}')
    return Zone(zone.id, tuple(range(zone.start, zone.end + 1)), details=details,
                preferred_pass=preferred, forward_valid=zone.forward_valid, reverse_valid=zone.reverse_valid,
                swap_in_forward=(zone.forward_swap or 0) > 0,
                swap_in_reverse=(zone.reverse_swap or 0) > 0)


def annotate_zone_missing(
    zones: list[Zone], missing_signal: np.ndarray | None,
    forward_missing_signal: np.ndarray | None = None,
    reverse_missing_signal: np.ndarray | None = None,
    missing_direction_known: np.ndarray | None = None,
) -> list[Zone]:
    """Attach frame-level missing-mask evidence to each hand annotation."""
    if missing_signal is None:
        return zones
    missing = np.asarray(missing_signal, dtype=bool)
    forward = (np.zeros_like(missing) if forward_missing_signal is None
               else np.asarray(forward_missing_signal) > 0)
    reverse = (np.zeros_like(missing) if reverse_missing_signal is None
               else np.asarray(reverse_missing_signal) > 0)
    known = (np.asarray(missing_direction_known, dtype=bool)
             if missing_direction_known is not None else
             np.full_like(missing, forward_missing_signal is not None and reverse_missing_signal is not None))
    if any(value.shape != missing.shape for value in (forward, reverse, known)):
        raise ValueError("Zone missing-mask signals must have the same length")
    result = []
    for zone in zones:
        frames = np.asarray(zone.frames, dtype=np.int64)
        marked = missing[frames]
        result.append(replace(
            zone,
            missing_in_forward=bool(np.any(marked & known[frames] & forward[frames])),
            missing_in_reverse=bool(np.any(marked & known[frames] & reverse[frames])),
            missing_direction_unknown=bool(np.any(marked & (~known[frames] | ~(forward[frames] | reverse[frames])))),
        ))
    return result
