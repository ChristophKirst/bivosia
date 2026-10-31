"""Annotation display records and missing-mask stripes for the shared viewer."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from dataclasses import dataclass, replace

import numpy as np

from bivosia.analysis.zones_of_disagreement import frame_event_signals, zone_frame_summary

@dataclass(frozen=True)
class ZodView:
    """A ZoD's display frames, labels and directional warning flags."""

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
    return ZodView(zone.id, tuple(range(zone.start, zone.end + 1)), details=details,
                preferred_pass=preferred, forward_valid=zone.forward_valid, reverse_valid=zone.reverse_valid,
                swap_in_forward=(zone.forward_swap or 0) > 0,
                swap_in_reverse=(zone.reverse_swap or 0) > 0)


def annotate_zone_missing(
    zones: list[ZodView], missing_signal: np.ndarray | None,
    forward_missing_signal: np.ndarray | None = None,
    reverse_missing_signal: np.ndarray | None = None,
    missing_direction_known: np.ndarray | None = None,
) -> list[ZodView]:
    """Attach shared analysis summaries to annotation display records."""
    if missing_signal is None:
        return zones
    events = frame_event_signals(missing_signal, forward_missing_signal,
                                reverse_missing_signal, missing_direction_known)
    result = []
    for zone in zones:
        summary = zone_frame_summary(zone.frames, events=events)
        result.append(replace(zone, **{key: summary[key] for key in (
            'missing_in_forward', 'missing_in_reverse', 'missing_direction_unknown')}))
    return result
