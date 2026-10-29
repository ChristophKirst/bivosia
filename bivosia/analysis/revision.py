"""Inclusive zone edits and default mask directions.

Examples
--------
>>> from bivosia.analysis.revision import new_zone
>>> import numpy as np
>>> missing_forward = np.array([False, True, False])
>>> missing_reverse = np.zeros(3, dtype=bool)
>>> new_zone(0, 2, missing_forward, missing_reverse).source
'reverse'
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from dataclasses import asdict, dataclass, replace
from uuid import uuid4
import hashlib
import json

import numpy as np


SOURCES = ("forward", "reverse", "invalid")


def revision_fingerprint(header, zones, links, link_frames):
    """Identify the annotation state used to validate a linked-mask export."""
    payload = {"header": header, "zones": {zone['id']: zone for zone in zones},
               "links": links, "link_frames": {str(frame): frame for frame in link_frames}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class RevisionZone:
    """Mask choice and optional directional hand judgments for one interval.

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
        if zone.source not in SOURCES or zone.status not in {"unseen", "draft", "reviewed", "needs_review"}:
            raise ValueError("Invalid zone source or review status")
        for value in (zone.forward_valid, zone.reverse_valid):
            if value is not None and not isinstance(value, bool):
                raise ValueError('Hand validity must be true, false or unknown')
        for value in (zone.forward_swap, zone.reverse_swap):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError('Hand swap counts must be nonnegative integers or omitted')
        previous_end = zone.end
    return ordered


def default_source(start, end, missing_forward, missing_reverse):
    """Both means each direction is missing somewhere in this entire zone."""
    forward = bool(np.any(missing_forward[start:end + 1]))
    reverse = bool(np.any(missing_reverse[start:end + 1]))
    return "invalid" if forward and reverse else "reverse" if forward else "forward"


def new_zone(start, end, missing_forward, missing_reverse):
    return RevisionZone(uuid4().hex, int(start), int(end),
                        default_source(start, end, missing_forward, missing_reverse))


def edit_zones(zones, zone_id, operation, *, frame_count, missing_forward, missing_reverse,
               start=None, end=None, frame=None, source=None, side="right"):
    """Apply an interval or source edit and validate the resulting zones."""
    result = validate_zones(zones, frame_count)
    if operation == "add":
        result.append(new_zone(start, end, missing_forward, missing_reverse))
        return validate_zones(result, frame_count)
    index = next((i for i, z in enumerate(result) if z.id == zone_id), None)
    if index is None:
        raise ValueError("Select a zone first")
    zone = result[index]

    def changed(z, **fields):
        if fields.get('start', z.start) != z.start or fields.get('end', z.end) != z.end:
            # Whole-zone judgments and counts cannot be assigned to a new interval.
            fields.update(forward_valid=None, reverse_valid=None, forward_swap=0, reverse_swap=0)
        z = replace(z, **fields, status="needs_review" if z.status == "reviewed" else "draft")
        if not z.source_manual:
            z = replace(z, source=default_source(z.start, z.end, missing_forward, missing_reverse))
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
