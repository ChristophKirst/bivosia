"""Application-specific plot configuration, shared by the existing viewers.

Only this layer knows what hand annotations, seeds and forward/reverse mean.
The plotting widgets consume the resulting generic descriptions.
"""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import numpy as np


from bivosia.analysis.zones_of_disagreement import boolean_runs, frame_event_intervals
from bivosia.analysis.keypoints import keypoint_identity_events
from bivosia.ui.colors import linked_zone_colors
from bivosia.ui.plot_models import IntervalBand, IntervalSpec, IntervalTrack, LegendGroup, MarkerSpec, ReferenceLine, RowSpec, SeriesSpec
from bivosia.ui.plot_styles import (
    HAND_BASE_COLORS, HAND_SWAP_COLORS, HAND_UNKNOWN_COLOR, HAND_BACKGROUND_COLOR,
    METRIC_COLORS, MISSING_COLORS, KEYPOINT_SWAP_BORDER_COLOR,
    REGION_COLORS,
)
from bivosia.ui.plot_data import feature_curves, feature_ranges


ZONE_ROW_LABELS = {
    "Hand labels": "Hand", "ZoD": "ZoD", "Keypoints": "Keypoints",
}

KEYPOINT_TRACKS = (
    {'Swap': 'swap zones', 'Swap over invalid zone': 'swaps over invalid zone'},
    {'Partial resolved swap': 'partial resolved swap zones',
     'Partial swap over invalid zone': 'partial swaps over invalid zone'},
    {'No overlap': 'no overlap'},
    {'Tie': 'ties', 'Unassigned': 'unassigned',
     'Tie + unassigned': 'ties + unassigned'},
    {'Missing skeleton': 'missing skeleton'},
)


def keypoint_alignment_tracks(keypoints, links, *, point_counts):
    """Display each keypoint skeleton's assignment events against linked masks."""
    events = keypoint_identity_events(keypoints.alignment, keypoints.source, keypoints.scores,
                                     keypoints.mask_present, links, point_counts=point_counts)
    tracks = {}
    for group in KEYPOINT_TRACKS[:2]:
        (name, zone_key), (invalid_name, invalid_key) = group.items()
        zones, over_invalid = events[zone_key], events[invalid_key]
        tracks[name] = [tuple(zone) for zone in zones[~over_invalid]]
        tracks[invalid_name] = [tuple(zone) for zone in zones[over_invalid]]
    tracks.update({name: list(boolean_runs(events[key]))
                   for group in KEYPOINT_TRACKS[2:] for name, key in group.items()})
    return tracks


def linked_zone_row(regions, *, links=None, visible=True):
    """Give each connected component its own blue shade."""
    components = sorted({component for _, _, component in regions})
    colors = dict(zip(components, linked_zone_colors(len(components))))
    intervals = tuple(
        IntervalSpec(f'{start}:{end}', start, end, str(component + 1),
                     colors[component].name())
        for start, end, component in regions
    )
    markers = () if links is None else tuple(
        MarkerSpec(job.id, job.right.start - .5, color=REGION_COLORS['Auto zones'])
        for job in links.jobs if job.id in links.automatic
    )
    return RowSpec('connections', 'Links', (IntervalTrack('regions', '', intervals),),
                   show_interval_labels=True, legend=(), double_click=False, visible=visible, markers=markers)


def _track_lane(label: str) -> str:
    """Return the visual lane shared by related detailed seed tracks."""
    if label in ("Missing forward only", "Missing reverse only", "Missing both", "Missing direction unknown",
                 "Switch frames", "Missing mask + switch", "Combined seeds", "Auto zones"):
        return "ZoD"
    if any(label in group for group in KEYPOINT_TRACKS):
        return 'Keypoints'
    return label


def _track_lanes(regions):
    """Keep the shared zone-row order independent of caller insertion order."""
    lanes = list(dict.fromkeys(_track_lane(label) for label in regions))
    if "Hand labels" not in lanes:
        lanes.append("Hand labels")
    order = (
        "Hand labels",
        "ZoD", 'Keypoints',
    )
    return [name for name in order if name in lanes] + [
        name for name in lanes if name not in order
    ]


def compose_zone_tracks(
    regions: dict[str, list[tuple[int, int]]] | None = None,
    *,
    missing_signal: np.ndarray | None = None,
    forward_missing_signal: np.ndarray | None = None,
    reverse_missing_signal: np.ndarray | None = None,
    missing_direction_known: np.ndarray | None = None,
    switch_frames: np.ndarray | None = None,
    seed_intervals: Iterable[tuple[int, int]] | None = None,
    auto_intervals: Iterable[tuple[int, int]] | None = None,
) -> dict[str, list[tuple[int, int]]]:
    """Build annotation, failure, seed and auto-zone tracks."""
    source = dict(regions or {})
    source["Hand labels"] = []

    source.update(frame_event_intervals(
        missing_signal, forward_missing_signal, reverse_missing_signal,
        missing_direction_known, switch_frames,
    ))
    if seed_intervals is not None:
        source["Combined seeds"] = list(seed_intervals)
    if auto_intervals is not None:
        source["Auto zones"] = list(auto_intervals)
    for label in (
        "Switch frames",
        "Combined seeds", "Auto zones",
    ):
        source.setdefault(label, [])
    ordered = (
        "Hand labels",
        "Missing forward only", "Missing reverse only", "Missing both",
        "Missing direction unknown", "Switch frames",
        "Missing mask + switch", "Combined seeds", "Auto zones",
    )
    return {label: source[label] for label in ordered if label in source}


def annotation_legend(zones):
    direction = LegendGroup("direction", tuple((short, HAND_BASE_COLORS[key]) for key, short in
                                               (("FORWARD", "F"), ("REVERSE", "R"), ("UNKNOWN", "?"))))
    groups = [direction]
    missing = tuple((short, MISSING_COLORS[key]) for key, short in (
        ("FORWARD", "F"), ("REVERSE", "R"), ("BOTH", "B")
    ))
    if any(zone.missing_direction_unknown for zone in zones):
        missing += (("?", MISSING_COLORS["UNKNOWN"]),)
    groups.append(LegendGroup("missing", missing))
    if any(
        zone.swap_in_forward or zone.swap_in_reverse or zone.swap_direction_unknown
        for zone in zones
    ):
        items = tuple((short, HAND_SWAP_COLORS[key]) for key, short in (
            ('FORWARD', 'F'), ('REVERSE', 'R'), ('BOTH', 'B')))
        if any(zone.swap_direction_unknown for zone in zones):
            items += (('?', HAND_UNKNOWN_COLOR),)
        groups.append(LegendGroup('swap', items))
    return tuple(groups)


def annotation_row(key, label, zones, *, on_select=None, show_names=True):
    """Translate annotation metadata to generic colored bands and payloads.

    >>> from bivosia.ui.plot_styles import HAND_SWAP_COLORS
    >>> from bivosia.ui.viewer_plot_config import annotation_row
    >>> from bivosia.ui.annotations import Zone
    >>> zone = Zone('1', (3, 4), swap_in_reverse=True)
    >>> row = annotation_row('Hand', 'Hand', [zone])
    >>> row.tracks[0].intervals[0].bands[-1].color == HAND_SWAP_COLORS['REVERSE']
    True
    """
    if any(zone.forward_valid is not None or zone.reverse_valid is not None for zone in zones):
        return directional_annotation_row(key, label, zones, on_select=on_select, show_names=show_names)
    intervals = []
    for index, zone in enumerate(zones):
        if not zone.frames:
            continue
        bands = []
        direction = (
            "BOTH" if zone.missing_in_forward and zone.missing_in_reverse
            else "FORWARD" if zone.missing_in_forward
            else "REVERSE" if zone.missing_in_reverse
            else "UNKNOWN" if zone.missing_direction_unknown
            else None
        )
        if direction is not None:
            bands.append(IntervalBand(MISSING_COLORS[direction], 1/3, 1/3))
        if zone.swap_in_forward or zone.swap_in_reverse or zone.swap_direction_unknown:
            swap_direction = (
                'BOTH' if zone.swap_in_forward and zone.swap_in_reverse
                else 'FORWARD' if zone.swap_in_forward
                else 'REVERSE' if zone.swap_in_reverse
                else 'UNKNOWN')
            bands.append(IntervalBand(HAND_SWAP_COLORS.get(swap_direction, HAND_UNKNOWN_COLOR),
                                      2/3, 1/3))
        intervals.append(IntervalSpec(
            str(index), zone.frames[0], zone.frames[-1],
            label=str(index + 1),
            color=HAND_BASE_COLORS.get(zone.preferred_pass, HAND_BASE_COLORS["UNKNOWN"]),
            bands=tuple(bands), payload=index,
        ))
    return RowSpec(key, label, (IntervalTrack(key, label, tuple(intervals), HAND_BASE_COLORS["UNKNOWN"]),),
                   selectable=on_select is not None, on_select=on_select,
                   show_interval_labels=show_names, legend=annotation_legend(zones))


def directional_annotation_row(key, label, zones, *, on_select=None, show_names=True):
    """Show directional validity, preferred masks and combined swaps in four strips."""
    intervals = []
    for index, zone in enumerate(zones):
        if not zone.frames:
            continue
        swap = (
            'BOTH' if zone.swap_in_forward and zone.swap_in_reverse
            else 'FORWARD' if zone.swap_in_forward
            else 'REVERSE' if zone.swap_in_reverse
            else None
        )
        colors = [
            HAND_BASE_COLORS[direction] if valid is True else
            HAND_BASE_COLORS['UNKNOWN'] if valid is False else HAND_UNKNOWN_COLOR
            for direction, valid in (('FORWARD', zone.forward_valid), ('REVERSE', zone.reverse_valid))
        ]
        colors.append(HAND_BASE_COLORS.get(zone.preferred_pass, HAND_UNKNOWN_COLOR))
        colors.append(HAND_SWAP_COLORS.get(swap, HAND_BACKGROUND_COLOR))
        intervals.append(IntervalSpec(
            str(index), zone.frames[0], zone.frames[-1], label=str(index + 1),
            color=HAND_BACKGROUND_COLOR,
            bands=tuple(IntervalBand(color, offset / 4, 1/4) for offset, color in enumerate(colors)),
            payload=index))
    directions = (('F', HAND_BASE_COLORS['FORWARD']), ('R', HAND_BASE_COLORS['REVERSE']))
    validity = directions + (('invalid', HAND_BASE_COLORS['UNKNOWN']),)
    if any(zone.forward_valid is None or zone.reverse_valid is None for zone in zones):
        validity += (('none', HAND_UNKNOWN_COLOR),)
    legend = (LegendGroup('valid', validity),
              LegendGroup('preferred', directions + (('none', HAND_UNKNOWN_COLOR),)),
              LegendGroup('swap', (('F', HAND_SWAP_COLORS['FORWARD']),
                                   ('R', HAND_SWAP_COLORS['REVERSE']),
                                   ('B', HAND_SWAP_COLORS['BOTH']))))
    return RowSpec(key, label, (IntervalTrack(key, label, tuple(intervals)),),
                   selectable=on_select is not None, on_select=on_select,
                   show_interval_labels=show_names, legend=legend)


def frame_event_legend(regions):
    """Shared keys for missing-mask and switch-frame colors."""
    missing = tuple((short, REGION_COLORS[name]) for name, short in (
        ("Missing forward only", "F"), ("Missing reverse only", "R"), ("Missing both", "B")))
    if regions.get("Missing direction unknown"):
        missing += (("?", REGION_COLORS["Missing direction unknown"]),)
    items = missing + (('switch', REGION_COLORS['Switch frames']),)
    if regions.get("Missing mask + switch"):
        items += (('mask+switch', REGION_COLORS['Missing mask + switch']),)
    return (LegendGroup('mask', items),)


def viewer_zone_rows(regions, zones=(), *, callbacks=None, show_names=True, annotation_label='Hand'):
    """Configure the standard annotation and detection rows."""
    callbacks = callbacks or {}
    rows = []
    for lane in _track_lanes(regions):
        label = ZONE_ROW_LABELS.get(lane, lane)
        if lane == "Hand labels":
            row = annotation_row(lane, annotation_label, zones, on_select=callbacks.get(lane), show_names=show_names)
            rows.append(row)
            continue
        legend_labels = {
            "Missing forward only": "missing F", "Missing reverse only": "missing R",
            "Missing both": "missing both", "Missing direction unknown": "missing ?",
            "Switch frames": "switch", "Missing mask + switch": "missing + switch",
            "Combined seeds": "", "Auto zones": "",
            "Partial swap over invalid zone": "Swap over invalid zone",
        }
        tracks = []
        for name, intervals in regions.items():
            if _track_lane(name) != lane:
                continue
            color = REGION_COLORS.get(name, '#aaaaaa')
            bands = ()
            if lane == 'ZoD' and name != 'Auto zones':
                offset = 2/3 if name == 'Combined seeds' else 1/3
                bands = (IntervalBand(color, offset, 1/3),)
            elif lane == 'Keypoints':
                height = 1 / len(KEYPOINT_TRACKS)
                index = next(index for index, group in enumerate(KEYPOINT_TRACKS) if name in group)
                # Swap bars span frame centers with a 0.05-frame gap at each end.
                bands = (IntervalBand(color, index * height, height,
                                      border_color=KEYPOINT_SWAP_BORDER_COLOR if index < 2 else None,
                                      frame_inset=.55 if index < 2 else 0.),)
            values = tuple(IntervalSpec(f'{start}:{end}', start, end,
                                        color='transparent' if bands else None, bands=bands)
                           for start, end in intervals)
            tracks.append(IntervalTrack(name, legend_labels.get(name, name), values, color))
        legend = None
        if lane == 'ZoD':
            tracks.sort(key=lambda track: track.key != 'Auto zones')
            items = tuple(item for group in frame_event_legend(regions) for item in group.items)
            legend = (LegendGroup('Auto', (('', REGION_COLORS['Auto zones']),)),
                      LegendGroup('Fail: mask', items),
                      LegendGroup('Seed', (('', REGION_COLORS['Combined seeds']),)))
        elif lane == 'Keypoints':
            legend = tuple(LegendGroup(items=tuple((legend_labels.get(name, name), REGION_COLORS[name])
                                                  for name in group))
                           for group in KEYPOINT_TRACKS)
        rows.append(RowSpec(lane, label, tuple(tracks), selectable=lane in callbacks,
                            on_select=callbacks.get(lane), legend=legend))
    return rows


def configure_quality_plot(graph, arrays, labels, *, left_axis_width=0, reference_labels_visible=False, colors=None):
    colors = METRIC_COLORS if colors is None else colors
    available = {key: label for key, label in labels.items() if key in arrays}
    graph.set_series([
        SeriesSpec(key, label, arrays[key], color=colors.get(key, "#b8bec7"))
        for key, label in available.items()
    ])
    graph.reference_labels_visible = reference_labels_visible
    graph.fit_shared_y_range(include=(0., 1.))
    graph.set_left_margin(max(left_axis_width, graph.required_left_margin()))
    graph.setToolTip("Scroll / drag: time. Ctrl+scroll: zoom y; Ctrl+drag: pan y; Ctrl+right-click: reset y.")
    return available


class FeaturePlotController:
    """Configure generic traces from separately calculated scientific measures."""

    def __init__(self, graph, sparse, dense, labels):
        self.graph, self.labels = graph, labels
        self.curves = feature_curves(sparse, dense, labels)
        self._ranges = {mode: feature_ranges(self.curves, labels, mode) for mode in ("p95", "3sigma", "6sigma", "raw_max")}
        initial = self._ranges["p95"]
        self.specs = {
            (direction, key): SeriesSpec(
                f"{direction}:{key}", f"{label} ({direction})", curve.values,
                color=METRIC_COLORS.get(key, "#b8bec7"), frames=curve.frames,
                minimum=curve.minimum, maximum=curve.maximum,
                value_range=initial.get((direction, key)), dashed=direction == "REVERSE",
                temporal_extrema=direction == "DIFFERENCE",
                width=1.0,
            )
            for (direction, key), curve in self.curves.items() for label in (labels[key],)
        }
        graph.set_series(self.specs.values())
        self.selected = list(labels)
        self.difference = False
        self.scale = "p95"
        self.update()

    def update(self):
        directions = ("DIFFERENCE",) if self.difference else ("FORWARD", "REVERSE")
        self.graph.set_visible_series(
            spec.key for (direction, key), spec in self.specs.items()
            if direction in directions and key in self.selected
        )
        self.graph.metric_ranges = {
            spec.key: self._ranges[self.scale].get(pair, (0., 1.))
            for pair, spec in self.specs.items()
        }
        self.graph.set_reference_lines([
            ReferenceLine("0 · F = R", 0., "#8b949e", series_key=key)
            for key in self.graph.visible_series
        ] if self.difference else [])
        self.graph.invalidate()

    def set_selected(self, selected):
        self.selected = selected
        self.update()

    def set_difference(self, enabled):
        self.difference = enabled
        self.update()

    def set_scale(self, scale):
        self.scale = scale
        self.update()
