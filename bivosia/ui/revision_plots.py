"""Revision-tool row configuration for the shared, generic timeline widgets."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from bivosia.ui.plot_models import IntervalBand, IntervalSpec, IntervalTrack, LegendGroup, MarkerSpec, RowSpec
from bivosia.ui.plot_styles import HAND_BASE_COLORS
from bivosia.ui.viewer_plot_config import linked_zone_row, viewer_zone_rows
from bivosia.analysis.zones_of_disagreement import frame_event_intervals


SOURCE_COLORS = {
    "forward": HAND_BASE_COLORS["FORWARD"],
    "reverse": HAND_BASE_COLORS["REVERSE"],
    "invalid": "#9b233c",
}
STATUS_COLORS = {"unseen": "#808895", "automatic": "#808895", "draft": "#b48cff", "reviewed": "#45c47c", "needs_review": "#ee7058"}


def annotation_row(zones, label='Annotations', **interactions):
    """Describe saved directions and review states with the shared annotation colors."""
    return RowSpec('annotations', label, (IntervalTrack('zones', '', tuple(
        IntervalSpec(z.id, z.start, z.end, str(i + 1), SOURCE_COLORS[z.source],
                     (IntervalBand(STATUS_COLORS[z.status], .82, .18),), z.id)
        for i, z in enumerate(zones))),),
        legend=(LegendGroup('direction', tuple((label, SOURCE_COLORS[source]) for source, label in
                                               (('forward', 'F'), ('reverse', 'R'), ('invalid', '∅')))),
                LegendGroup('status', tuple((name.replace('_', ' '), color) for name, color in STATUS_COLORS.items()))),
        **interactions)


def editor_rows(data, detected, session=None, *, on_zone=None, on_select=None, on_edit=None,
                on_boundary=None, editable=True, overview=False, regions=(), links=None):
    intervals = session.automatic if session else detected.final_zods
    events = frame_event_intervals(data.missing, data.missing_forward, data.missing_reverse,
                                  None, data.evidence.switch_frames)
    events['Combined seeds'] = detected.initial_zods
    events['Auto zones'] = intervals
    zod = next(row for row in viewer_zone_rows(events) if row.key == 'ZoD')
    annotation_label = 'Annotations' if session and session.header.get('mode') == 'automatic' else 'Hand'
    manual = annotation_row(session.zones if session else (), annotation_label,
        selectable=session is not None, on_select=on_select,
        editable=session is not None and editable and not overview,
        edit_requires_alt=True,
        draggable=session is not None and editable and not overview, on_double_click=on_zone, on_edit=on_edit,
        markers=tuple(MarkerSpec(f"{job.id}:{frame}", frame, payload=job.id)
                      for job in (session.jobs if session else ())
                      for frame in sorted({job.left.end + .5, job.right.start - .5})),
        on_marker_double_click=on_boundary,
        show_interval_labels=not overview)
    rows = [zod, manual]
    if overview and regions:
        rows.append(linked_zone_row(regions, links=links))
    return rows
