"""Shared playback setup, legends and markers for key-frame views."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from bivosia.ui.colors import KEY_FRAME_COLORS
from bivosia.ui.plot_models import CursorSpec, LegendGroup, RowSpec


def configure_key_frame_view(recording, player, timeline, previews):
    session = recording.session
    player.configure(session.frame_count, recording.videos.source_fps)
    player.set_loop(*session.crop)
    timeline.frame_count = session.frame_count
    timeline.reset_zoom()
    previews.set_video(recording.videos)
    previews.set_session(session)


def key_frame_row(label):
    legend = LegendGroup(items=tuple((label, KEY_FRAME_COLORS[key]) for key, label in (
        ('proposed', 'Unlabeled'), ('labeled', 'Labeled'), ('boundary', 'Boundary'))))
    return RowSpec('key_frames', label, legend=(legend,))


def key_frame_cursors(session, frame, *, editable=False):
    cursors = [CursorSpec('playhead', frame)]
    for position in session.frames:
        boundary, labeled = position in session.crop, position in session.masks
        cursors.append(CursorSpec(f'key_frame:{position}', position,
            KEY_FRAME_COLORS['labeled' if labeled else 'proposed'], line=False, level=1,
            draggable=editable and not (boundary or labeled), selectable=editable,
            outline=KEY_FRAME_COLORS['boundary'] if boundary else None))
    return cursors
