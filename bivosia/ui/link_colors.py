"""Display label selection and cached identity previews for the editor."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from bivosia.analysis.linking import inverse, permutation, prepare_links
from bivosia.ui.colors import component_colors, identity_colors


def current_link_labels(mapping, n):
    """Use left raw identities on both sides of the displayed assignment."""
    return {"left": tuple(range(n)), "right": inverse(permutation(mapping, n))}


def aligned_labels(linker, direction, source, frame, labels, seed):
    conversion, _ = linker.conversion(direction, source, frame, seed)
    return tuple(labels[i] for i in conversion)


def global_labels(linker, preview, direction, frame, seed):
    region = next(s for s in preview.partition if s.start <= frame <= s.end)
    if region.source == "invalid":
        return tuple(range(linker.n)), None
    labels = aligned_labels(linker, direction, region.source, frame, preview.maps[region.id], seed)
    return labels, preview.components[region.id]


def global_label_colors(linker, preview, direction, frame, seed):
    """Color local mask IDs through the editor's connected identity components."""
    labels, component = global_labels(linker, preview, direction, frame, seed)
    colors = (identity_colors(linker.n, .27) if component is None
              else component_colors(linker.n, component))
    return [colors[i] for i in labels]


class LinkColors:
    def __init__(self, data):
        self.data = data
        self._base_key = self._preview_key = None
        self._base = self._preview = None

    def links(self, session):
        """Cache automatic/approved boundary results without changing annotations."""
        base_key = (session, session.revision)
        if base_key != self._base_key:
            self._base = prepare_links(session, self.data.linking.boundary_match)
            self._base_key = base_key
        return self._base

    def preview(self, session, job, draft, seed):
        """Cache global preview construction; never modify annotation records."""
        base = self.links(session)
        current = None if job is None or draft is None else (
            job.id, *(draft[f"{side}_{field}"] for side in ("left", "right") for field in ("source", "frame")),
            tuple(draft["mapping"]))
        key = (self._base_key, current, seed)
        if key != self._preview_key:
            mapping = self.data.linking.canonical_mapping(job, draft, seed)[0] if current else None
            self._preview = base.resolve(job.id if current else None, mapping)
            self._preview_key = key
        return self._preview


def boundary_status(record, automatic=None):
    """Display automatic boundary status with its selected-assignment quality."""
    if automatic is not None:
        return f'Auto · quality {automatic.quality:.3f}'
    return record['status']
