"""Stream validated globally relabeled masks without changing source masks.

Examples
--------
With an existing dataset and a reviewed annotation ready for export:

>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.dataset.session import RevisionSession
>>> from bivosia.processing.export import export_masks
>>> from bivosia.processing.revision import RevisionData
>>> dataset = Dataset("/path/to/dataset")
>>> data = RevisionData(dataset)  # doctest: +SKIP
>>> try:  # doctest: +SKIP
...     session = RevisionSession.load(dataset.annotation, data.missing_forward, data.missing_reverse, data.n)
...     result = export_masks(session, dataset, dataset.linked_masks(session.path))
... finally:
...     data.close()
"""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import logging
from contextlib import closing

import numpy as np

from bivosia.analysis.linking import automatic_zone_frames, match_masks, prepare_links, linked_frame, validation_errors
from bivosia.dataset.masks import linked_mask_arrays, mask_pair, write_linked_frame
from bivosia.dataset.containers import container_output
from bivosia.dataset.jsonio import read_json
from bivosia.analysis.zones_of_disagreement import revision_fingerprint
from bivosia.processing.log import Progress


logger = logging.getLogger(__name__)


def export_job(dataset, *, destination=None, progress=None, cancelled=None):
    from bivosia.dataset.annotation_data import AnnotationData
    with closing(AnnotationData(dataset)) as data:
        annotation = dataset.annotation
        if annotation is None:
            raise FileNotFoundError('Select an annotation before exporting')
        session = data.session(annotation)
        path = dataset.path(destination, write=True) if destination else dataset.linked_masks(annotation)
        callback = None if progress is None else lambda done, total: progress({'completed': done, 'total': total})
        result = export_masks(session, dataset, path, quality=data.evidence.quality,
                              switches=data.evidence.switch_frames, progress=callback, cancelled=cancelled)
        session.export = result
        session.dirty = True
        session.save()
        return [str(path)]


def export_masks(session, dataset, destination, *, quality=None, switches=None, progress=None, cancelled=None):
    destination = dataset.path(destination, write=True)
    name = dataset.export_name(destination)
    for array in (quality, switches):
        if array is not None and array.shape != (session.frame_count,):
            raise ValueError("Quality/switch arrays must match the mask frame count")
    report = Progress("Mask export", session.frame_count, logger=logger, callback=progress, cancelled=cancelled)
    unresolved = 0
    automatic_frames = automatic_zone_frames(session)
    with mask_pair(dataset.mask("forward"), dataset.mask("reverse")) as (forward, reverse):
        if forward.shape[0] != session.frame_count:
            raise ValueError("Mask and annotation frame counts differ")
        sources = {'forward': forward, 'reverse': reverse}

        def boundary_match(job):
            return match_masks(sources[job.left.source][job.left.end],
                               sources[job.right.source][job.right.start], session.n)

        links = prepare_links(session, boundary_match)
        errors = validation_errors(session, links)
        if errors:
            raise ValueError("Cannot export:\n" + "\n".join(errors[:30]))
        result = links.resolve()
        with container_output(destination, overwrite=False) as output:
            arrays = linked_mask_arrays(output, forward.shape,
                                        dataset.relative(session.path) if session.path else "", session.fingerprint())
            output.attrs['annotation_mode'] = session.header.get('mode', 'manual')
            for segment in result.partition:
                for frame in range(segment.start, segment.end + 1):
                    report.check_cancelled()
                    try:
                        masks, source, valid = linked_frame(
                            np.asarray(forward[frame]), np.asarray(reverse[frame]), segment.source,
                            result.maps.get(segment.id), session.n,
                            quality=None if quality is None else quality[frame],
                            switch=False if switches is None else bool(switches[frame]), seed=session.header["seed"],
                            allow_missing=bool(automatic_frames[frame]))
                    except ValueError as error:
                        raise ValueError(f"Frame {frame}: {error}") from error
                    write_linked_frame(arrays, frame, masks, source, valid)
                    unresolved += int(source != 2 and not all(valid))
                    report.update(1)
            output.attrs["unresolved_other_direction_frames"] = unresolved
            if session.path is not None:
                saved = read_json(session.path)
                if revision_fingerprint(saved['header'], saved['zones'], saved['links'],
                                        saved.get('link_frames', [])) != session.fingerprint():
                    raise ValueError('Annotation changed during export; result was not published')
    report.finish(dataset.relative(destination))
    return {"path": name, "fingerprint": session.fingerprint(),
            "unresolved_other_direction_frames": unresolved}
