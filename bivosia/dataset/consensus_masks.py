"""Read globally labelled preferred masks from an annotation or its export.

Examples
--------
>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.dataset.consensus_masks import ConsensusMasks
>>> dataset = Dataset('/path/to/dataset')
>>> with ConsensusMasks(dataset) as masks:  # doctest: +SKIP
...     labels = masks[0]
...     valid = bool(masks.valid[0])
>>> with ConsensusMasks(dataset, source='annotation') as masks:  # doctest: +SKIP
...     labels = masks[0]
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import ExitStack, closing
import operator

import numpy as np

from bivosia.config import CONSENSUS_MASK_DEFAULTS, CONSENSUS_MASK_SOURCES
from bivosia.analysis.linking import automatic_zone_frames, preferred_mask, prepare_links, validation_errors
from bivosia.analysis.mask_features import missing_mask_counts
from bivosia.analysis.zones_of_disagreement import preferred_sources, revision_fingerprint, validate_consensus_export
from bivosia.dataset.arrays import MaskOverlaps
from bivosia.dataset.containers import open_container
from bivosia.dataset.frame_data import MaskData
from bivosia.dataset.jsonio import read_json
from bivosia.dataset.masks import mask_pair, open_linked_masks
from bivosia.dataset.paths import format_paths
from bivosia.dataset.session import RevisionSession


def matching_export(dataset, record):
    """Return an existing export only when its annotation fingerprint matches."""
    exported = record.get('export')
    if not exported:
        return None
    path = dataset.export_path(exported['path'])
    if not path.is_file():
        return None
    with open_container(path) as file:
        try:
            validate_consensus_export(record, file.attrs.get('annotation_fingerprint'))
        except ValueError:
            return None
    return path


def _source(dataset, source):
    if source not in CONSENSUS_MASK_SOURCES:
        raise ValueError(f'Unknown mask source: {source}')
    annotation = dataset.annotation
    if annotation is None:
        raise FileNotFoundError('Select a ZoD annotation before reading consensus masks')
    record = read_json(annotation)
    fingerprint = revision_fingerprint(record['header'], record['zones'], record['links'], record.get('link_frames', []))
    exported = matching_export(dataset, record) if source != 'annotation' else None
    if source == 'export' and exported is None:
        if record.get('export') and dataset.export_path(record['export']['path']).is_file():
            raise ValueError('Exported masks do not match the selected annotation; export again')
        raise FileNotFoundError('The selected annotation has no available exported masks')
    source = 'export' if exported is not None else 'annotation'
    if exported is None:
        paths = [dataset.mask(direction) for direction in ('forward', 'reverse')]
        companions = [companion for path in paths
                      for companion in format_paths('masks', path, 'companions').values() if companion.is_file()]
        files = paths + companions + [dataset.array(name) for name in ('mask_intersections', 'mask_areas')]
    else:
        paths = [exported]
        files = paths
    signatures = []
    for path in files:
        stat = path.stat()
        signatures.append({'path': dataset.relative(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
    metadata = {'mask_source': source, 'source_files': signatures,
                'annotation': dataset.relative(annotation), 'annotation_fingerprint': fingerprint}
    return record, paths, metadata


def consensus_metadata(dataset, *, source=CONSENSUS_MASK_DEFAULTS['source']):
    """Identify the selected annotation and source files without reading mask pixels."""
    return _source(dataset, source)[2]


class ConsensusMasks(MaskData):
    """Read a matching export when present, otherwise apply the saved annotation.

    Saved links are resolved once. Original labels are read and relabelled only
    for requested frames; no linking proposals are generated. Invalid intervals
    return zero labels and have false validity. The reader owns all open handles.
    Explicit ``annotation`` or ``export`` sources disable automatic selection.
    """

    def __init__(self, dataset, *, source=CONSENSUS_MASK_DEFAULTS['source']):
        self._resources = ExitStack()
        self._closed = False
        self._export = None
        try:
            record, paths, self.metadata = _source(dataset, source)
            if self.metadata['mask_source'] == 'export':
                streams = self._resources.enter_context(open_linked_masks(paths[0]))
                self._export = streams['preferred']
                self.shape, self.dtype, self.valid = self._export.shape, self._export.dtype, self._export.valid
            else:
                with closing(MaskOverlaps(dataset.array('mask_intersections'), dataset.array('mask_areas'))) as counts:
                    missing_forward, missing_reverse = missing_mask_counts(counts.areas)
                    self._session = RevisionSession.from_record(record, missing_forward, missing_reverse, counts.shape[1])
                links = prepare_links(self._session)
                errors = validation_errors(self._session, links)
                if errors:
                    raise ValueError('Cannot read consensus masks:\n' + '\n'.join(errors[:30]))
                self._links = links.resolve()
                self._ends = np.array([part.end for part in self._links.partition])
                self._automatic = automatic_zone_frames(self._session)
                self._masks = self._resources.enter_context(mask_pair(*paths))
                self.shape = self._masks[0].shape
                if self.shape[0] != self._session.frame_count:
                    raise ValueError('Mask and annotation frame counts differ')
                self.dtype = np.dtype('uint16')
                self.valid = preferred_sources(self._session.zones, self.shape[0]) != 2
        except BaseException:
            self.close()
            raise

    def __getitem__(self, frames):
        if self._closed:
            raise ValueError('Mask reader is closed')
        if self._export is not None:
            return self._export[frames]
        if isinstance(frames, slice):
            indices = range(*frames.indices(self.shape[0]))
            values = np.empty((len(indices), *self.image_shape), dtype=self.dtype)
            for offset, frame in enumerate(indices):
                values[offset] = self[frame]
            return values
        frame = operator.index(frames)
        if frame < 0:
            frame += self.shape[0]
        if not 0 <= frame < self.shape[0]:
            raise IndexError('Mask frame outside the recording')
        if not self.valid[frame]:
            return np.zeros(self.image_shape, dtype=self.dtype)
        part = self._links.partition[np.searchsorted(self._ends, frame)]
        raw = self._masks[0 if part.source == 'forward' else 1][frame]
        return preferred_mask(raw, self._links.maps[part.id], self._session.n,
                              allow_missing=self._automatic[frame])

    def close(self):
        self._closed = True
        self._resources.close()
