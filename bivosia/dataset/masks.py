"""Read mask sources and write HDF5, NPY and NPZ labels with frame validity.

Examples
--------
>>> from pathlib import Path
>>> from tempfile import TemporaryDirectory
>>> import numpy as np
>>> from bivosia.dataset.masks import mask_output, write_masks
>>> labels = np.zeros((2, 3, 4), dtype=np.uint8)
>>> valid = np.array([True, False])
>>> with TemporaryDirectory() as directory:
...     path = write_masks(Path(directory) / 'masks.npy', labels, valid=valid)
...     with mask_output(Path(directory) / 'masks.npz', labels.shape) as (output, output_valid):
...         output[:] = labels
...         output_valid[:] = valid

With an existing dataset containing a saved annotation or matching export:

>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.dataset.masks import ConsensusMaskReader
>>> dataset = Dataset('/path/to/dataset')
>>> with ConsensusMaskReader(dataset) as masks:  # doctest: +SKIP
...     labels = masks[0]
...     valid = bool(masks.valid[0])
"""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import ExitStack, closing, contextmanager
from pathlib import Path
import operator
import os

import numpy as np

from bivosia.config import CONSENSUS_MASK_DEFAULTS, CONSENSUS_MASK_SOURCES, LINKED_MASK_STREAMS
from bivosia.analysis.keyframes import validate_coverage
from bivosia.analysis.mask_linking import automatic_zone_frames, preferred_mask, prepare_links, validation_errors
from bivosia.analysis.mask_features import missing_mask_counts
from bivosia.analysis.masks import validate_mask_data
from bivosia.analysis.zones_of_disagreement import preferred_sources, revision_fingerprint, validate_consensus_export
from bivosia.dataset.annotations.masks import RevisionSession
from bivosia.dataset.io.containers import open_container
from bivosia.dataset.io.interfaces import MaskReader
from bivosia.dataset.io.json import read_json
from bivosia.dataset.io.registry import format_handler
from bivosia.dataset.paths import auxiliary_path, format_paths
from bivosia.dataset.mask_quality import MaskOverlapReader


# Mask readers

def open_masks(path, dataset="masks", *, require_valid=True, direction=None, format=None):
    """Open HDF5, NPY or NPZ masks through the common MaskReader interface."""
    masks = format_handler('masks', path, format=format)(path, dataset, direction=direction)
    try:
        if require_valid and not masks.valid.all():
            raise ValueError('These masks cover only a crop; full-recording analysis requires all frames. '
                             'The VOS viewer can display the cropped result.')
        return masks
    except BaseException:
        masks.close()
        raise


def open_linked_masks(path, *, format=None):
    """Open a context of named MaskReader streams sharing one linked-export file.

    Examples
    --------
    >>> from bivosia.dataset.masks import open_linked_masks
    >>> with open_linked_masks('/path/to/review.linked.h5') as streams:  # doctest: +SKIP
    ...     frame = streams['preferred'][0]
    ...     valid = bool(streams['preferred'].valid[0])
    """
    return format_handler('linked_masks', path, format=format)(path)


@contextmanager
def mask_pair(forward, reverse, dataset="masks", *, require_valid=True):
    first = open_masks(forward, dataset, require_valid=require_valid, direction='forward')
    try:
        second = open_masks(reverse, dataset, require_valid=require_valid, direction='reverse')
        try:
            if first.shape != second.shape:
                raise ValueError("Forward/reverse mask dimensions or frame counts differ")
            yield first, second
        finally:
            second.close()
    finally:
        first.close()


def mask_shape(forward, reverse, dataset="masks"):
    with mask_pair(forward, reverse, dataset) as (first, _second):
        return first.shape


# Export and annotation source selection

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


class ConsensusMaskReader(MaskReader):
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
                with closing(MaskOverlapReader(dataset.array('mask_intersections'), dataset.array('mask_areas'))) as counts:
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


# Mask writers

def output_files(path, *, include_optional=False):
    """List the mask file and its existing or possible format companions."""
    path = Path(path)
    companions = format_paths('masks', path, 'companions').values()
    return [path, *(item for item in companions if include_optional or item.exists())]


def _partial(path):
    return auxiliary_path(path, 'partial')


def check_output(path):
    """Refuse existing results, partial arrays, or a competing writer."""
    path = Path(path)
    format_handler('masks', path, operation='writer')
    candidates = [auxiliary_path(path, 'lock')]
    candidates.extend(format_paths('masks', path, 'scratch').values())
    for target in output_files(path, include_optional=True):
        candidates.extend((target, _partial(target)))
    for candidate in candidates:
        if candidate.exists():
            raise FileExistsError(f'Existing result or active/incomplete job: {candidate}')


@contextmanager
def mask_outputs(paths, shape, *, dtype=np.uint8, metadata=None):
    """Publish a group of closed mask files, discarding the group's output on failure.

    Destinations are exclusive: existing labels, validity files and unfinished
    outputs are protected. NPY output is memory-mapped while writing. NPZ output
    uses a disk array during calculation and compresses it after the caller exits.
    Every validity vector starts false; callers mark the frames they produce.
    """
    paths = {name: Path(path) for name, path in paths.items()}
    targets = [target.resolve() for path in paths.values()
               for target in output_files(path, include_optional=True)]
    if len(targets) != len(set(targets)):
        raise ValueError('Mask outputs must have distinct destinations')
    for path in paths.values():
        check_output(path)
    locks, pending, published = [], [], []
    try:
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            lock = auxiliary_path(path, 'lock')
            with lock.open('x', encoding='utf-8') as stream:
                stream.write(str(os.getpid()))
            locks.append(lock)
        with ExitStack() as stack:
            outputs = {name: stack.enter_context(format_handler('masks', path, operation='writer')(
                path, shape, dtype, metadata or {}, pending))
                       for name, path in paths.items()}
            yield outputs
        # Reserve every destination before publishing any completed array.
        for partial, path in pending:
            with path.open('xb'):
                pass
            published.append((partial, path))
        for partial, path in pending:
            os.replace(partial, path)
    except BaseException:
        for partial, path in published:
            path.unlink(missing_ok=True)
        for partial, path in pending:
            partial.unlink(missing_ok=True)
        raise
    finally:
        for lock in locks:
            lock.unlink()


@contextmanager
def mask_output(path, shape, *, dtype=np.uint8):
    """Yield writable (masks, valid) arrays for one new mask file."""
    with mask_outputs({'masks': path}, shape, dtype=dtype) as outputs:
        yield outputs['masks']


def write_masks(path, masks, *, valid=None):
    """Save a supplied integer label array without replacing existing files."""
    masks = np.asarray(masks)
    if masks.ndim != 3:
        raise ValueError('Expected integer masks with shape (frames, height, width)')
    valid = np.ones(masks.shape[0], dtype=bool) if valid is None else np.asarray(valid)
    validate_mask_data(masks, valid)
    with mask_output(path, masks.shape, dtype=masks.dtype) as (labels, flags):
        labels[:] = masks
        flags[:] = valid
    return Path(path)


# VOS and linked-mask outputs

def output_paths(dataset, directions):
    return {direction: dataset.mask(direction, write=True) for direction in directions}


@contextmanager
def vos_outputs(dataset, directions, count, shape, crop):
    """Publish masks in the selected formats only after crop coverage is complete."""
    with mask_outputs(output_paths(dataset, directions), (count, *shape),
                      metadata={'crop': crop}) as outputs:
        yield outputs
        for masks, valid in outputs.values():
            validate_coverage(valid[:], crop)


def linked_mask_arrays(output, shape, annotation_file, fingerprint):
    """Create globally labeled outputs and their validity flags."""
    output.attrs["annotation_fingerprint"] = fingerprint
    output.attrs["annotation_file"] = annotation_file
    output.attrs["invalid_or_unresolved"] = "Empty masks are unknown when their validity flag is false; originals are unchanged."
    arrays = {key: output.create_dataset(key, shape=shape, dtype=np.uint16,
              chunks=(1, *shape[1:]), compression="lzf")
              for key in LINKED_MASK_STREAMS}
    arrays["preferred_valid"] = output.create_dataset("preferred_valid", shape=(shape[0],), dtype=bool)
    arrays["direction_valid"] = output.create_dataset("direction_valid", shape=(2, shape[0]), dtype=bool)
    arrays["source"] = output.create_dataset("source", shape=(shape[0],), dtype=np.uint8)
    arrays["source"].attrs["codes"] = "0=forward, 1=reverse, 2=invalid"
    return arrays


def write_linked_frame(arrays, frame, masks, source, valid):
    for direction, mask in masks.items():
        arrays[direction][frame] = mask
    arrays["source"][frame] = source
    arrays["direction_valid"][:, frame] = valid
    arrays["preferred_valid"][frame] = source != 2
