"""Read external mask frames and plot data through one persistent SSH session."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import ExitStack, redirect_stdout
from operator import index
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
import json
import sys

import numpy as np

from bivosia.config import REMOTE_PROTOCOL_VERSION
from bivosia.dataset.io.transport import pack_arrays, unpack_arrays
from bivosia.dataset.io.arrays import read_array
from bivosia.dataset.io.interfaces import MaskReader
from bivosia.dataset.io.json import atomic_json, read_json
from bivosia.dataset.layout import Dataset
from bivosia.dataset.storage.locations import remote_path
from bivosia.dataset.masks import open_masks
from bivosia.processing.remote.session import RemoteSession


# Only the numerical products used by the recording viewer are transferred whole.
PLOT_ARRAYS = ('mask_intersections', 'mask_areas', 'mask_iou_match', 'mask_quality',
               'mask_missing', 'mask_switch', 'mask_temporal_quality', 'mask_measures')
PLOT_METADATA = (('mask_quality_metadata', None), ('mask_measures', 'metadata'))


class RemoteMaskReader(MaskReader):
    """A frame-indexed stream whose connection belongs to its dataset reader."""

    def __init__(self, reader, direction, header):
        self.reader, self.direction = reader, direction
        self.shape, self.dtype = tuple(header['shape']), np.dtype(header['dtype'])
        self.valid = unpack_arrays(header['arrays'])['valid']
        self.closed = False
        if len(self.shape) != 3 or not np.issubdtype(self.dtype, np.integer) or self.valid.shape != self.shape[:1]:
            raise ValueError('Invalid remote mask header')

    def __getitem__(self, frames):
        if self.closed:
            raise ValueError('Mask reader is closed')
        if isinstance(frames, slice):
            return np.stack([self[t] for t in range(*frames.indices(self.shape[0]))]) if len(
                range(*frames.indices(self.shape[0]))) else np.empty((0, *self.image_shape), self.dtype)
        frame = index(frames)
        if frame < 0:
            frame += self.shape[0]
        if not 0 <= frame < self.shape[0]:
            raise IndexError(frame)
        response = self.reader.request('frame', direction=self.direction, frame=frame)
        masks = unpack_arrays(response['arrays'])['mask']
        if masks.shape != self.image_shape or masks.dtype != self.dtype:
            raise ValueError('Remote mask frame differs from its header')
        return masks

    def close(self):
        self.closed = True


class RemoteDatasetReader:
    """Own forward/reverse readers, one connection, and a temporary plot-data snapshot."""

    def __init__(self, remote):
        self.remote = remote
        self.session = RemoteSession(remote, 'data_session')
        self.lock = RLock()
        self.masks = {}
        self._temporary = None
        self._analysis = None

    def request(self, operation, **values):
        with self.lock:
            return self.session.request(operation, **values)

    def open_masks(self, direction):
        if direction not in self.masks:
            header = self.request('open_masks', direction=direction)
            self.masks[direction] = RemoteMaskReader(self, direction, header)
        return self.masks[direction]

    def mask_paths(self):
        """Find available directions without requiring both inference passes to exist."""
        outputs = self.remote.call('outputs')['masks']
        errors = [entry['error'] for entry in outputs.values() if entry['error']]
        if errors:
            raise ValueError('\n'.join(errors))
        return {name: Path(entry['path']) for name, entry in outputs.items() if entry['exists']}

    def analysis_dataset(self):
        """Use normal array readers on a private snapshot; never write into the local dataset."""
        if self._analysis is None:
            response = self.request('analysis')
            self._temporary = TemporaryDirectory(prefix='bivosia-plot-')
            snapshot = Dataset(self._temporary.name)
            for name, values in unpack_arrays(response['arrays']).items():
                if name not in PLOT_ARRAYS:
                    raise ValueError('Unexpected remote analysis array')
                path = snapshot.file(name, write=True)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, values, allow_pickle=False)
            for name, companion, values in response['metadata']:
                if (name, companion) not in PLOT_METADATA:
                    raise ValueError('Unexpected remote analysis metadata')
                path = snapshot.file(name, companion=companion, write=True)
                atomic_json(path, values)
            if response['annotation'] is not None:
                name, record = response['annotation']
                path = snapshot.annotation_path(name, write=True)
                atomic_json(path, record)
                snapshot.select_annotation(path)
            self._analysis = snapshot
        return self._analysis

    def close(self):
        self.session.close()
        for masks in self.masks.values():
            masks.close()
        self.masks.clear()
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
        self._analysis = None


def _analysis_snapshot(dataset):
    paths = {name: dataset.file(name) for name in PLOT_ARRAYS}
    metadata_paths = [(name, companion, dataset.file(name, companion=companion))
                      for name, companion in PLOT_METADATA]
    annotation = dataset.annotation
    present = [p for p in [*paths.values(), *(p for _, _, p in metadata_paths)] if p.is_file()]
    if annotation is not None:
        present.append(annotation)
    def signature():
        return [(p.stat().st_size, p.stat().st_mtime_ns) for p in present]
    before = signature()
    with ExitStack() as stack:
        arrays = {}
        for name, path in paths.items():
            if path.is_file():
                arrays[name] = read_array(path)
                mapping = getattr(arrays[name], '_mmap', None)
                if mapping is not None:
                    stack.callback(mapping.close)
        result = {'arrays': pack_arrays(**arrays), 'metadata': [
            (name, companion, read_json(path)) for name, companion, path in metadata_paths if path.is_file()]}
        result['annotation'] = None if annotation is None else (dataset.relative(annotation), read_json(annotation))
        if signature() != before:
            raise RuntimeError('Analysis changed while reading it; wait for the job to finish and reopen the viewer')
        return result


def serve_data(request):
    """Keep registered mask adapters open until the client closes its connection."""
    output = sys.stdout
    readers = {}
    with redirect_stdout(sys.stderr), ExitStack() as stack:
        try:
            if request.get('protocol') != REMOTE_PROTOCOL_VERSION:
                raise ValueError('Update BiVOSIA on both machines: incompatible session protocol')
            dataset = Dataset(remote_path(request['dataset']))
            for line in sys.stdin.buffer:
                values = json.loads(line)
                operation = values['operation']
                if operation == 'analysis':
                    result = _analysis_snapshot(dataset)
                elif operation in ('open_masks', 'frame'):
                    direction = values['direction']
                    if direction not in ('forward', 'reverse'):
                        raise ValueError('Choose forward or reverse masks')
                    if direction not in readers:
                        readers[direction] = stack.enter_context(open_masks(
                            dataset.mask(direction), direction=direction, require_valid=False))
                    masks = readers[direction]
                    if operation == 'open_masks':
                        result = {'shape': masks.shape, 'dtype': str(masks.dtype),
                                  'arrays': pack_arrays(valid=masks.valid)}
                    else:
                        frame = index(values['frame'])
                        if not 0 <= frame < masks.shape[0]:
                            raise IndexError(frame)
                        result = {'arrays': pack_arrays(mask=masks[frame])}
                else:
                    raise ValueError(f'Unknown data-session operation: {operation}')
                print(json.dumps({'protocol': REMOTE_PROTOCOL_VERSION, **result}), file=output, flush=True)
        except Exception as error:
            print(json.dumps({'protocol': REMOTE_PROTOCOL_VERSION, 'error': str(error)}), file=output, flush=True)
