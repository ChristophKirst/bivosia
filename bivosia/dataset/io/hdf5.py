"""HDF5 containers, header inspection and mask adapters."""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import h5py

from bivosia.config import LINKED_MASK_STREAMS
from bivosia.analysis.masks import validate_linked_mask_data, validate_mask_data
from bivosia.dataset.io.interfaces import FileMetadata, MaskReader
from bivosia.dataset.paths import auxiliary_path


# Container I/O and header inspection

def open_hdf5(path):
    """Open a read-only HDF5 container; its caller owns the handle."""
    return h5py.File(path, 'r')


def inspect_arrays(path):
    """Return dataset names, shapes and dtypes without reading array values."""
    rows = []
    with open_hdf5(path) as file:
        def collect(name, item):
            if isinstance(item, h5py.Dataset):
                rows.append((name, item.shape, str(item.dtype)))
        file.visititems(collect)
    return FileMetadata(arrays=tuple(rows))


@contextmanager
def hdf5_output(path, *, overwrite=True):
    """Commit a closed HDF5 file on success and discard failed partial output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = auxiliary_path(path, "partial")
    if not overwrite and (path.exists() or partial.exists()):
        raise ValueError("Output or partial output already exists; choose a new filename")
    with partial.open('wb' if overwrite else 'xb'):
        pass
    try:
        with h5py.File(partial, 'w') as output:
            yield output
        if not overwrite and path.exists():
            raise ValueError(f"Output appeared during export: {path}")
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


# Mask adapters and staged outputs

class Hdf5MaskReader(MaskReader):
    """HDF5 mask sequence from a path or a shared open file.

    Supplied validity overrides the file's ordinary ``valid`` array. A reader
    given an open file borrows it; closing that reader leaves the file open.
    """

    def __init__(self, path, dataset='masks', *, direction=None, valid=None):
        self._owns_file = not isinstance(path, h5py.File)
        self.file = open_hdf5(path) if self._owns_file else path
        self._closed = False
        try:
            self.masks = self.file[dataset]
            self.shape, self.dtype = self.masks.shape, self.masks.dtype
            validate_mask_data(self.masks)
            if not self.file.attrs.get('complete', True):
                raise ValueError('Mask output is incomplete')
            self.valid = valid
            if self.valid is None:
                self.valid = (self.file['valid'][:] if 'valid' in self.file
                              else np.ones(self.shape[0], dtype=bool))
            validate_mask_data(self.masks, self.valid)
        except BaseException:
            self.close()
            raise

    def __getitem__(self, frames):
        if self._closed:
            raise ValueError('Mask reader is closed')
        return self.masks[frames]

    def close(self):
        self._closed = True
        if self._owns_file:
            self.file.close()


@contextmanager
def linked_mask_streams(path):
    """Share one HDF5 file across forward, reverse and preferred mask readers."""
    with open_hdf5(path) as file, ExitStack() as readers:
        validate_linked_mask_data(file)
        direction_valid = file['direction_valid'][:]
        validity = dict(zip(LINKED_MASK_STREAMS[:2], direction_valid))
        validity['preferred'] = file['preferred_valid'][:]
        yield {name: readers.enter_context(Hdf5MaskReader(file, name, valid=validity[name]))
               for name in LINKED_MASK_STREAMS}


@contextmanager
def stage_masks(path, shape, dtype, metadata, pending):
    """Write closed HDF5 partials for publication by the mask transaction."""
    partial = auxiliary_path(path, 'partial')
    with partial.open('xb'):
        pass
    pending.append((partial, path))
    with h5py.File(partial, 'w') as file:
        for name, value in metadata.items():
            file.attrs[name] = value
        file.attrs['complete'] = False
        masks = file.create_dataset('masks', shape, dtype=dtype,
                                    chunks=(1, *shape[1:]), compression='lzf')
        valid = file.create_dataset('valid', (shape[0],), dtype=bool)
        validate_mask_data(masks, valid)
        yield masks, valid
        file.attrs['complete'] = True
        file.flush()
