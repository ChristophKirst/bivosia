"""NumPy arrays, header inspection and mask adapters.

Examples
--------
>>> import numpy as np
>>> from bivosia.dataset.io.numpy import NumpyMaskReader
>>> masks = NumpyMaskReader(np.zeros((3, 2, 4), dtype=np.uint8))
>>> masks.shape, masks.valid.tolist()
((3, 2, 4), [True, True, True])
"""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import contextmanager
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from bivosia.analysis.masks import validate_mask_data
from bivosia.dataset.io.interfaces import FileMetadata, MaskReader
from bivosia.dataset.paths import auxiliary_path, format_paths


# Array I/O and header inspection

def read_array(path):
    return np.load(path, mmap_mode='r', allow_pickle=False)


def _array_header(stream, name):
    """Read the NPY header without reading or unpickling the array payload."""
    version = np.lib.format.read_magic(stream)
    # NumPy's shared reader handles NPY versions 1, 2 and 3, including UTF-8 names.
    shape, _, dtype = np.lib.format._read_array_header(stream, version)
    return name, shape, str(dtype)


def inspect_npy(path):
    """Inspect an NPY file's name, shape and dtype, reading only its header."""
    with Path(path).open('rb') as stream:
        return FileMetadata(arrays=(_array_header(stream, Path(path).stem),))


def inspect_npz(path):
    """Read member headers without expanding the compressed array payloads."""
    rows = []
    with ZipFile(path) as archive:
        for member in archive.infolist():
            if member.filename.endswith('.npy'):
                with archive.open(member) as stream:
                    rows.append(_array_header(stream, member.filename[:-4]))
    return FileMetadata(arrays=tuple(rows))


@contextmanager
def array_output(path, shape, dtype, *, fill=None):
    """Commit a flushed array on success and discard failed partial output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = auxiliary_path(path, "partial")
    with partial.open('wb'):
        pass
    try:
        array = np.lib.format.open_memmap(partial, mode="w+", dtype=dtype, shape=shape)
        try:
            if fill is not None:
                array[:] = fill
            yield array
            array.flush()
        finally:
            array._mmap.close()
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


# Mask adapters and staged outputs

def validity_path(path):
    """Return the validity sidecar for an NPY label array."""
    return format_paths('masks', path, 'companions', format='npy')['validity']


class NumpyMaskReader(MaskReader):
    """Adapt an array, optionally selecting one pass from (2, frames, height, width)."""

    def __init__(self, masks, valid=None, *, direction=None):
        if masks.ndim == 4 and masks.shape[0] == 2:
            if direction not in ('forward', 'reverse'):
                raise ValueError('Select forward or reverse for a two-pass mask array')
            index = ('forward', 'reverse').index(direction)
            masks = masks[index]
            if valid is not None and valid.ndim == 2:
                valid = valid[index]
        validate_mask_data(masks)
        self.masks = masks
        self.shape, self.dtype = masks.shape, masks.dtype
        self.valid = np.ones(self.shape[0], dtype=bool) if valid is None else np.asarray(valid)
        validate_mask_data(self.masks, self.valid)

    def __getitem__(self, frames):
        return self.masks[frames]

    def close(self):
        """In-memory arrays remain owned by their caller."""


class NpyMaskReader(NumpyMaskReader):
    """Read memory-mapped labels and an optional boolean .valid.npy sidecar."""

    def __init__(self, path, dataset='masks', *, direction=None):
        self.array = np.load(path, mmap_mode='r', allow_pickle=False)
        try:
            sidecar = validity_path(path)
            valid = np.load(sidecar, allow_pickle=False) if sidecar.is_file() else None
            super().__init__(self.array, valid, direction=direction)
        except BaseException:
            self.close()
            raise

    def close(self):
        self.array._mmap.close()


class NpzMaskReader(NumpyMaskReader):
    """Load labels and validity from one archive; NPZ members are not memory-mapped."""

    def __init__(self, path, dataset='masks', *, direction=None):
        with np.load(path, allow_pickle=False) as archive:
            masks = archive[dataset]
            valid = archive['valid'] if 'valid' in archive else None
        super().__init__(masks, valid, direction=direction)


@contextmanager
def _mask_buffer(path, shape, dtype):
    with path.open('xb'):
        pass
    try:
        masks = np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)
        try:
            valid = np.zeros(shape[0], dtype=bool)
            validate_mask_data(masks, valid)
            yield masks, valid
            masks.flush()
        finally:
            masks._mmap.close()
    except BaseException:
        path.unlink(missing_ok=True)
        raise


@contextmanager
def stage_npy_masks(path, shape, dtype, metadata, pending):
    """Stage memory-mapped labels and, when needed, a validity sidecar."""
    partial = auxiliary_path(path, 'partial')
    with _mask_buffer(partial, shape, dtype) as (masks, valid):
        pending.append((partial, path))
        yield masks, valid
        if not valid.all():
            sidecar = validity_path(path)
            sidecar_partial = auxiliary_path(sidecar, 'partial')
            with sidecar_partial.open('xb') as stream:
                pending.append((sidecar_partial, sidecar))
                np.save(stream, valid, allow_pickle=False)


@contextmanager
def stage_npz_masks(path, shape, dtype, metadata, pending):
    """Stage a compressed archive using a disk-backed array during calculation."""
    partial = auxiliary_path(path, 'partial')
    buffer = format_paths('masks', path, 'scratch', format='npz')['buffer']
    with _mask_buffer(buffer, shape, dtype) as (masks, valid):
        yield masks, valid
        with partial.open('xb') as stream:
            pending.append((partial, path))
            np.savez_compressed(stream, masks=masks, valid=valid, **metadata)
    buffer.unlink()
