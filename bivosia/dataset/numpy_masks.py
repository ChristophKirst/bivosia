"""NumPy mask adapters for arrays, memory-mapped NPY files and NPZ archives.

Examples
--------
>>> import numpy as np
>>> from bivosia.dataset.numpy_masks import NumpyMasks
>>> masks = NumpyMasks(np.zeros((3, 2, 4), dtype=np.uint8))
>>> masks.shape, masks.valid.tolist()
((3, 2, 4), [True, True, True])
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import contextmanager

import numpy as np

from bivosia.analysis.masks import validate_mask_data
from bivosia.dataset.frame_data import MaskData
from bivosia.dataset.paths import auxiliary_path, format_paths


def validity_path(path):
    """Return the validity sidecar for an NPY label array."""
    return format_paths('masks', path, 'companions', format='npy')['validity']


class NumpyMasks(MaskData):
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


class NpyMasks(NumpyMasks):
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


class NpzMasks(NumpyMasks):
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
