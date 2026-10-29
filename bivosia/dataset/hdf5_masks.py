"""HDF5 label-image and frame-validity storage."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import ExitStack, contextmanager

import numpy as np
import h5py

from bivosia.config import LINKED_MASK_STREAMS
from bivosia.analysis.masks import validate_linked_mask_data, validate_mask_data
from bivosia.dataset.frame_data import MaskData
from bivosia.dataset.hdf5_io import open_hdf5
from bivosia.dataset.paths import auxiliary_path


class Hdf5Masks(MaskData):
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
        yield {name: readers.enter_context(Hdf5Masks(file, name, valid=validity[name]))
               for name in LINKED_MASK_STREAMS}


@contextmanager
def stage_masks(path, shape, dtype, metadata, pending):
    """Write closed HDF5 partials for publication by the mask transaction."""
    partial = auxiliary_path(path, 'partial')
    with h5py.File(partial, 'x') as file:
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
    pending.append((partial, path))
