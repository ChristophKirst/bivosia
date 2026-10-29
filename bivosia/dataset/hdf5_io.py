"""HDF5 containers with atomic result publication."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import contextmanager
from pathlib import Path

import h5py

from bivosia.dataset.inventory import FileMetadata
from bivosia.dataset.paths import auxiliary_path


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
    """Commit a closed HDF5 file on success; retain partial output on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = auxiliary_path(path, "partial")
    if not overwrite and (path.exists() or partial.exists()):
        raise ValueError("Output or partial output already exists; choose a new filename")
    with h5py.File(partial, "w" if overwrite else "x") as output:
        yield output
    if not overwrite and path.exists():
        raise ValueError(f"Output appeared during export; completed partial retained at {partial}")
    partial.replace(path)
