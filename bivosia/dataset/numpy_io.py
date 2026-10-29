"""NumPy array headers and memory-mapped NPY storage."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import contextmanager
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from bivosia.dataset.inventory import FileMetadata
from bivosia.dataset.paths import auxiliary_path


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
    """Commit a flushed array on success; retain partial output on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = auxiliary_path(path, "partial")
    array = np.lib.format.open_memmap(partial, mode="w+", dtype=dtype, shape=shape)
    try:
        if fill is not None:
            array[:] = fill
        yield array
        array.flush()
    finally:
        array._mmap.close()
    partial.replace(path)
