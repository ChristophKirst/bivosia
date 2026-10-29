"""Write HDF5, NPY and NPZ mask sequences with explicit frame validity.

Examples
--------
>>> from pathlib import Path
>>> from tempfile import TemporaryDirectory
>>> import numpy as np
>>> from bivosia.dataset.mask_output import mask_output, write_masks
>>> labels = np.zeros((2, 3, 4), dtype=np.uint8)
>>> valid = np.array([True, False])
>>> with TemporaryDirectory() as directory:
...     path = write_masks(Path(directory) / 'masks.npy', labels, valid=valid)
...     with mask_output(Path(directory) / 'masks.npz', labels.shape) as (output, output_valid):
...         output[:] = labels
...         output_valid[:] = valid
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import contextmanager, ExitStack
from pathlib import Path
import os

import numpy as np

from bivosia.analysis.masks import validate_mask_data
from bivosia.dataset.paths import auxiliary_path, format_paths
from bivosia.dataset.formats import format_handler


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
    """Publish a group of closed mask files, restoring partials on publication failure.

    Destinations are exclusive: existing labels, validity files and unfinished
    outputs are preserved. NPY output is memory-mapped while writing. NPZ output
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
            if partial.exists():
                path.unlink()
            else:
                os.replace(path, partial)
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
