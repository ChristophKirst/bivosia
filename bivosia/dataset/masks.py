"""Read mask sequences and write validated mask arrays."""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import contextmanager

import numpy as np

from bivosia.config import LINKED_MASK_STREAMS
from bivosia.dataset.formats import format_handler


def open_masks(path, dataset="masks", *, require_valid=True, direction=None, format=None):
    """Open HDF5, NPY or NPZ masks through the common MaskData interface."""
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
    """Open a context of named MaskData streams sharing one linked-export file.

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
