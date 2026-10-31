"""Read and write arrays through registered format adapters."""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from bivosia.dataset.io.registry import format_handler


def read_array(path):
    """Open a read-only, memory-mapped NumPy array."""
    return format_handler('arrays', path)(path)


def array_output(path, shape, dtype, *, fill=None):
    """Create an array through its registered writer, removing failed partials."""
    return format_handler('arrays', path, operation='writer')(path, shape, dtype, fill=fill)


def write_array(path, values):
    with array_output(path, values.shape, values.dtype) as output:
        output[:] = values
