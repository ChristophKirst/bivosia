"""Lossless NumPy array messages for remote readers and image models.

Examples
--------
>>> import numpy as np
>>> from bivosia.dataset.array_transport import pack_arrays, unpack_arrays
>>> np.array_equal(unpack_arrays(pack_arrays(a=np.arange(3)))['a'], np.arange(3))
True
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from io import BytesIO
import base64

import numpy as np


def pack_arrays(**arrays):
    """Encode numeric arrays losslessly in a compressed NumPy archive."""
    stream = BytesIO()
    np.savez_compressed(stream, **arrays)
    return base64.b64encode(stream.getvalue()).decode('ascii')


def unpack_arrays(payload):
    """Decode only NumPy arrays that do not require pickle."""
    with np.load(BytesIO(base64.b64decode(payload, validate=True)), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


