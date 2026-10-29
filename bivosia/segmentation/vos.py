"""Reference selection and model interfaces for video-object segmentation.

Examples
--------
>>> from bivosia.segmentation.vos import reference_frames
>>> reference_frames([0, 10, 20], 0, 10, 'reverse', 'start')
[10]
>>> reference_frames([0, 10, 20], 0, 10, 'forward', 'ends')
[0, 10]
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from abc import abstractmethod
from functools import partial

from bivosia.config import VOS_BACKENDS
from bivosia.segmentation.models import Segmenter, create_segmenter as _create, model_availability as _availability


def reference_frames(key_frames, start, end, direction, mode):
    if direction not in ('forward', 'reverse'):
        raise ValueError('Choose forward or reverse')
    if mode == 'start':
        return [start if direction == 'forward' else end]
    if mode == 'ends':
        return sorted(set((start, end)))
    if mode == 'all':
        return list(key_frames)
    raise ValueError('Choose start, ends or all references')


class VideoSegmenter(Segmenter):
    """Segment supplied RGB frames using labeled image references.

    Frames are (source_index, RGB_array) pairs in propagation order; references
    map source indices to (RGB_array, integer_mask) pairs. Each call starts fresh
    tracking state and yields (source_index, integer_mask) pairs in the same order.
    Model weights may be reused across calls. No files are opened by this interface.
    """

    backends = VOS_BACKENDS

    @property
    def reference_modes(self):
        return self.definition['reference_modes']

    @abstractmethod
    def propagate(self, frames, references, *, direction, animals, cancelled=None):
        """Yield label images with background 0 and stable supplied object IDs."""


create_segmenter = partial(_create, VOS_BACKENDS)
model_availability = partial(_availability, VOS_BACKENDS)
