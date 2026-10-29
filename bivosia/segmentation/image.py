"""Common interface for prompted image segmentation."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from abc import abstractmethod
from functools import partial

from bivosia.config import KEY_FRAME_BACKENDS
from bivosia.segmentation.models import Segmenter, create_segmenter as _create, model_availability as _availability


class ImageSegmenter(Segmenter):
    """Return candidate boolean masks and scores for one image and its prompts."""

    backends = KEY_FRAME_BACKENDS

    @abstractmethod
    def predict(self, rgb, points, labels, *, image_key):
        """Predict candidates, reusing the image embedding when its key is unchanged."""


def create_segmenter(model, checkpoint=None, device='auto', parameters=None, *, remote=None):
    """Construct an image predictor locally or in a configured remote installation."""
    if remote is not None:
        from bivosia.segmentation.remote_image import RemoteImageSegmenter
        return RemoteImageSegmenter(remote, model, checkpoint, device, parameters)
    return _create(KEY_FRAME_BACKENDS, model, checkpoint, device, parameters)


model_availability = partial(_availability, KEY_FRAME_BACKENDS)
