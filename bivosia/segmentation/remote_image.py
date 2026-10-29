"""Run the existing image segmenter in a configured SSH installation.

Examples
--------
>>> from bivosia.processing.remote import Remote
>>> from bivosia.segmentation.image import create_segmenter
>>> host = Remote('workstation', 'python', '/software/bivosia')
>>> predictor = create_segmenter('sam2', '/models/sam2.1_hiera_tiny.pt', remote=host)
>>> predictor.label
'SAM2'
>>> predictor.close()
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path

from bivosia.processing.remote_images import pack_arrays, unpack_arrays
from bivosia.processing.remote_session import RemoteSession
from bivosia.segmentation.image import ImageSegmenter


class RemoteImageSegmenter(ImageSegmenter):
    """Preserve the image predictor interface across one persistent SSH session."""

    def __init__(self, remote, model, checkpoint, device='auto', parameters=None):
        self.name, self.remote = model, remote
        if not checkpoint:
            raise ValueError('Specify the checkpoint path on the remote installation')
        super().__init__(checkpoint, device, parameters)
        # Home directories in checkpoint paths belong to the remote account.
        self.checkpoint = Path(checkpoint)
        self.session = None
        self.image_key = object()
        self._metadata = None

    @property
    def metadata(self):
        if self._metadata is None:
            raise RuntimeError('The remote model has not been checked')
        return self._metadata.copy()

    def check(self):
        self.check_parameters()
        if self.session is None:
            self.session = RemoteSession(self.remote, 'image_session')
            try:
                response = self.session.request('configure', settings={
                    'model': self.name, 'checkpoint': str(self.checkpoint),
                    'device': self.device, 'parameters': self.parameters})
                self.device = response['device']
                self._metadata = response['metadata']
            except Exception:
                self.close()
                raise
        return self.device

    def predict(self, rgb, points, labels, *, image_key):
        try:
            self.check()
            arrays = {'points': points, 'labels': labels}
            if image_key != self.image_key:
                arrays['rgb'] = rgb
            response = self.session.request('predict', arrays=pack_arrays(**arrays))
            result = unpack_arrays(response['arrays'])
            self.image_key = image_key
            self._metadata = response['metadata']
            return result['masks'], result['scores']
        except Exception:
            self.close()
            raise

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None
        self.image_key = object()
