"""SAM2 and SAM3 point-prompt inference for individual key-frame images.

Examples
--------
>>> from bivosia.segmentation.backends.sam import Sam3Segmenter
>>> predictor = Sam3Segmenter()
>>> predictor.name
'sam3'
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from abc import abstractmethod
import logging
import sys

import numpy as np

from bivosia.segmentation.image import ImageSegmenter
from bivosia.segmentation.device import resolve_device


class SamSegmenter(ImageSegmenter):
    """Retain a SAM model and reuse the current image embedding."""

    def __init__(self, checkpoint=None, device='auto', parameters=None):
        super().__init__(checkpoint, device, parameters)
        self.model = self.predictor = self.image_key = None

    @abstractmethod
    def components(self):
        """Import this SAM version's model builder and image predictor."""

    @abstractmethod
    def build(self, build_model, image_predictor):
        """Construct the model and its image predictor from local weights."""

    def check(self):
        self.check_parameters()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f'{self.label} checkpoint not found: {self.checkpoint}')
        self.components()
        return resolve_device(self.device)

    def _load(self):
        self.device = self.check()
        logging.getLogger(__name__).info('Loading %s: %s', self.label, self.checkpoint.name)
        self.build(*self.components())

    def predict(self, rgb, points, labels, *, image_key):
        if self.predictor is None:
            self._load()
        import torch

        with torch.inference_mode():
            if self.image_key != image_key:
                self.predictor.set_image(np.ascontiguousarray(rgb))
                self.image_key = image_key
            masks, scores, _ = self.predictor.predict(point_coords=points, point_labels=labels,
                                                       multimask_output=True)
        return np.asarray(masks, dtype=bool), np.asarray(scores)

    def close(self):
        self.model = self.predictor = self.image_key = None


class Sam2Segmenter(SamSegmenter):
    name = 'sam2'

    def components(self):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        return build_sam2, SAM2ImagePredictor

    def build(self, build_model, image_predictor):
        self.model = build_model(self.parameters['config'], str(self.checkpoint), device=self.device,
                                apply_postprocessing=False)
        self.predictor = image_predictor(self.model, max_hole_area=0, max_sprinkle_area=0)

    @property
    def metadata(self):
        return {**super().metadata, 'config': self.parameters['config']}


class Sam3Segmenter(SamSegmenter):
    name = 'sam3'

    def components(self):
        if sys.version_info < (3, 12):
            raise RuntimeError('SAM3 requires Python 3.12 or newer')
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor

        return build_sam3_image_model, SAM3InteractiveImagePredictor

    def build(self, build_model, image_predictor):
        self.model = build_model(checkpoint_path=str(self.checkpoint), device=self.device,
                                load_from_HF=False, enable_inst_interactivity=True)
        self.predictor = image_predictor(self.model.inst_interactive_predictor.model,
                                         max_hole_area=0, max_sprinkle_area=0)
