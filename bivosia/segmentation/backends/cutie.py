"""Cutie propagation with one or more labeled reference frames.

Examples
--------
With an installed Cutie backend and a labeled reference frame:

>>> import numpy as np
>>> from bivosia.segmentation.backends.cutie import CutieSegmenter
>>> model = CutieSegmenter('/path/to/cutie-base-mega.pth')
>>> image = np.zeros((64, 64, 3), dtype=np.uint8)
>>> labels = np.zeros((64, 64), dtype=np.uint8)
>>> labels[8:24, 8:24], labels[40:56, 40:56] = 1, 2
>>> frames = [(0, image), (1, image.copy())]
>>> references = {0: (image, labels)}
>>> try:  # doctest: +SKIP
...     masks = list(model.propagate(frames, references, direction='forward', animals=2))
... finally:
...     model.close()
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from itertools import chain
from pathlib import Path
import logging

import numpy as np

from bivosia.segmentation.device import resolve_device
from bivosia.segmentation.vos import VideoSegmenter


class CutieSegmenter(VideoSegmenter):
    name = 'cutie'

    def __init__(self, checkpoint=None, device='auto', parameters=None):
        super().__init__(checkpoint, device, parameters)
        self.model = self.config = None

    def check(self):
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f'Cutie checkpoint not found: {self.checkpoint}')
        from cutie.model.cutie import CUTIE
        from cutie.inference.inference_core import InferenceCore
        from hydra import compose, initialize_config_dir

        p = self.parameters
        self.check_parameters()
        if p['internal_size'] != -1 and p['internal_size'] < 16:
            raise ValueError('Internal size must be at least 16, or -1 for native size')
        if p['memory_every'] < 1 or not 2 <= p['memory_min'] < p['memory_max']:
            raise ValueError('Memory every must be positive; require 2 <= memory min < max')
        if p['long_term_tokens'] < 4096:
            raise ValueError('Long-term memory must hold at least 4096 tokens')
        self.backbone_directory()
        return resolve_device(self.device)

    def backbone_directory(self):
        """Locate the cached encoder weights required by Cutie's constructor."""
        import torch

        roots = (self.checkpoint.parent, Path(torch.hub.get_dir()) / 'checkpoints')
        for root in roots:
            if all((root / name).is_file() for name in self.definition['backbones']):
                return root
        raise FileNotFoundError('Cutie also needs cached encoder weights: ' + ', '.join(self.definition['backbones'])
                                + f'. Place them beside {self.checkpoint.name}; no automatic download is performed.')

    def _load(self):
        if self.model is not None:
            return
        self.device = self.check()
        import cutie
        import torch
        from hydra import compose, initialize_config_dir
        from omegaconf import open_dict
        from cutie.model.cutie import CUTIE

        root = Path(next(iter(cutie.__path__))) / 'config'
        with initialize_config_dir(version_base='1.3.2', config_dir=str(root.resolve())):
            self.config = compose(config_name='video_config')
        p, cfg = self.parameters, self.config
        cfg.max_internal_size = p['internal_size']
        cfg.mem_every = p['memory_every']
        cfg.use_long_term = p['long_term']
        cfg.max_mem_frames = p['memory_max']
        cfg.long_term.min_mem_frames = p['memory_min']
        cfg.long_term.max_mem_frames = p['memory_max']
        cfg.long_term.max_num_tokens = p['long_term_tokens']
        cfg.amp = p['mixed_precision']
        with open_dict(cfg.model):
            cfg.model.resnet_model_path = str(self.backbone_directory())
        logging.getLogger(__name__).info('Loading Cutie: %s on %s', self.checkpoint.name, self.device)
        self.model = CUTIE(cfg).eval().to(self.device)
        self.model.load_state_dict(torch.load(self.checkpoint, map_location=self.device, weights_only=True))

    def propagate(self, frames, references, *, direction, animals, cancelled=None):
        self._load()
        import torch
        from cutie.inference.inference_core import InferenceCore

        def check_cancel():
            if cancelled is not None and cancelled():
                raise InterruptedError('Segmentation cancelled')

        def image_tensor(rgb):
            return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).to(
                self.device, dtype=torch.float32).div_(255)

        def mask_tensor(mask):
            return torch.from_numpy(np.ascontiguousarray(mask)).to(self.device)

        processor = InferenceCore(self.model, cfg=self.config)
        objects = list(range(1, animals + 1))
        frames = iter(frames)
        try:
            first = next(frames)
        except StopIteration:
            return
        if first[0] not in references:
            raise ValueError('The first propagation frame needs a reference mask')
        device_type = self.device.split(':')[0]
        with torch.inference_mode(), torch.autocast(
                device_type=device_type, enabled=self.parameters['mixed_precision'] and device_type == 'cuda'):
            # Reference frames are appearance examples, not adjacent temporal states.
            for index, (rgb, mask) in sorted(references.items()):
                if index == first[0]:
                    continue
                check_cancel()
                processor.clear_sensory_memory()
                processor.step(image_tensor(rgb), mask_tensor(mask), objects=objects, force_permanent=True)
            processor.clear_sensory_memory()
            for index, rgb in chain((first,), frames):
                check_cancel()
                mask = references[index][1] if index in references else None
                probability = processor.step(image_tensor(rgb), None if mask is None else mask_tensor(mask),
                                             objects=objects if mask is not None else None,
                                             force_permanent=index == first[0])
                # Preserve reviewed key-frame pixels rather than resampling them through the model.
                output = mask.copy() if mask is not None else processor.output_prob_to_mask(
                    probability).cpu().numpy().astype(np.uint8)
                yield index, output

    def close(self):
        self.model = self.config = None
