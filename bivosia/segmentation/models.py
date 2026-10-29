"""Shared model configuration and lazy backend construction."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from abc import ABC, abstractmethod
from copy import deepcopy
from importlib import import_module
from pathlib import Path


class Segmenter(ABC):
    """Own a backend's checkpoint, device and independent parameter values."""

    name: str
    backends: dict

    def __init__(self, checkpoint=None, device='auto', parameters=None):
        self.checkpoint = Path(checkpoint or self.definition['checkpoint']).expanduser()
        self.device = device
        self.parameters = deepcopy(self.definition['defaults'])
        self.parameters.update(parameters or {})

    @property
    def definition(self):
        return self.backends[self.name]

    @property
    def label(self):
        return self.definition['label']

    @property
    def metadata(self):
        """Describe the model used to produce an annotation."""
        return {'name': self.name, 'checkpoint': self.checkpoint.name}

    def check_parameters(self):
        unknown = set(self.parameters) - self.definition['defaults'].keys()
        if unknown:
            raise ValueError(f'Unknown {self.label} parameters: {unknown}')

    @abstractmethod
    def check(self):
        """Validate dependencies and settings; return the resolved device."""

    @abstractmethod
    def close(self):
        """Release model resources."""


def backend_class(backends, model):
    if model not in backends:
        raise ValueError(f'Unknown segmentation backend: {model}')
    module, name = backends[model]['class']
    return getattr(import_module(module), name)


def create_segmenter(backends, model, checkpoint=None, device='auto', parameters=None):
    return backend_class(backends, model)(checkpoint, device, parameters)


def model_availability(backends, model, checkpoint=None, device='auto', parameters=None):
    """Check dependencies and settings without loading model weights."""
    try:
        segmenter = create_segmenter(backends, model, checkpoint, device, parameters)
        try:
            return segmenter.check(), ''
        finally:
            segmenter.close()
    except Exception as error:
        return None, str(error)
