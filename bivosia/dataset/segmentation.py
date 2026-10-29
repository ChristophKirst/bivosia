"""Write source-indexed VOS masks without replacing existing results."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import contextmanager

from bivosia.analysis.segmentation import validate_coverage
from bivosia.dataset.mask_output import mask_outputs


def output_paths(dataset, directions):
    return {direction: dataset.mask(direction, write=True) for direction in directions}


@contextmanager
def vos_outputs(dataset, directions, count, shape, crop):
    """Publish masks in the selected formats only after crop coverage is complete."""
    with mask_outputs(output_paths(dataset, directions), (count, *shape),
                      metadata={'crop': crop}) as outputs:
        yield outputs
        for masks, valid in outputs.values():
            validate_coverage(valid[:], crop)
