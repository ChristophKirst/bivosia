"""Read dataset metadata for mask-feature and keypoint-alignment settings.

Examples
--------
>>> from bivosia.dataset.analysis_inputs import analysis_inputs
>>> from bivosia.dataset.layout import Dataset
>>> settings = analysis_inputs(Dataset('/path/to/dataset'))  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from bivosia.dataset.jsonio import read_json
from bivosia.dataset.keypoints import open_keypoints
from bivosia.dataset.paths import auxiliary_path


def analysis_inputs(dataset, *, keypoints=False):
    """Read animal counts, annotation paths and optional keypoint names."""
    for direction in ('forward', 'reverse'):
        path = dataset.mask(direction)
        if not path.is_file():
            raise FileNotFoundError(f'Missing {direction} masks: {path}')
    animals = None
    for name, field in (('mask_quality_metadata', 'n_animals'), ('key_frames', 'animals')):
        path = dataset.file(name)
        if path.is_file():
            animals = read_json(path)[field]
            break
    result = {'animals': animals, 'annotations': [], 'annotation': None, 'node_names': []}
    if keypoints:
        paths = dataset.annotation_files
        result['annotations'] = [dataset.relative(path) for path in paths]
        selected = dataset.settings.get('annotation')
        if selected:
            result['annotation'] = dataset.relative(dataset.annotation_path(selected))
        elif len(paths) == 1:
            result['annotation'] = result['annotations'][0]
        result['node_names'] = list(open_keypoints(dataset).node_names)
    return result


def require_new_outputs(paths):
    """Reject saved or incomplete outputs before starting a new calculation."""
    for path in paths:
        for candidate in (path, auxiliary_path(path, 'partial')):
            if candidate.exists():
                raise FileExistsError(f'Output already exists: {candidate}')
