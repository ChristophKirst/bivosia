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


def processing_inputs(dataset, operation, settings):
    """Validate the inputs on the host selected for a processing operation."""
    if operation in ('workflow', 'detect_link', 'export'):
        from bivosia.config import WORKFLOW_STEPS
        from bivosia.dataset.processing_inputs import require_products
        from bivosia.processing.workflow import required_products, validate_step_settings
        if operation == 'workflow':
            validate_step_settings(settings['steps'])
        products = required_products(settings['steps']) if operation == 'workflow' else WORKFLOW_STEPS[operation]['inputs']
        require_products(dataset, products)
        detection = (next((step['settings'] for step in settings['steps'] if step['operation'] == 'detect_link'), None)
                     if operation == 'workflow' else settings if operation == 'detect_link' else None)
        if detection is not None and 'analysis' in products and detection.get('direction_method', 'temporal_quality') == 'temporal_quality':
            if not dataset.array('mask_temporal_quality').is_file():
                raise FileNotFoundError('Temporal quality is missing; run Mask analysis first')
        return {'animals': None, 'annotations': [], 'annotation': dataset.settings.get('annotation'), 'node_names': []}
    if operation == 'shape_features':
        from bivosia.dataset.shape_features import shape_analysis_inputs
        return shape_analysis_inputs(dataset, settings)
    if operation == 'quantify':
        from contextlib import ExitStack
        from bivosia.dataset.masks import open_masks
        with ExitStack() as stack:
            readers = [stack.enter_context(open_masks(dataset.mask(name), direction=name))
                       for name in ('forward', 'reverse')]
            if readers[0].shape != readers[1].shape:
                raise ValueError('Forward/reverse masks must have the same shape')
    return analysis_inputs(dataset, keypoints=operation == 'align')
