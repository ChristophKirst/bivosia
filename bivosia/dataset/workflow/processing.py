"""Load processing settings and validate dataset inputs and output destinations.

Examples
--------
>>> from tempfile import TemporaryDirectory
>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.dataset.workflow.processing import step_settings, save_step_settings
>>> with TemporaryDirectory() as directory:
...     dataset = Dataset(directory)
...     save_step_settings(dataset, 'quantify', {'n_animals': 3})
...     step_settings(dataset, 'quantify')['n_animals']
3
"""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from bivosia.config import ANALYSIS_DEFAULTS, PROCESSING_DEFAULTS, VOS_BACKENDS
from bivosia.dataset.io.json import read_json
from bivosia.dataset.keypoints import open_keypoints
from bivosia.dataset.layout import Dataset
from bivosia.dataset.mask_features import shape_settings
from bivosia.dataset.paths import auxiliary_path


# Processing settings

def step_settings(dataset, operation):
    current = Dataset(dataset.directory)
    saved = current.settings.get('processing', {}).get(operation, {})
    animals = 2
    for name, key in (('mask_quality_metadata', 'n_animals'), ('key_frames', 'animals')):
        path = current.file(name)
        if path.is_file():
            animals = int(read_json(path)[key])
            break
    if operation == 'vos':
        model = saved.get('model', next(iter(VOS_BACKENDS)))
        defaults = dict(model=model, checkpoint='', device='auto', direction='both', references='start',
                        output_scale=1., parameters=dict(VOS_BACKENDS[model]['defaults']))
    elif operation == 'quantify':
        defaults = dict(n_animals=animals, workers=PROCESSING_DEFAULTS['analysis']['mask_quality']['workers'] or 0)
    elif operation == 'detect_link':
        defaults = {key: value for key, value in ANALYSIS_DEFAULTS['zones_of_disagreement'].items()
                    if key != 'expand_to_baseline'}
    elif operation == 'shape_features':
        defaults = shape_settings()
    elif operation == 'align':
        defaults = dict(n_animals=animals, workers=PROCESSING_DEFAULTS['analysis']['alignment']['workers'] or 0,
                        fallback_previous=ANALYSIS_DEFAULTS['alignment']['fallback_previous'], exclude_keypoints=[])
    elif operation == 'export':
        defaults = {'destination': ''}
    else:
        raise ValueError(f'Unknown processing step: {operation}')
    result = {**defaults, **saved}
    if operation == 'vos':
        result['parameters'] = {**defaults['parameters'], **saved.get('parameters', {})}
    return result


def save_step_settings(dataset, operation, values):
    current = Dataset(dataset.directory)
    current.settings.setdefault('processing', {})[operation] = values
    current.save()


def execution_settings(operation, values):
    result = dict(values)
    if 'workers' in result:
        result['workers'] = result['workers'] or None
    if operation == 'export' and not result.get('destination'):
        result.pop('destination', None)
    if operation == 'vos' and not result.get('checkpoint'):
        result.pop('checkpoint', None)
    return result


# Required products and dataset metadata

def require_products(dataset, products):
    paths = []
    if 'consensus_masks' in products:
        from bivosia.dataset.masks import consensus_metadata
        metadata = consensus_metadata(dataset)
        paths.extend(dataset.path(item['path']) for item in metadata['source_files'])
    if 'video' in products:
        try:
            paths.append(dataset.video)
        except ValueError as error:
            raise FileNotFoundError(str(error)) from error
    if 'key_frames' in products:
        paths.extend(dataset.file(key) for key in ('key_frames', 'key_frame_masks'))
    if 'masks' in products:
        paths.extend(dataset.mask(direction) for direction in ('forward', 'reverse'))
    if 'analysis' in products:
        paths.extend(dataset.array(key) for key in ('mask_intersections', 'mask_areas', 'mask_iou_match',
                     'mask_quality', 'mask_missing', 'mask_switch', 'mask_quality_metadata'))
    if 'keypoints' in products:
        paths.append(dataset.keypoint_file())
    if {'annotations', 'export', 'consensus_masks'} & set(products):
        annotation = dataset.annotation
        if annotation is None:
            raise FileNotFoundError('Select a ZoD annotation')
        paths.append(annotation)
        if 'export' in products:
            record = read_json(annotation)
            if not record.get('export'):
                raise FileNotFoundError('Export consensus masks before calculating features')
            paths.append(dataset.export_path(record['export']['path']))
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Required input is missing: {path}')
    return paths


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


# Input and output validation

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
        from bivosia.dataset.mask_features import shape_analysis_inputs
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
