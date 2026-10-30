"""Read and save per-step processing parameters in the dataset description.

Examples
--------
>>> from tempfile import TemporaryDirectory
>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.dataset.processing_settings import step_settings, save_step_settings
>>> with TemporaryDirectory() as directory:
...     dataset = Dataset(directory)
...     save_step_settings(dataset, 'quantify', {'n_animals': 3})
...     step_settings(dataset, 'quantify')['n_animals']
3
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from bivosia.config import ANALYSIS_DEFAULTS, PROCESSING_DEFAULTS, VOS_BACKENDS
from bivosia.dataset.jsonio import read_json
from bivosia.dataset.layout import Dataset
from bivosia.dataset.shape_features import shape_settings


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
        defaults = dict(n_animals=animals, workers=PROCESSING_DEFAULTS['quantify']['workers'] or 0)
    elif operation == 'detect_link':
        defaults = {key: value for key, value in ANALYSIS_DEFAULTS['zones_of_disagreement'].items()
                    if key != 'expand_to_baseline'}
    elif operation == 'shape_features':
        defaults = shape_settings()
    elif operation == 'align':
        defaults = dict(n_animals=animals, workers=PROCESSING_DEFAULTS['alignment']['workers'] or 0,
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
