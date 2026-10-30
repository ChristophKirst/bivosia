"""Check required data at the selected execution location before submitting a job."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from bivosia.dataset.analysis_inputs import processing_inputs
from bivosia.dataset.workflow import inspect_outputs
from bivosia.processing.jobs import require_idle


def check_inputs(dataset, operation, settings):
    require_idle(dataset)
    try:
        if not dataset.description.is_file():
            raise FileNotFoundError('Dataset description is missing')
        if operation in ('align', 'shape_features') and dataset.annotation is None:
            raise FileNotFoundError('Selected annotation is missing')
        if operation == 'vos':
            for path in (dataset.video, dataset.file('key_frames'), dataset.file('key_frame_masks')):
                if not path.is_file():
                    raise FileNotFoundError(f'Required input is missing: {path}')
            from bivosia.processing.segmentation import check
            check(dataset, **settings)
        else:
            processing_inputs(dataset, operation, settings)
    except FileNotFoundError as error:
        return {'missing': True, 'message': str(error)}
    except ValueError as error:
        if str(error).startswith('Set video in'):
            return {'missing': True, 'message': str(error)}
        raise
    replaces_analysis = operation == 'quantify' or (operation == 'workflow' and any(
        step['operation'] == 'quantify' for step in settings['steps']))
    existing = (any(entry['exists'] for entry in inspect_outputs(dataset)['analysis'].values())
                if replaces_analysis else False)
    return {'missing': False, 'replace_analysis': existing}
