"""Execute selected processing steps in dependency order at one location.

Examples
--------
>>> from bivosia.processing.workflow import required_products
>>> sorted(required_products([{'operation': 'quantify', 'settings': {}},
...                           {'operation': 'detect_link', 'settings': {}}]))
['masks']
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import logging

from bivosia.config import CONSENSUS_MASK_DEFAULTS, CONSENSUS_MASK_SOURCES, WORKFLOW_STEPS
from bivosia.dataset.layout import Dataset
from bivosia.processing.jobs import operation


logger = logging.getLogger(__name__)


def ordered_steps(steps):
    by_name = {step['operation']: step for step in steps}
    if not by_name or len(by_name) != len(steps) or set(by_name) - WORKFLOW_STEPS.keys():
        raise ValueError('Select each supported processing step at most once')
    return [by_name[name] for name in WORKFLOW_STEPS if name in by_name]


def validate_step_settings(steps):
    for step in ordered_steps(steps):
        name, values = step['operation'], step['settings']
        step_inputs(name, values)
        if name == 'shape_features' and values.get('radius', 0) <= 0:
            raise ValueError('Set the Zernike radius in Mask features settings before running')
        if name in ('quantify', 'align') and values.get('n_animals', 0) < (2 if name == 'quantify' else 1):
            raise ValueError(f"Set the animal count in {WORKFLOW_STEPS[name]['label']} settings")


def step_inputs(operation, settings=None):
    """Select required products using the requested mask source."""
    if operation == 'shape_features':
        source = (settings or {}).get('mask_source', CONSENSUS_MASK_DEFAULTS['source'])
        if source not in CONSENSUS_MASK_SOURCES:
            raise ValueError(f'Unknown mask source: {source}')
        return set(CONSENSUS_MASK_SOURCES[source]['inputs'])
    return set(WORKFLOW_STEPS[operation]['inputs'])


def _with_consensus(products):
    products = set(products)
    if any(set(CONSENSUS_MASK_SOURCES[source]['inputs']) <= products for source in ('export', 'annotation')):
        products.add('consensus_masks')
    return products


def required_products(steps):
    required, produced = set(), set()
    for step in ordered_steps(steps):
        definition = WORKFLOW_STEPS[step['operation']]
        required.update(step_inputs(step['operation'], step.get('settings')) - _with_consensus(produced | required))
        produced.add(definition['output'])
    return required


def step_availability(available, selected, *, settings=None):
    """Enable a step when saved inputs or selected predecessors supply its inputs."""
    products = set(available)
    enabled = {}
    for name, definition in WORKFLOW_STEPS.items():
        missing = step_inputs(name, (settings or {}).get(name)) - _with_consensus(products)
        enabled[name] = missing
        if not missing and name in selected:
            products.add(definition['output'])
    return enabled


def workflow_job(dataset, *, steps, progress=None, cancelled=None):
    from bivosia.dataset.workflow.processing import processing_inputs
    results = []
    validate_step_settings(steps)
    for index, step in enumerate(ordered_steps(steps)):
        if cancelled and cancelled():
            raise InterruptedError('Workflow cancelled')
        name, settings = step['operation'], step['settings']
        stage = WORKFLOW_STEPS[name]['label']
        logger.info('Workflow step %s/%s: %s', index + 1, len(steps), stage)
        dataset = Dataset(dataset.directory)
        if name != 'vos':
            processing_inputs(dataset, name, settings)
        if progress:
            progress({'stage': stage, 'step': index + 1, 'steps': len(steps), 'completed': 0, 'total': 0})
        def report(values):
            if progress:
                progress({**values, 'stage': stage, 'step': index + 1, 'steps': len(steps)})
        results.extend(operation(name)(dataset, **settings, progress=report, cancelled=cancelled))
    return results
