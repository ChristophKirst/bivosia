"""Read the files and headers available to a dataset workflow.

Examples
--------
>>> from tempfile import TemporaryDirectory
>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.dataset.workflow import inspect_workflow
>>> with TemporaryDirectory() as directory:
...     info = inspect_workflow(Dataset(directory))
...     bool(info['masks']['forward']['exists'])
False
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import closing

from bivosia.analysis.linking import prepare_links, validation_errors
from bivosia.analysis.mask_features import missing_mask_counts
from bivosia.analysis.zones_of_disagreement import revision_fingerprint
from bivosia.dataset.arrays import MaskOverlaps
from bivosia.dataset.consensus_masks import matching_export
from bivosia.dataset.formats import file_inspector
from bivosia.dataset.jsonio import read_json
from bivosia.dataset.session import RevisionSession
from bivosia.dataset.shape_features import ShapeResults


def _inspect(resolve):
    try:
        path = resolve()
        exists = path.is_file()
        inspector = file_inspector(path) if exists else None
        metadata = inspector(path) if inspector is not None else None
        return {'path': path, 'exists': exists, 'metadata': metadata, 'error': ''}
    except Exception as error:
        return {'path': None, 'exists': False, 'metadata': None, 'error': str(error)}


def inspect_annotation(dataset, record):
    """Validate saved choices and links using counts, without reading mask pixels."""
    result = {'valid': False, 'issues': [], 'error': '', 'fingerprint': None}
    try:
        result['fingerprint'] = revision_fingerprint(
            record['header'], record['zones'], record['links'], record.get('link_frames', []))
        with closing(MaskOverlaps(dataset.array('mask_intersections'), dataset.array('mask_areas'))) as counts:
            missing_forward, missing_reverse = missing_mask_counts(counts.areas)
            session = RevisionSession.from_record(record, missing_forward, missing_reverse, counts.shape[1])
        result['issues'] = validation_errors(session, prepare_links(session))
        result['valid'] = not result['issues']
    except Exception as error:
        result['error'] = str(error)
    return result


def inspect_workflow(dataset, *, feature_settings=None):
    """Inspect saved outputs and validate the selected annotation independently of export."""
    result = {'video': _inspect(lambda: dataset.video)}
    result['key_frames'] = {name: _inspect(lambda: dataset.file(name))
                            for name in ('key_frames', 'key_frame_masks')}
    result.update(inspect_outputs(dataset))
    if feature_settings is None:
        feature_settings = dataset.settings.get('processing', {}).get('shape_features', {})
    result['features'] = {}
    try:
        shapes = ShapeResults(dataset, feature_settings)
    except Exception as error:
        shapes, shape_error = None, str(error)
    for name in ('mask_zernike', 'mask_umap'):
        if shapes is None:
            result['features'][name] = {'path': None, 'exists': False, 'metadata': None, 'error': shape_error}
            continue
        entry = result['features'][name] = _inspect(lambda: shapes.path(name))
        if entry['exists'] and not entry['error']:
            try:
                shapes.validate_saved(name, infer_parameters=True)
            except Exception as error:
                entry['error'] = str(error)
    result['alignment'] = {'keypoint_alignment': _inspect(lambda: dataset.array('keypoint_alignment'))}
    result['keypoints'] = _inspect(dataset.keypoint_file)
    result['animals'] = None
    metadata = result['key_frames']['key_frame_masks']['metadata']
    result['key_frame_count'] = next((shape[0] for name, shape, _ in metadata.arrays
                                    if name == 'frames' and len(shape) == 1), None) if metadata else None
    if result['key_frames']['key_frames']['exists']:
        try:
            result['animals'] = read_json(dataset.file('key_frames'))['animals']
        except Exception as error:
            result['key_frames']['key_frames']['error'] = str(error)
    if result['animals'] is None and dataset.file('mask_quality_metadata').is_file():
        try:
            result['animals'] = int(read_json(dataset.file('mask_quality_metadata'))['n_animals'])
        except (KeyError, ValueError, OSError):
            pass
    try:
        result['annotations'] = dataset.annotation_files
        result['annotation_error'] = ''
    except Exception as error:
        result['annotations'], result['annotation_error'] = [], str(error)
    result['annotation'] = None
    saved = None
    try:
        annotation = dataset.annotation
        if annotation is not None:
            saved = read_json(annotation)
            result['annotation'] = dict(inspect_annotation(dataset, saved), path=annotation)
    except Exception as error:
        result['annotation_error'] = str(error)
    result['export'] = None
    result['export_error'] = ''
    result['export_current'] = False
    try:
        exported = saved.get('export') if saved else None
        if exported:
            result['export'] = _inspect(lambda: dataset.export_path(exported['path']))
            result['export_current'] = matching_export(dataset, saved) is not None
    except Exception as error:
        result['export_error'] = str(error)
    return result


def workflow_snapshot(dataset, *, annotation_record=None, feature_settings=None):
    """Serialize workflow file headers and annotation/export status for SSH."""
    from dataclasses import asdict
    from pathlib import Path
    from bivosia.dataset.inventory import FileMetadata
    def encode(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, FileMetadata):
            return asdict(value)
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [encode(item) for item in value]
        return value
    result = inspect_workflow(dataset, feature_settings=feature_settings)
    if annotation_record is not None:
        result['annotation_validation'] = inspect_annotation(dataset, annotation_record)
    return encode(result)


def inspect_outputs(dataset):
    """Inspect mask and quantification headers at one dataset location."""
    result = {}
    result['masks'] = {name: _inspect(lambda: dataset.mask(name))
                       for name in ('forward', 'reverse')}
    result['analysis'] = {name: _inspect(lambda: dataset.array(name)) for name in (
        'mask_intersections', 'mask_areas', 'mask_iou_match', 'mask_quality',
        'mask_missing', 'mask_switch', 'mask_quality_metadata')}
    return result
