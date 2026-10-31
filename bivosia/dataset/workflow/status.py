"""Read the files and headers available to a dataset workflow.

Examples
--------
>>> from tempfile import TemporaryDirectory
>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.dataset.workflow.status import inspect_workflow
>>> with TemporaryDirectory() as directory:
...     info = inspect_workflow(Dataset(directory))
...     bool(info['masks']['forward']['exists'])
False
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import closing
from pathlib import Path

from bivosia.analysis.mask_linking import prepare_links, validation_errors
from bivosia.analysis.mask_features import missing_mask_counts
from bivosia.analysis.zones_of_disagreement import revision_fingerprint
from bivosia.dataset.mask_quality import MaskOverlapReader
from bivosia.dataset.masks import matching_export
from bivosia.dataset.io.registry import file_inspector
from bivosia.dataset.io.json import read_json
from bivosia.dataset.annotations.masks import RevisionSession
from bivosia.dataset.mask_features import MaskShapeResults
from bivosia.dataset.storage.locations import remote_locations
from bivosia.processing.remote.client import Remote


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
        with closing(MaskOverlapReader(dataset.array('mask_intersections'), dataset.array('mask_areas'))) as counts:
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
        shapes = MaskShapeResults(dataset, feature_settings)
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
    from bivosia.dataset.io.interfaces import FileMetadata
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


def inspect_locations(dataset, profiles, preferred_host=None, *, allow_partial=False, include_workflow=False):
    """Find a complete local or registered remote mask pair and its analysis outputs."""
    result = inspect_workflow(dataset)
    result.update(mask_remote=None, analysis_remote=None, remote_errors=[])
    annotation_record = None
    if include_workflow and result['annotation'] is not None and result['annotation']['error']:
        annotation_record = read_json(result['annotation']['path'])
    def available(group):
        return all(item['exists'] and not item['error'] for item in group.values())
    local_masks = available(result['masks'])
    if local_masks and available(result['analysis']) and not include_workflow:
        return result
    candidates = []
    snapshots = []
    for location in remote_locations(dataset):
        host = location['host']
        try:
            profile = next((p for p in profiles if p['host'] == host), None)
            if profile is None:
                raise ValueError(f'Register a BiVOSIA installation for {host}')
            remote = Remote(**{key: profile[key] for key in ('host', 'python', 'code')}, dataset=location['path'])
            options = {'annotation_record': annotation_record} if annotation_record is not None else {}
            if include_workflow:
                options['feature_settings'] = dataset.settings.get('processing', {}).get('shape_features', {})
            outputs = remote.call('workflow' if include_workflow else 'outputs', **options)
            if include_workflow:
                required = {'annotation', 'annotation_validation'} if annotation_record is not None else {'annotation'}
                if not required.issubset(outputs):
                    raise ValueError('Remote workflow status is incomplete; update the remote BiVOSIA installation.')
                snapshots.append((remote, outputs))
            if available(outputs['masks']) or (allow_partial and any(
                    item['exists'] and not item['error'] for name in ('masks', 'analysis') for item in outputs[name].values())):
                candidates.append((remote, outputs))
        except Exception as error:
            result['remote_errors'].append(f'{host}: {error}')
    def select(items):
        selected = [item for item in items if item[0].host == preferred_host] or items
        if len(selected) > 1:
            result['remote_errors'].append('Multiple remote results are available; select an installation for this dataset.')
        return selected[0] if len(selected) == 1 else None
    def use(name, remote, entries):
        for entry in entries.values():
            entry['path'] = Path(entry['path']) if entry['path'] else None
            entry['host'] = remote.host
        result[name] = entries
    if not local_masks:
        selected = select(candidates)
        if selected:
            remote, outputs = selected
            result['mask_remote'] = remote
            use('masks', remote, outputs['masks'])
            if not available(result['analysis']) and (available(outputs['analysis']) or
                    (allow_partial and any(item['exists'] for item in outputs['analysis'].values()))):
                use('analysis', remote, outputs['analysis'])
                result['analysis_remote'] = remote
    elif not available(result['analysis']):
        selected = select([item for item in candidates if available(item[1]['analysis']) or
                           (allow_partial and any(entry['exists'] for entry in item[1]['analysis'].values()))])
        if selected:
            remote, outputs = selected
            result['analysis_remote'] = remote
            use('analysis', remote, outputs['analysis'])
    if include_workflow:
        if annotation_record is not None and result['analysis_remote'] is not None:
            for remote, snapshot in snapshots:
                if remote != result['analysis_remote']:
                    continue
                validation = snapshot['annotation_validation']
                if validation['fingerprint'] == result['annotation']['fingerprint']:
                    result['annotation'].update(validation)
        selected = select(snapshots)
        if selected:
            from bivosia.dataset.io.interfaces import FileMetadata
            remote, snapshot = selected
            if result['animals'] is None:
                result['animals'] = snapshot['animals']
            for name in ('key_frames', 'features', 'alignment'):
                if name == 'features' and result['annotation'] is not None:
                    selected_fingerprint = result['annotation']['fingerprint']
                    if (selected_fingerprint is None or snapshot['annotation'] is None
                            or snapshot['annotation']['fingerprint'] != selected_fingerprint):
                        continue
                if not available(result[name]) and available(snapshot[name]):
                    use(name, remote, snapshot[name])
                    if name == 'key_frames':
                        result['key_frame_count'] = snapshot['key_frame_count']
            for name in ('video', 'keypoints'):
                if not result[name]['exists'] and snapshot[name]['exists']:
                    item = dict(snapshot[name], host=remote.host)
                    item['path'] = Path(item['path'])
                    if item['metadata'] is not None:
                        item['metadata'] = FileMetadata(**item['metadata'])
                    result[name] = item
            if not result['annotations'] and not result['annotation_error'] and snapshot['annotations']:
                result['annotations'] = [Path(path) for path in snapshot['annotations']]
                result['annotation_error'] = snapshot['annotation_error']
                result['annotation'] = snapshot['annotation']
                if result['annotation'] is not None:
                    result['annotation'] = dict(result['annotation'], path=Path(result['annotation']['path']), host=remote.host)
            selected_annotation = result['annotation']
            remote_annotation = snapshot['annotation']
            same_annotation = (selected_annotation is not None and remote_annotation is not None
                               and selected_annotation['fingerprint'] is not None
                               and selected_annotation['fingerprint'] == remote_annotation['fingerprint'])
            if not result['export_current'] and snapshot['export_current'] and same_annotation:
                result['export'] = dict(snapshot['export'], host=remote.host)
                result['export']['path'] = Path(result['export']['path'])
                result['export_current'], result['export_error'] = True, snapshot['export_error']
    return result
