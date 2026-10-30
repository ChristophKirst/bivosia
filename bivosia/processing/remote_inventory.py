"""Collect a dataset's file inventory from mounted folders and registered SSH hosts."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path

from bivosia.dataset.inventory import FileInfo, scan_dataset
from bivosia.dataset.jsonio import read_json
from bivosia.dataset.layout import Dataset
from bivosia.dataset.locations import remote_locations
from bivosia.processing.remote import Remote


def scan_locations(dataset, profiles):
    """Keep inaccessible remote locations visible alongside the available files."""
    dataset = Dataset(dataset.directory)
    entries = list(scan_dataset(dataset))
    for location in remote_locations(dataset):
        host, path = location['host'], location['path']
        try:
            profile = next((p for p in profiles if p['host'] == host), None)
            if profile is None:
                raise ValueError(f'Register a BiVOSIA installation for {host}')
            remote = Remote(**{key: profile[key] for key in ('host', 'python', 'code')}, dataset=path)
            response = remote.call('inventory')
            for record in response['files']:
                record['relative'], record['path'] = Path(record['relative']), Path(record['path'])
                record['source'] = f"{host}:{record['source']}"
                record['host'] = host
                entries.append(FileInfo(**record))
        except Exception as error:
            entries.append(FileInfo(Path(), Path(path), True, error=str(error),
                                     source=f'{host}:{path}', host=host))
    return tuple(entries)


def inspect_locations(dataset, profiles, preferred_host=None, *, allow_partial=False, include_workflow=False):
    """Find a complete local or registered remote mask pair and its analysis outputs."""
    from bivosia.dataset.workflow import inspect_workflow

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
            from bivosia.dataset.inventory import FileMetadata
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
