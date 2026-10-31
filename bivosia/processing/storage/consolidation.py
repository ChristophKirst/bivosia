"""Plan and copy datasets or required job inputs after resolving conflicting files."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import json

from bivosia.config import CONSENSUS_MASK_SOURCES
from bivosia.dataset.io.json import read_json
from bivosia.dataset.layout import Dataset
from bivosia.dataset.paths import file_path, folder_path, folder_suffixes, format_paths
from bivosia.dataset.storage.consolidation import consolidated_settings, file_signatures
from bivosia.dataset.storage.file_operations import copy_bytes, file_state
from bivosia.dataset.storage.inventory import scan_locations
from bivosia.dataset.workflow.jobs import is_job_path
from bivosia.dataset.workflow.status import inspect_locations
from bivosia.processing.storage.file_operations import _remote, _writer, copy_selected_file
from bivosia.processing.jobs import require_idle
from bivosia.processing.workflow import required_products, step_inputs


@dataclass
class ConsolidationPlan:
    location: dict
    files: dict
    signatures: dict
    defaults: dict
    conflicts: tuple
    complete: bool = True


def plan_consolidation(dataset, location, profiles, progress=None, *, required=None, entries=None):
    """List conflicts before copying, hashing only files with multiple physical copies."""
    require_idle(dataset)
    if progress:
        progress('Checking dataset locations…')
    remote = _remote(dataset, location['host'], profiles, location['path'])
    location = dict(location, path=remote.call('location')['path'] if remote else str(Path(location['path']).resolve()))
    entries = scan_locations(dataset, profiles) if entries is None else entries
    errors = [entry.error for entry in entries if entry.error]
    if errors:
        raise ValueError('\n'.join(errors))
    files = defaultdict(list)
    for entry in entries:
        if (not entry.is_directory and not is_job_path(entry.relative) and
                (required is None or str(entry.relative) in required)):
            files[entry.relative.as_posix()].append(entry)
    if required is not None:
        description = str(file_path('description'))
        if any(_at_destination(entry, location) for entry in files.get(description, ())):
            files.pop(description, None)
    signatures = {}
    by_host = defaultdict(list)
    for name, copies in files.items():
        for entry in copies:
            if len(copies) > 1 or entry.relative == file_path('description'):
                by_host[entry.host].append(entry)
    for host, copies in by_host.items():
        if progress:
            progress(f'Comparing files on {host or "Local"}…')
        remote = _remote(dataset, host, profiles)
        requests = [{'path': str(entry.path), 'relative': str(entry.relative),
                     'expected': {'size': entry.size, 'modified': entry.modified}} for entry in copies]
        response = (remote.call('file_signatures', timeout=None, files=requests) if remote else
                    file_signatures(dataset, requests))
        signatures.update({(host, path): value for path, value in response.items()})
    defaults, conflicts = {}, []
    for name, copies in files.items():
        target = next((i for i, entry in enumerate(copies) if _at_destination(entry, location)), 0)
        defaults[name] = target
        if len(copies) > 1 and len({signatures[(entry.host, str(entry.path))]['sha256'] for entry in copies}) > 1:
            conflicts.append(name)
    return ConsolidationPlan(location, dict(files), signatures, defaults, tuple(conflicts), complete=required is None)


def plan_input_sync(dataset, location, profiles, operation, *, annotation=None, settings=None, progress=None):
    """Include only the inputs needed by the requested calculation, retaining target settings."""
    if annotation:
        dataset = Dataset(dataset.directory)
        dataset.settings['annotation'] = annotation
    entries = scan_locations(dataset, profiles)
    files = {entry.relative.as_posix() for entry in entries if not entry.is_directory and not entry.error}
    products = (required_products(settings['steps']) if operation == 'workflow' else
                step_inputs(operation, settings))
    if 'consensus_masks' in products:
        info = inspect_locations(dataset, profiles, location.get('host') or None, include_workflow=True)
        exported = info['export']
        source = ('export' if exported and exported['exists'] and not exported['error']
                  and info['export_current'] and not info['export_error'] else 'annotation')
        products.update(CONSENSUS_MASK_SOURCES[source]['inputs'])
    required = set()
    if 'video' in products:
        video = dataset.settings.get('video')
        if video:
            video = Path(video).name if Path(video).is_absolute() else video
            required.add(str(file_path('video', filename=video)))
        else:
            videos = [name for name in files if Path(name).parent == folder_path('recording')]
            if len(videos) != 1:
                raise ValueError('Select the experimental video in the dataset before synchronizing')
            required.add(videos[0])
    if 'key_frames' in products:
        required.update(str(file_path(key)) for key in ('key_frames', 'key_frame_masks'))
    if 'masks' in products:
        for direction in ('forward', 'reverse'):
            selected = dataset.settings.get('masks', {}).get(direction)
            names = ([str(folder_path('masks') / selected)] if selected else
                     [str(file_path(direction).with_suffix(suffix)) for suffix in folder_suffixes('masks')])
            present = [name for name in names if name in files]
            if len(present) != 1:
                raise ValueError(f'Select one available {direction} mask file in the dataset; found {len(present)}')
            name = present[0]
            required.add(name)
            if Path(name).suffix == '.npy':
                companion = str(format_paths('masks', name, 'companions', format='npy')['validity'])
                if companion in files:
                    required.add(companion)
    if 'analysis' in products:
        required.update(str(file_path(key)) for key in ('mask_intersections', 'mask_areas', 'mask_iou_match',
                        'mask_quality', 'mask_missing', 'mask_switch', 'mask_quality_metadata'))
        temporal = str(file_path('mask_temporal_quality'))
        if temporal in files:
            required.add(temporal)
    if 'keypoints' in products:
        required.add(dataset.relative(dataset.keypoint_file()))
    if {'annotations', 'export'} & products:
        selected = annotation or dataset.settings.get('annotation')
        if selected is None:
            annotations = [name for name in files if Path(name).parent == folder_path('annotations')
                           and name.endswith('.json')]
            if len(annotations) != 1:
                raise ValueError('Select an annotation before synchronizing analysis inputs')
            selected = annotations[0]
        selected = dataset.relative(selected)
        required.add(selected)
        if 'export' in products:
            found_export = False
            for entry in entries:
                if entry.relative.as_posix() != selected or entry.is_directory:
                    continue
                remote = _remote(dataset, entry.host, profiles)
                record = (remote.call('read_json', path=str(entry.path),
                                     expected={'size': entry.size, 'modified': entry.modified})['value']
                          if remote else read_json(entry.path))
                if record.get('export'):
                    found_export = True
                    required.add(str(folder_path('masks') / record['export']['path']))
            if not found_export:
                raise ValueError('The selected annotation has no exported masks to synchronize')
    missing = required - files
    if missing:
        raise FileNotFoundError('Required files are unavailable in the registered locations: ' + ', '.join(sorted(missing)))
    # A new location needs a portable description; existing location settings are preserved.
    required.add(str(file_path('description')))
    return plan_consolidation(dataset, location, profiles, progress, required=required, entries=entries)


def _at_destination(entry, location):
    return entry.host == location['host'] and entry.path == Path(location['path']) / entry.relative


def consolidate(dataset, plan, choices, profiles, progress=None):
    """Copy the chosen versions, then publish a self-contained dataset description."""
    require_idle(dataset)
    if any(name not in choices for name in plan.conflicts):
        raise ValueError('Resolve all conflicts before consolidating')
    description = file_path('description').as_posix()
    selections = {name: copies[choices.get(name, plan.defaults[name])] for name, copies in plan.files.items()}
    for name, entry in selections.items():
        if name == description or _at_destination(entry, plan.location):
            continue
        previous = next((candidate for candidate in plan.files[name] if _at_destination(candidate, plan.location)), None)
        replace = plan.signatures[(previous.host, str(previous.path))]['state'] if previous else None
        if progress:
            progress(f'Copying {name}…')
        copy_selected_file(dataset, entry, plan.location, profiles, replace_expected=replace,
                           source_expected=plan.signatures.get((entry.host, str(entry.path)), {}).get('state'))
    if not plan.complete and description not in selections:
        if progress:
            progress('Inputs synchronized')
        return
    # Dataset metadata is specific to its destination; retain all analysis settings.
    selected = selections.get(description)
    settings = (plan.signatures[(selected.host, str(selected.path))]['settings'] if selected else
                consolidated_settings(dataset.settings))
    existing = next((entry for entry in plan.files.get(description, ()) if _at_destination(entry, plan.location)), None)
    replace = plan.signatures[(existing.host, str(existing.path))]['state'] if existing else None
    remote = _remote(dataset, plan.location['host'], profiles, plan.location['path'])
    with TemporaryDirectory(prefix='bivosia-description-') as directory:
        path = Path(directory) / file_path('description')
        path.write_text(json.dumps(settings, indent=2) + '\n', encoding='utf-8')
        state = file_state(path)
        with path.open('rb') as source, _writer(remote, plan.location['path'], description, state, replace) as (writer, publish):
            publish(copy_bytes(source, writer, state['size']))
    if progress:
        progress('Dataset consolidated')
