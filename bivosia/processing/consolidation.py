"""Plan and copy a dataset into one location after resolving conflicting files."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import json

from bivosia.dataset.consolidation import consolidated_settings, file_signatures
from bivosia.dataset.file_operations import copy_bytes, file_state
from bivosia.dataset.jobs import is_job_path
from bivosia.dataset.paths import file_path
from bivosia.processing.file_operations import _remote, _writer, copy_selected_file
from bivosia.processing.jobs import require_idle
from bivosia.processing.remote_inventory import scan_locations


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
