"""Copy, move or delete a selected dataset category or job folder across locations."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from collections import defaultdict
from pathlib import Path

from bivosia.dataset.folder_operations import check_contents, create_folders, delete_contents
from bivosia.dataset.layout import Dataset
from bivosia.dataset.jobs import completed_job_folders, is_job_path
from bivosia.dataset.paths import file_name, folder_path
from bivosia.processing.consolidation import _at_destination, plan_consolidation
from bivosia.processing.file_operations import _remote, copy_selected_file
from bivosia.processing.jobs import require_idle


def folder_entries(entries, relative):
    """Select the listed subtree, rejecting unavailable contents and dataset roots."""
    relative = Path(relative)
    if not relative.parts or relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Select a dataset category or subfolder')
    selected = tuple(entry for entry in entries if entry.relative.is_relative_to(relative))
    errors = [entry.error for entry in entries if entry.error and
              (entry.relative.is_relative_to(relative) or relative.is_relative_to(entry.relative))]
    if errors:
        raise ValueError('\n'.join(errors))
    return selected


def contents_on_host(dataset, action, *, files=(), folders=(), root=None, relatives=()):
    """Check job activity once before a batch changes job settings or status files."""
    require_idle(dataset)
    for location in dataset.roots:
        if location != dataset.directory:
            require_idle(Dataset(location))
    if action == 'check_contents':
        return check_contents(dataset, files, folders)
    if action == 'completed_job_folders':
        return completed_job_folders(dataset, files)
    if action == 'delete_contents':
        delete_contents(dataset, files, folders)
        return {}
    if action == 'create_folders':
        require_idle(Dataset(root))
        create_folders(root, relatives)
        return {}
    raise ValueError(f'Unknown folder operation: {action}')


def _requests(entries, states=None):
    by_host = defaultdict(lambda: {'files': [], 'folders': []})
    for entry in entries:
        kind = 'folders' if entry.is_directory else 'files'
        expected = ({'modified': entry.modified} if entry.is_directory else
                    {'size': entry.size, 'modified': entry.modified})
        if states is not None:
            expected = states[entry.host][kind][str(entry.path)]
        by_host[entry.host][kind].append({'path': str(entry.path), 'expected': expected})
    return by_host


def _call(dataset, host, profiles, action, **values):
    remote = _remote(dataset, host, profiles)
    return (remote.call(action, timeout=None, **values) if remote else
            contents_on_host(Dataset(dataset.directory), action, **values))


def check_folder_entries(dataset, entries, profiles):
    """Validate all participating hosts before any removal begins."""
    return {host: _call(dataset, host, profiles, 'check_contents', **values)
            for host, values in _requests(entries).items()}


def delete_folder_entries(dataset, entries, profiles, progress=None):
    """Delete exactly the confirmed files and their empty folders at each location."""
    states = check_folder_entries(dataset, entries, profiles)
    _delete_checked(dataset, entries, states, profiles, progress)


def completed_job_entries(dataset, entries, profiles):
    """Select complete job subtrees from the displayed local and remote listings."""
    statuses = [entry for entry in entries if not entry.is_directory and
                entry.relative.name == file_name('job_status') and
                entry.relative.parent.parent == folder_path('jobs')]
    roots = {host: tuple(Path(path) for path in _call(dataset, host, profiles,
                         'completed_job_folders', files=values['files']))
             for host, values in _requests(statuses).items()}
    return tuple(entry for entry in entries if any(entry.path.is_relative_to(root)
                                                  for root in roots.get(entry.host, ())))


def _delete_checked(dataset, entries, states, profiles, progress):
    for host, values in _requests(entries, states).items():
        if progress:
            progress(f'Removing selected contents on {host or "Local"}…')
        _call(dataset, host, profiles, 'delete_contents', **values)


def plan_folder_transfer(dataset, entries, location, profiles, progress=None):
    """Reuse consolidation's conflict comparison for the selected subtree only."""
    entries = _transferable(entries)
    return plan_consolidation(dataset, location, profiles, progress,
                              required={str(entry.relative) for entry in entries if not entry.is_directory},
                              entries=entries)


def _transferable(entries):
    result = tuple(entry for entry in entries if not is_job_path(entry.relative))
    if not result:
        raise ValueError('Job records stay at their execution location; delete them when no longer needed')
    return result


def transfer_folder(dataset, entries, plan, choices, profiles, *, move=False, progress=None):
    """Copy all chosen versions before removing matching source copies for a move.

    Differing, unchosen versions remain at their original locations. Empty folders
    are preserved at the destination and removed from sources only when empty.
    """
    if any(name not in choices for name in plan.conflicts):
        raise ValueError('Resolve all file conflicts before transferring the folder')
    entries = _transferable(entries)
    states = check_folder_entries(dataset, entries, profiles)
    if any(value['link'] for host in states.values() for value in host['folders'].values()):
        raise ValueError('Folder transfers do not follow directory links')
    for (host, path), signature in plan.signatures.items():
        if states[host]['files'][path] != signature['state']:
            raise ValueError(f'File changed after comparison; refresh the listing: {path}')
    location = plan.location
    destination = _remote(dataset, location['host'], profiles, location['path'])
    folders = sorted({str(entry.relative) for entry in entries if entry.is_directory})
    values = {'root': location['path'], 'relatives': folders}
    if destination:
        destination.call('create_folders', **values)
    else:
        contents_on_host(Dataset(location['path']), 'create_folders', **values)
    removals = []
    targets = []
    for name in sorted(plan.files):
        copies = plan.files[name]
        chosen = copies[choices.get(name, plan.defaults[name])]
        previous = next((entry for entry in copies if _at_destination(entry, location)), None)
        if not _at_destination(chosen, location):
            if progress:
                progress(f'Copying {name}…')
            replace = states[previous.host]['files'][str(previous.path)] if previous else None
            copy_selected_file(dataset, chosen, location, profiles, replace_expected=replace,
                               source_expected=states[chosen.host]['files'][str(chosen.path)])
        targets.append({'path': str(Path(location['path']) / name), 'expected': {
            'size': chosen.size, 'modified': chosen.modified}})
        if move:
            digest = plan.signatures.get((chosen.host, str(chosen.path)), {}).get('sha256')
            removals.extend(entry for entry in copies if not _at_destination(entry, location) and
                            (entry == chosen or plan.signatures[(entry.host, str(entry.path))]['sha256'] == digest))
    if move:
        # Recheck destination files before removing any source, including reused copies.
        if destination:
            destination.call('check_contents', files=targets)
        else:
            contents_on_host(Dataset(location['path']), 'check_contents', files=targets)
        removals.extend(entry for entry in entries if entry.is_directory and not _at_destination(entry, location))
        _delete_checked(dataset, removals, states, profiles, progress)
        current = Dataset(dataset.directory)
        reference = current.settings.get('video')
        if reference and Path(reference).is_absolute():
            video = next((entry for entry in removals if not entry.host and not entry.is_directory and
                          entry.path == Path(reference).expanduser().resolve()), None)
            if video is not None:
                current.settings['video'] = video.relative.relative_to(folder_path('recording')).as_posix()
                current.save()
    if progress:
        progress('Folder moved' if move else 'Folder copied')
