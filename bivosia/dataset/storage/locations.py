"""Manage local storage roots and SSH dataset locations.

Examples
--------
>>> from bivosia.dataset.storage.locations import remote_path
>>> remote_path('/data/experiments/trial')
'/data/experiments/trial'
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path, PurePosixPath
import re

from bivosia.dataset.io.json import atomic_json, read_json


def remote_path(value):
    """Require an absolute server path rather than a file-transfer URL."""
    value = str(value).strip()
    path = PurePosixPath(value)
    if not path.is_absolute() or '..' in path.parts or '\x00' in value or '://' in value:
        raise ValueError('Use an absolute workstation folder, such as /data/trial, not an sftp:// URL')
    return str(path)


def remote_locations(dataset):
    """Read current locations from disk so independently opened windows agree."""
    settings = read_json(dataset.description) if dataset.description.is_file() else dataset.settings
    return _validated_locations(settings.get('remote_locations', []))


def _validated_locations(entries):
    if not isinstance(entries, list):
        raise ValueError('remote_locations must be a list of host/path entries')
    result = []
    for entry in entries:
        host = entry['host']
        if not isinstance(host, str) or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@:-]*', host):
            raise ValueError('Remote location host must be an SSH alias or user@host')
        if any(item['host'] == host for item in result):
            raise ValueError(f'Several dataset locations are configured for {host}')
        result.append({'host': host, 'path': remote_path(entry['path'])})
    return result


def location_for(dataset, profile):
    """Resolve an explicit dataset location or an installation's default folder."""
    for entry in remote_locations(dataset):
        if entry['host'] == profile['host']:
            return entry['path']
    base = profile.get('data_folder', '')
    return str(PurePosixPath(remote_path(base)) / dataset.directory.name) if base else ''


def save_locations(dataset, storage_folders, locations):
    """Update only locations in the latest description, preserving other settings."""
    settings = read_json(dataset.description) if dataset.description.is_file() else dict(dataset.settings)
    local = []
    seen = {dataset.directory}
    for value in storage_folders:
        if '://' in str(value):
            raise ValueError('Use a mounted filesystem path or add a remote location')
        path = (dataset.directory / Path(value).expanduser()).resolve()
        if path not in seen:
            local.append(str(value))
            seen.add(path)
    settings['storage_folders'] = local
    settings['remote_locations'] = _validated_locations(locations)
    for key in ('storage_folders', 'remote_locations'):
        if not settings[key]:
            settings.pop(key)
    atomic_json(dataset.description, settings)
    dataset.reload()


def remember_location(dataset, host, path):
    """Persist an automatically derived destination for this dataset."""
    entries = remote_locations(dataset)
    if {'host': host, 'path': remote_path(path)} in entries:
        return
    entries = [entry for entry in entries if entry['host'] != host]
    entries.append({'host': host, 'path': remote_path(path)})
    settings = read_json(dataset.description) if dataset.description.is_file() else dataset.settings
    save_locations(dataset, settings.get('storage_folders', []), entries)
