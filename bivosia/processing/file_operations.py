"""Copy, move and delete dataset files locally or through registered SSH installations."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import contextmanager
from pathlib import Path
import json
import subprocess
import sys
import tempfile

from bivosia.dataset.file_operations import (FileDestination, checked_file, copy_bytes,
                                            delete_file, destination_path)
from bivosia.dataset.layout import Dataset
from bivosia.dataset.locations import location_for, remote_locations, save_locations
from bivosia.dataset.organize import FileMove, apply_moves
from bivosia.dataset.paths import folder_path
from bivosia.processing.jobs import require_idle
from bivosia.processing.log import Progress
from bivosia.processing.remote import Remote


def file_locations(dataset):
    """List local and mounted dataset roots before registered SSH locations."""
    current = Dataset(dataset.directory)
    return [*({'host': '', 'path': str(root)} for root in current.roots), *remote_locations(current)]


def _remote(dataset, host, profiles, directory=None):
    if not host:
        return None
    profile = next((profile for profile in profiles if profile['host'] == host), None)
    if profile is None:
        raise ValueError(f'Register a BiVOSIA installation for {host}')
    return Remote(**{key: profile[key] for key in ('host', 'python', 'code')},
                  dataset=directory or location_for(dataset, profile))


def _response(stream):
    line = stream.readline()
    if not line:
        raise IOError('Remote file operation ended without a response')
    response = json.loads(line)
    if 'error' in response:
        raise RuntimeError(response['error'])
    return response


@contextmanager
def _connection(remote, action, **values):
    with tempfile.TemporaryFile() as errors, subprocess.Popen(
            remote.command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors) as process:
        try:
            process.stdin.write((json.dumps(remote.request(action, **values)) + '\n').encode('utf-8'))
            process.stdin.flush()
            yield process
            if process.wait():
                errors.seek(0)
                raise RuntimeError(errors.read().decode('utf-8', errors='replace') or 'SSH file operation failed')
        except BaseException:
            process.kill()
            raise


@contextmanager
def _reader(dataset, remote, path, expected):
    if remote is None:
        path, state = checked_file(dataset, path, expected)
        with path.open('rb') as stream:
            yield stream, state, lambda checksum: checked_file(dataset, path, state)
        return
    with _connection(remote, 'read_file', path=str(path), expected=expected) as process:
        process.stdin.close()
        state = _response(process.stdout)['state']

        def verify(checksum):
            if _response(process.stdout).get('sha256') != checksum:
                raise IOError('Transferred file checksum differs; source was preserved')

        yield process.stdout, state, verify


@contextmanager
def _writer(remote, root, relative, state):
    if remote is None:
        with FileDestination(destination_path(root, relative), state) as destination:
            yield destination.stream, destination.publish
        return
    with _connection(remote, 'write_file', root=str(root), relative=str(relative), state=state) as process:
        _response(process.stdout)

        def publish(checksum):
            process.stdin.write((json.dumps({'sha256': checksum}) + '\n').encode('utf-8'))
            process.stdin.flush()
            process.stdin.close()
            _response(process.stdout)

        yield process.stdin, publish


def delete_selected_file(dataset, entry, profiles):
    """Delete the selected physical copy, checking the file listing before removal."""
    require_idle(dataset)
    expected = {'size': entry.size, 'modified': entry.modified}
    remote = _remote(dataset, entry.host, profiles)
    if remote is None:
        delete_file(Dataset(dataset.directory), entry.path, expected)
    else:
        remote.call('delete_file', path=str(entry.path), expected=expected)


def transfer_on_host(dataset, path, expected, root, relative, *, move):
    """Copy or move a file on one host, verifying copied bytes before removing a source."""
    require_idle(dataset)
    source, state = checked_file(dataset, path, expected)
    target = destination_path(root, relative)
    parent = target.parent
    while not parent.exists():
        parent = parent.parent
    if move and not source.is_symlink() and source.stat().st_dev == parent.stat().st_dev:
        apply_moves((FileMove(source, target),))
        return
    with FileDestination(target, state) as destination, source.open('rb') as stream:
        progress = Progress('Moving file' if move else 'Copying file', state['size'])
        checksum = copy_bytes(stream, destination.stream, state['size'], progress.update)
        checked_file(dataset, source, state)
        destination.publish(checksum)
    if move:
        _remove_source(dataset, None, source, state, str(target))
    progress.finish()


def _remove_source(dataset, remote, path, state, destination):
    try:
        if remote is None:
            delete_file(dataset, path, state)
        else:
            remote.call('delete_file', path=str(path), expected=state)
    except Exception as error:
        raise RuntimeError(f'File was copied to {destination}, but its source could not be removed: {error}') from error


def move_selected_file(dataset, entry, location, profiles):
    """Keep the relative path when moving between local, mounted or SSH dataset roots."""
    _transfer_selected_file(dataset, entry, location, profiles, move=True)


def copy_selected_file(dataset, entry, location, profiles):
    """Copy to another dataset root while preserving the selected source and its reference."""
    _transfer_selected_file(dataset, entry, location, profiles, move=False)


def _transfer_selected_file(dataset, entry, location, profiles, *, move):
    require_idle(dataset)
    current = Dataset(dataset.directory)
    reference = current.settings.get('video')
    selected_video = (not entry.host and reference and Path(reference).expanduser().is_absolute()
                      and Path(reference).expanduser().resolve() == entry.path)
    root, host = location['path'], location.get('host', '')
    if not host and not Path(root).is_dir():
        raise FileNotFoundError(f'Destination dataset folder is unavailable: {root}')
    source = _remote(current, entry.host, profiles)
    destination = _remote(current, host, profiles, root)
    expected = {'size': entry.size, 'modified': entry.modified}
    if entry.host == host:
        if source is None:
            transfer_on_host(current, entry.path, expected, root, entry.relative, move=move)
        else:
            source.call('move_file' if move else 'copy_file', timeout=None, path=str(entry.path), expected=expected,
                        root=root, relative=str(entry.relative))
    else:
        with _reader(current, source, entry.path, expected) as (reader, state, verify):
            with _writer(destination, root, entry.relative, state) as (writer, publish):
                progress = Progress('Moving file' if move else 'Copying file', state['size'])
                checksum = copy_bytes(reader, writer, state['size'], progress.update)
                verify(checksum)
                publish(checksum)
        if move:
            target = f'{host}:' if host else ''
            _remove_source(current, source, entry.path, state, target + str(Path(root) / entry.relative))
        progress.finish()
    if entry.host or entry.path != current.description:
        current.reload()
        if not host and Path(root).resolve() not in current.roots:
            folders = [*current.settings.get('storage_folders', []), str(Path(root).resolve())]
            save_locations(current, folders, remote_locations(current))
        if move and selected_video:
            current.settings['video'] = entry.relative.relative_to(folder_path('recording')).as_posix()
            current.save()


def serve_file(request):
    """Stream a selected file or receive a verified file into its chosen dataset root."""
    dataset = Dataset(request['dataset'])
    require_idle(dataset)
    if request['action'] == 'read_file':
        path, state = checked_file(dataset, request['path'], request['expected'])
        with path.open('rb') as source:
            print(json.dumps({'state': state}), flush=True)
            checksum = copy_bytes(source, sys.stdout.buffer, state['size'])
            sys.stdout.buffer.flush()
            checked_file(dataset, path, state)
            print(json.dumps({'sha256': checksum}), flush=True)
    else:
        target = destination_path(request['root'], request['relative'])
        state = request['state']
        with FileDestination(target, state) as destination:
            print(json.dumps({'ready': True}), flush=True)
            checksum = copy_bytes(sys.stdin.buffer, destination.stream, state['size'])
            if _response(sys.stdin.buffer).get('sha256') != checksum:
                raise IOError('Transferred file checksum differs; source was preserved')
            destination.publish(checksum)
            print(json.dumps({'path': str(target)}), flush=True)
