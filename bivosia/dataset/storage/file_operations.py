"""Inspect, copy and remove dataset files and folder contents safely.

Examples
--------
>>> from pathlib import Path
>>> from tempfile import TemporaryDirectory
>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.dataset.storage.file_operations import file_state, delete_file
>>> with TemporaryDirectory() as directory:
...     path = Path(directory) / 'notes.txt'
...     _ = path.write_text('notes')
...     delete_file(Dataset(directory), path, file_state(path))
...     path.exists()
False
"""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path
import errno
import hashlib
import os
import stat
import tempfile


def file_digest(path, advance=None):
    """Hash actual file contents, including the target of a video symlink."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        while chunk := source.read(1024**2):
            digest.update(chunk)
            if advance:
                advance(len(chunk))
    return digest.hexdigest()


def file_state(path):
    """Describe file contents and identity, including a symbolic link's own identity."""
    path = Path(path)
    link, info = path.lstat(), path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f'Expected a file: {path}')
    signature = [[value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]
                 for value in (link, info)]
    return {'size': info.st_size, 'modified': info.st_mtime, 'signature': signature,
            'mode': stat.S_IMODE(info.st_mode) & 0o777, 'mtime_ns': info.st_mtime_ns}


def checked_file(dataset, path, expected):
    """Require the selected physical file to still belong to this dataset and match its listing."""
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('Expected an absolute file path')
    path = path.parent.resolve() / path.name
    reference = dataset.settings.get('video')
    external = (Path(reference).expanduser().resolve()
                if reference and Path(reference).expanduser().is_absolute() else None)
    if not any(path.is_relative_to(root) for root in dataset.roots) and path != external:
        raise ValueError(f'File is outside this dataset: {path}')
    current = file_state(path)
    for key in ('size', 'modified', 'signature'):
        if key in expected and expected[key] is not None and current[key] != expected[key]:
            raise ValueError(f'File changed since it was selected; refresh the listing: {path}')
    return path, current


def delete_file(dataset, path, expected):
    """Remove only the selected file or symbolic link after checking its current identity."""
    path, _ = checked_file(dataset, path, expected)
    path.unlink()


def destination_path(root, relative, *, replace_expected=None):
    """Keep a transferred file inside the chosen dataset folder."""
    root, relative = Path(root).expanduser().resolve(), Path(relative)
    if relative.is_absolute() or not relative.parts or '..' in relative.parts:
        raise ValueError('Expected a dataset-relative file path')
    target = root / relative
    if not target.parent.resolve().is_relative_to(root):
        raise ValueError(f'Destination leaves the chosen dataset folder: {target}')
    if os.path.lexists(target) and replace_expected is None:
        raise FileExistsError(f'Destination already exists: {target}')
    return target


def copy_bytes(source, destination, size, advance=None):
    """Copy exactly the advertised bytes and return their SHA-256 checksum."""
    if not isinstance(size, int) or size < 0:
        raise ValueError('Expected a nonnegative file size')
    digest, remaining = hashlib.sha256(), size
    while remaining:
        chunk = source.read(min(1024**2, remaining))
        if not chunk:
            raise IOError('File transfer ended before all bytes arrived')
        destination.write(chunk)
        digest.update(chunk)
        remaining -= len(chunk)
        if advance:
            advance(len(chunk))
    return digest.hexdigest()


class FileDestination:
    """Stage and verify a file on its destination volume before publishing it exclusively."""

    def __init__(self, target, state, *, replace_expected=None):
        self.target, self.state = Path(target), state
        self.replace_expected = replace_expected
        self.temporary = None
        self.stream = None

    def __enter__(self):
        self._check_target()
        self.target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix='.bivosia-move-', dir=self.target.parent)
        self.temporary = Path(name)
        self.stream = os.fdopen(descriptor, 'wb')
        return self

    def publish(self, checksum):
        """Verify saved bytes and refuse a destination created during the copy."""
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        if (self.temporary.stat().st_size != self.state['size'] or
                file_digest(self.temporary) != checksum):
            raise IOError('Copied file checksum differs; source was preserved')
        os.chmod(self.temporary, self.state['mode'])
        os.utime(self.temporary, ns=(self.state['mtime_ns'], self.state['mtime_ns']))
        self._check_target()
        if self.replace_expected is None:
            os.link(self.temporary, self.target)
        else:
            os.replace(self.temporary, self.target)

    def _check_target(self):
        if self.replace_expected is None:
            if os.path.lexists(self.target):
                raise FileExistsError(f'Destination already exists: {self.target}')
        elif file_state(self.target) != self.replace_expected:
            raise ValueError(f'Destination changed after the conflict was resolved: {self.target}')

    def __exit__(self, *error):
        if self.stream is not None:
            self.stream.close()
        if self.temporary is not None:
            self.temporary.unlink(missing_ok=True)


def _folder_state(dataset, path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('Expected an absolute folder path')
    path = path.parent.resolve() / path.name
    if path in dataset.roots or not any(path.is_relative_to(root) for root in dataset.roots):
        raise ValueError(f'Folder is outside the dataset contents: {path}')
    info = path.lstat()
    if not (stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)):
        raise ValueError(f'Expected a folder: {path}')
    return {'identity': [info.st_dev, info.st_ino, info.st_mode],
            'modified': path.stat().st_mtime, 'link': path.is_symlink()}


def check_contents(dataset, files, folders):
    """Check every listed item before changing any of them; return stronger identities."""
    states = {'files': {}, 'folders': {}}
    for entry in folders:
        state = _folder_state(dataset, entry['path'])
        for key, value in entry['expected'].items():
            if value is not None and state.get(key) != value:
                raise ValueError(f'Folder changed since it was selected; refresh the listing: {entry["path"]}')
        states['folders'][entry['path']] = state
    for entry in files:
        _, state = checked_file(dataset, entry['path'], entry['expected'])
        states['files'][entry['path']] = state
    return states


def delete_contents(dataset, files, folders):
    """Remove checked files and empty folders; leave new or unselected contents intact."""
    states = check_contents(dataset, files, folders)
    for path, state in states['files'].items():
        delete_file(dataset, path, state)
    for name, state in sorted(states['folders'].items(), key=lambda item: len(Path(item[0]).parts), reverse=True):
        if _folder_state(dataset, name)['identity'] != state['identity']:
            raise ValueError(f'Folder changed during deletion: {name}')
        path = Path(name)
        if state['link']:
            path.unlink()
        else:
            try:
                path.rmdir()
            except OSError as error:
                if error.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                    raise


def create_folders(root, relatives):
    """Preserve empty subfolders while keeping destinations inside the selected root."""
    for relative in relatives:
        target = destination_path(root, relative, replace_expected={})
        if not target.resolve().is_relative_to(Path(root).resolve()):
            raise ValueError(f'Destination leaves the chosen dataset folder: {target}')
        target.mkdir(parents=True, exist_ok=True)
