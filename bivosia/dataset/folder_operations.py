"""Validate and remove listed folder contents without following directory links."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path
import errno
import stat

from bivosia.dataset.file_operations import checked_file, delete_file, destination_path


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
