"""Inventory existing dataset files across local and external storage.

Examples
--------
>>> from pathlib import Path
>>> from tempfile import TemporaryDirectory
>>> from bivosia.dataset.storage.inventory import scan_dataset
>>> from bivosia.dataset.layout import Dataset
>>> with TemporaryDirectory() as directory:
...     path = Path(directory) / 'notes.txt'
...     _ = path.write_text('recording notes')
...     files = [entry for entry in scan_dataset(Dataset(directory))
...              if not entry.is_directory]
...     [(entry.relative.as_posix(), entry.size) for entry in files]
[('notes.txt', 15)]
"""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from dataclasses import dataclass
from pathlib import Path
import os
import stat

from bivosia.dataset.layout import Dataset
from bivosia.dataset.paths import file_path
from bivosia.dataset.storage.locations import remote_locations
from bivosia.processing.remote.client import Remote


@dataclass(frozen=True)
class FileInfo:
    """One physical entry and its logical path within the dataset."""

    relative: Path
    path: Path
    is_directory: bool
    size: int | None = None
    modified: float | None = None
    error: str = ''
    source: Path | str | None = None
    host: str = ''


def scan_dataset(dataset):
    """Read file statistics without opening file contents or following directory links.

    Keep duplicate logical filenames from different roots as separate entries.
    Unavailable paths carry their error instead of hiding the remaining files.
    Nested storage roots are scanned once, at their own logical location.
    """
    entries = {}
    roots = dataset.roots

    def record(path, root, *, directory=False, error=None):
        relative = path.relative_to(root)
        try:
            if error is not None:
                raise error
            info = path.stat()
            directory = stat.S_ISDIR(info.st_mode)
            if not directory and not stat.S_ISREG(info.st_mode):
                raise OSError(f'Not a regular file: {path}')
            entries[path] = FileInfo(relative, path, directory,
                                     None if directory else info.st_size, info.st_mtime, source=root)
        except OSError as failure:
            entries[path] = FileInfo(relative, path, directory, error=str(failure), source=root)

    for root in roots:
        record(root, root, directory=True)
        if entries[root].error:
            continue
        if not entries[root].is_directory:
            entries[root] = FileInfo(Path(), root, True, error=f'Not a folder: {root}')
            continue

        def on_error(error):
            record(Path(error.filename), root, directory=True, error=error)

        for parent, directories, files in os.walk(root, onerror=on_error, followlinks=False):
            parent = Path(parent)
            directories[:] = sorted(name for name in directories
                                    if parent / name not in roots)
            for name in directories:
                record(parent / name, root, directory=True)
            for name in sorted(files):
                record(parent / name, root)
    reference = dataset.settings.get('video')
    if reference and Path(reference).expanduser().is_absolute():
        path = Path(reference).expanduser().resolve()
        if path not in entries:
            relative = file_path('video', filename=path.name)
            try:
                info = path.stat()
                if not stat.S_ISREG(info.st_mode):
                    raise OSError(f'Not a regular file: {path}')
                entries[path] = FileInfo(relative, path, False, info.st_size, info.st_mtime,
                                         source=path.parent)
            except OSError as error:
                entries[path] = FileInfo(relative, path, False, error=str(error), source=path.parent)
    return tuple(entries.values())


def inventory_records(dataset):
    """Serialize file statistics for an SSH response without reading file contents."""
    from dataclasses import asdict
    return [{key: str(value) if isinstance(value, Path) else value
             for key, value in asdict(entry).items()} for entry in scan_dataset(dataset)]


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
