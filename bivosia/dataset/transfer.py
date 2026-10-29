"""Snapshot VOS inputs and publish verified uploads without replacing existing files."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import tempfile

from bivosia.dataset.layout import Dataset
from bivosia.dataset.paths import file_path, folder_path


def file_digest(path, advance=None):
    """Hash actual file contents, including the target of a video symlink."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        while chunk := source.read(1024**2):
            digest.update(chunk)
            if advance:
                advance(len(chunk))
    return digest.hexdigest()


@contextmanager
def vos_inputs(dataset):
    """Snapshot metadata and key frames while retaining the original video stream."""
    current = Dataset(dataset.directory)
    video = current.video
    if not video.is_file():
        raise FileNotFoundError(f'Video is unavailable: {video}')
    with tempfile.TemporaryDirectory(prefix='bivosia-inputs-') as directory:
        root = Path(directory)
        sources = {str(file_path('video', filename=video.name)): video}
        for key in ('key_frames', 'key_frame_masks'):
            path = current.file(key)
            if not path.is_file():
                raise FileNotFoundError(f'Save key-frame annotations before running VOS: {path}')
            relative = file_path(key)
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            sources[str(relative)] = destination
        settings = {key: value for key, value in current.settings.items()
                    if key not in ('storage_folders', 'remote_locations')}
        settings['video'] = video.name
        description = root / file_path('description')
        description.write_text(json.dumps(settings, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        sources[str(file_path('description'))] = description
        yield sources


class InputUpload:
    """Stage a complete input set and refuse different or concurrently changed files."""

    def __init__(self, dataset, manifest):
        self.dataset, self.manifest = dataset, manifest
        self.targets, self.missing, self.staged = {}, [], {}

    def plan(self):
        required = {str(file_path(key)) for key in ('description', 'key_frames', 'key_frame_masks')}
        videos = set(self.manifest) - required
        if len(videos) != 1 or not required.issubset(self.manifest):
            raise ValueError('Upload must contain a description, video and both key-frame files')
        video = Path(next(iter(videos)))
        if video.parent != folder_path('recording') or video.name in ('', '.', '..'):
            raise ValueError('Upload video must be directly inside the recording folder')
        conflicts = []
        for relative, metadata in self.manifest.items():
            path = Path(relative)
            if path.is_absolute() or '..' in path.parts or str(path) != relative:
                raise ValueError('Upload paths must be dataset-relative')
            if (not isinstance(metadata.get('size'), int) or metadata['size'] < 0 or
                    not re.fullmatch(r'[a-f0-9]{64}', metadata.get('sha256', ''))):
                raise ValueError('Invalid upload size or checksum')
            target = self.dataset.path(relative)
            self.targets[relative] = target
            if target.is_file() and self.matches(relative, target):
                continue
            if os.path.lexists(target):
                conflicts.append(relative)
            else:
                self.missing.append(relative)
        if conflicts:
            raise FileExistsError('Remote files differ; nothing was uploaded or overwritten: ' +
                                  ', '.join(conflicts) + '. Choose another dataset folder or resolve the differences.')
        return self.missing

    def matches(self, relative, path):
        metadata = self.manifest[relative]
        return path.stat().st_size == metadata['size'] and file_digest(path) == metadata['sha256']

    def receive(self, source):
        """Read exactly the planned bytes into temporary files on their target volumes."""
        for relative in self.missing:
            target = self.targets[relative]
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(prefix='.bivosia-upload-', dir=target.parent)
            temporary = Path(name)
            self.staged[relative] = temporary
            remaining = self.manifest[relative]['size']
            digest = hashlib.sha256()
            with os.fdopen(descriptor, 'wb') as stream:
                while remaining:
                    chunk = source.read(min(1024**2, remaining))
                    if not chunk:
                        raise IOError('Upload ended before all inputs arrived')
                    stream.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            if digest.hexdigest() != self.manifest[relative]['sha256']:
                raise IOError(f'Upload checksum differs: {relative}; no inputs were published')

    def publish(self):
        """Recheck reused inputs and use exclusive links to prevent overwrite races."""
        for relative, target in self.targets.items():
            if relative not in self.staged and (not target.is_file() or not self.matches(relative, target)):
                raise FileExistsError(f'Remote input changed during upload: {relative}')
        published = []
        try:
            for relative, temporary in self.staged.items():
                target = self.targets[relative]
                os.link(temporary, target)
                published.append((temporary, target))
        except BaseException:
            for temporary, target in reversed(published):
                if target.exists() and os.path.samefile(temporary, target):
                    target.unlink()
            raise

    def close(self):
        for path in self.staged.values():
            path.unlink(missing_ok=True)
