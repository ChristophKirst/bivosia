"""Dataset locations and the hand annotation selected for analysis.

Examples
--------
>>> from bivosia.dataset.layout import Dataset
>>> dataset = Dataset("experiment")
>>> dataset.mask("forward").name
'masks_forward.h5'
>>> dataset.new_annotation(0.6, 0.9).name
'seed_0.60_recover_0.90.json'
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from copy import deepcopy
from pathlib import Path

from bivosia.dataset.jsonio import atomic_json, read_json
from bivosia.dataset.paths import (companion_path, file_name, file_path, file_suffixes, folder_path,
                    folder_suffixes, storage_for)


def datasets_in(directory):
    """List immediate dataset folders, using their description or default mask folder."""
    directory = Path(directory).expanduser().resolve()
    datasets = []
    for path in sorted(directory.iterdir()):
        if path.is_dir():
            dataset = Dataset(path)
            if dataset.description.is_file() or (path / folder_path("masks")).is_dir():
                datasets.append(dataset)
    if not datasets:
        raise ValueError(f"No dataset folders found in {directory}; expected {file_name('description')} or {folder_path('masks')}")
    return datasets


class Dataset:
    def __init__(self, directory, storage_folder=None, video=None):
        self.directory = Path(directory).expanduser().resolve()
        self.description = self.directory / file_path("description")
        self.reload()
        if storage_folder is not None:
            self.settings["storage_folders"] = [str(storage_folder), *self.settings.get("storage_folders", [])[1:]]
        if video is not None:
            self.settings["video"] = self.video_name(video)

    def reload(self):
        """Read current settings after another window updates the description."""
        self.settings = read_json(self.description) if self.description.is_file() else {}
        self._saved_settings = deepcopy(self.settings)

    @property
    def roots(self):
        """Dataset and additional storage folders, without duplicate locations."""
        folders = self.settings.get("storage_folders", [])
        if not isinstance(folders, list):
            raise ValueError("storage_folders must be a list of paths")
        return tuple(dict.fromkeys([self.directory, *[
            (self.directory / Path(folder).expanduser()).resolve() for folder in folders]]))

    def output_path(self, path):
        """Choose the configured write location for a dataset-relative path."""
        relative = Path(self.relative(path))
        storage = storage_for(relative)
        if storage not in ("dataset", "external"):
            raise ValueError(f"Unknown write storage: {storage}")
        root = self.roots[1] if storage == "external" and len(self.roots) > 1 else self.directory
        return root / relative

    def path(self, path, *, write=False):
        """Search all roots for a file; writes follow the configured folder map."""
        relative = self.relative(path)
        found = list(dict.fromkeys((root / relative).resolve() for root in self.roots
                                   if (root / relative).is_file()))
        if len(found) > 1:
            raise ValueError(f"File exists in multiple storage folders: {relative}")
        destination = self.output_path(relative)
        if write:
            if found and found[0] != destination.resolve():
                raise ValueError(f"File already exists outside its write location: {relative}; move it before rewriting")
            root = next(root for root in sorted(self.roots, key=lambda item: len(item.parts), reverse=True)
                        if destination.is_relative_to(root))
            if root != self.directory and not root.is_dir():
                raise FileNotFoundError(f"Storage folder is unavailable: {root}")
        return destination if write or not found else found[0]

    def relative(self, path):
        """Return a logical path relative to its registered storage root."""
        path = Path(path).expanduser()
        if not path.is_absolute():
            if ".." in path.parts or not path.parts:
                raise ValueError("Expected a dataset-relative file path")
            return path.as_posix()
        path = path.resolve()
        for root in sorted(self.roots, key=lambda item: len(item.parts), reverse=True):
            if path.is_relative_to(root):
                return path.relative_to(root).as_posix()
        raise ValueError(f"Path is outside the dataset and its storage folders: {path}")

    def glob(self, pattern):
        """List logical files across all roots, rejecting duplicate filenames."""
        names = {path.relative_to(root).as_posix() for root in self.roots
                 for path in root.glob(pattern) if path.is_file()}
        return [self.path(name) for name in sorted(names)]

    def folder(self, key, **parameters):
        """Return a configured folder at its selected write location."""
        return self.output_path(folder_path(key, **parameters))

    def file(self, key, *, write=False, companion=None, **parameters):
        """Resolve a declared file or its companion across registered storage roots."""
        relative = file_path(key, **parameters)
        if companion is not None:
            relative = companion_path(relative, key, companion)
        return self.path(relative, write=write)

    @property
    def recording(self):
        return self.folder("recording")

    @property
    def masks(self):
        return self.folder("masks")

    @property
    def keypoints(self):
        return self.folder("keypoints")

    def keypoint_file(self, *, write=False):
        name = self.settings.get('keypoints', {}).get('file', file_name('keypoints'))
        relative = file_path('keypoints').parent / name
        relative.relative_to(folder_path('keypoints'))
        return self.path(relative, write=write)

    @property
    def analysis(self):
        return self.folder("analysis")

    @property
    def annotations(self):
        return self.folder("annotations")

    @property
    def experiment(self):
        return self.folder("experiment")

    @property
    def cache(self):
        return self.folder("cache")

    def video_name(self, value):
        """Store a recording-relative name or an absolute reference to an external video."""
        path = Path(value).expanduser()
        if path.is_absolute():
            path = path.resolve()
            for root in self.roots:
                recording = root / folder_path('recording')
                if path.is_relative_to(recording):
                    return path.relative_to(recording).as_posix()
            return str(path)
        relative = Path(folder_path("recording")) / path
        if ".." in relative.parts:
            raise ValueError("Video must be inside the recording folder")
        return relative.relative_to(folder_path("recording")).as_posix()

    @property
    def video(self):
        name = self.settings.get("video")
        if name:
            if Path(name).expanduser().is_absolute():
                return Path(name).expanduser().resolve()
            return self.path(Path(folder_path("recording")) / self.video_name(name))
        paths = [path for path in self.glob(f"{folder_path('recording')}/*")
                 if path.suffix.lower() in file_suffixes('video') and not path.name.startswith("._")]
        if len(paths) != 1:
            raise ValueError(f"Set video in {self.description.name}: the recording folder must otherwise contain exactly one video")
        return paths[0]

    def mask(self, direction, *, write=False):
        """Resolve a selected mask file or a unique standard HDF5/NumPy filename."""
        selected = self.settings.get('masks', {}).get(direction)
        base = file_path(direction)
        if selected is not None:
            base = Path(folder_path('masks')) / selected
            base.relative_to(folder_path('masks'))
            if base.suffix.lower() not in folder_suffixes('masks'):
                raise ValueError(f'Unsupported mask format: {base.suffix}')
        else:
            found = [candidate for suffix in folder_suffixes('masks')
                     if (candidate := self.path(base.with_suffix(suffix))).is_file()]
            if len(found) > 1:
                raise ValueError(f'Several {direction} masks exist; select a filename in dataset.json under masks')
            if found:
                base = found[0]
        return self.path(base, write=write)

    def analysis_file(self, filename, *, write=False):
        return self.path(Path(folder_path("analysis")) / filename, write=write)

    def array(self, name, *, write=False):
        return self.file(name, write=write)

    def annotation_path(self, value, *, write=False):
        relative = Path(self.relative(value))
        relative.relative_to(folder_path("annotations"))
        return self.path(relative, write=write)

    def new_annotation(self, seed, recover, prefix=""):
        return self.file("annotation", seed=seed, recover=recover, prefix=prefix, write=True)

    @property
    def annotation_files(self):
        """Zone annotation files in the configured annotation folder."""
        return self.glob(f"{folder_path('annotations')}/*.json")

    @property
    def annotation(self):
        selected = self.settings.get("annotation")
        if selected:
            path = self.annotation_path(selected)
            if not path.is_file():
                raise FileNotFoundError(f"Selected annotation does not exist: {path}")
            return path
        paths = self.annotation_files
        if len(paths) > 1:
            raise ValueError("Several annotations exist. Choose one with Open or specify annotation in the dataset description.")
        return paths[0] if paths else None

    def select_annotation(self, path):
        self.settings["annotation"] = self.relative(self.annotation_path(path))
        self.save()

    def linked_masks(self, annotation):
        return self.file("linked_masks", annotation=Path(annotation).stem, write=True)

    def export_name(self, path):
        return Path(self.relative(path)).relative_to(folder_path("masks")).as_posix()

    def export_path(self, name):
        relative = Path(folder_path("masks")) / name
        relative.relative_to(folder_path("masks"))
        return self.path(relative)

    def save(self):
        settings = dict(self.settings)
        # Location dialogs may save while an editor still holds this dataset object.
        current = read_json(self.description) if self.description.is_file() else {}
        for key in ('storage_folders', 'remote_locations'):
            if settings.get(key) == self._saved_settings.get(key):
                if key in current:
                    settings[key] = current[key]
                else:
                    settings.pop(key, None)
        folders = settings.get("storage_folders", [])
        settings["storage_folders"] = [
            (Path(folder).relative_to(self.directory).as_posix()
             if Path(folder).is_absolute() and Path(folder).is_relative_to(self.directory) else str(folder))
            for folder in folders if (self.directory / Path(folder).expanduser()).resolve() != self.directory]
        if not settings["storage_folders"]:
            settings.pop("storage_folders")
        selected = settings.get("annotation")
        if selected and self.annotation_files == [self.path(selected)]:
            settings.pop("annotation")
        atomic_json(self.description, settings)
        self.settings = settings
        self._saved_settings = deepcopy(settings)
