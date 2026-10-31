"""Plan and apply file moves into a dataset's configured storage layout.

Examples
--------
>>> from bivosia.dataset.storage.organize import apply_plan, plan_collection
>>> plan = plan_collection("incoming")  # doctest: +SKIP
>>> apply_plan(plan)  # doctest: +SKIP
>>> # Optionally save the reorganization record when applying a plan:
>>> apply_plan(plan, "incoming/experiment/file_reorganization.json")  # doctest: +SKIP
"""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from dataclasses import dataclass
from pathlib import Path
import ctypes
import fnmatch
import hashlib
import json
import logging
import os
import sys

from bivosia.dataset.masks import output_files
from bivosia.dataset.paths import (companion_path, file_entries, file_name, file_suffixes,
                    folder_entries, folder_path, folder_suffixes)
from bivosia.dataset.layout import Dataset


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FileMove:
    source: Path
    destination: Path


@dataclass(frozen=True)
class LayoutPlan:
    datasets: tuple[Path, ...]
    moves: tuple[FileMove, ...]
    existing_analysis: tuple[Path, ...]


def _is_video(path):
    return path.is_file() and path.suffix.lower() in file_suffixes('video') and not path.name.startswith("._")


def _is_mask(path):
    names = {spec['name'] for key, relative, spec in file_entries()
             if relative.is_relative_to(folder_path('masks')) and '{' not in spec['name']}
    for direction in ('forward', 'reverse'):
        for suffix in folder_suffixes('masks'):
            filename = Path(file_name(direction)).with_suffix(suffix)
            names.update(item.name for item in output_files(filename, include_optional=True))
    linked = Path(file_name('linked_masks', annotation='*'))
    patterns = [item.name for suffix in folder_suffixes('masks')
                for item in output_files(linked.with_suffix(suffix), include_optional=True)]
    return path.is_file() and (path.name in names or any(
        fnmatch.fnmatchcase(path.name, pattern) for pattern in patterns))


def _mask_destination(dataset, path):
    for key, relative, spec in file_entries():
        if relative.is_relative_to(folder_path('masks')) and path.name == spec['name']:
            return dataset.file(key, write=True)
    return dataset.path(folder_path('masks') / path.name, write=True)


def existing_analysis(directory):
    """Inventory declared analysis filenames; presence does not establish validity."""
    names = set()
    for key, relative, spec in file_entries():
        if relative.parent == folder_path('analysis'):
            names.add(relative.name)
            names.update(companion_path(relative, key, name).name
                         for name in spec.get('companions', {}))
    return tuple(sorted(path for path in directory.rglob('*') if path.is_file() and path.name in names))


def _standard_entry(path):
    if path.is_symlink():
        raise ValueError(f"Inspect symbolic links before organizing: {path}")
    if path.name in {relative.name for _, relative, _, _ in folder_entries() if len(relative.parts) == 1}:
        if not path.is_dir():
            raise ValueError(f"Expected a folder: {path}")
        return True
    if path.name == file_name("description"):
        if not path.is_file():
            raise ValueError(f"Expected a dataset description file: {path}")
        return True
    return False


def plan_layout(dataset, source=None):
    """Organize one recording; retain standard folders and preserve other data."""
    source = Path(source).expanduser().resolve(strict=True) if source is not None else dataset.directory
    moves = []
    for path in sorted(source.iterdir()):
        if path.is_symlink():
            raise ValueError(f"Inspect symbolic links before organizing: {path}")
        if source == dataset.directory and _standard_entry(path):
            continue
        folder = "recording" if _is_video(path) else "masks" if _is_mask(path) else "experiment"
        target = (_mask_destination(dataset, path) if folder == 'masks'
                  else dataset.path(folder_path(folder) / path.name, write=True))
        if path != target:
            moves.append(FileMove(path, target))
    validate_moves(moves)
    products = {path for root in {source, *dataset.roots} for path in existing_analysis(root)}
    return LayoutPlan((dataset.directory,), tuple(moves), tuple(sorted(products)))


def plan_collection(directory, *, data_folders=()):
    """Organize loose videos or existing dataset subfolders without guessing associations.

    Each loose video creates a dataset named after its stem. Standard masks in
    a multi-video folder are ambiguous and require assignment first. Collection
    subfolders with a single video, standard masks, or dataset metadata retain
    their names. Other data goes into the collection's experiment folder.
    Use data_folders to identify multi-video exports that are not datasets.
    """
    root = Path(directory).expanduser().resolve(strict=True)
    entries = sorted(root.iterdir())
    standard = {path for path in entries if _standard_entry(path)}
    data_folders = set(data_folders)
    if any(not (root / name).is_dir() and not (root / folder_path("experiment") / name).is_dir()
           or Path(name).name != name for name in data_folders):
        raise ValueError("Each data folder must name an immediate subfolder of the collection")
    videos = [path for path in entries if _is_video(path)]
    datasets, moves, products = [], [], []
    if videos:
        if len({path.stem for path in videos}) != len(videos):
            raise ValueError("Video stems must be unique")
        masks = [path for path in entries if _is_mask(path)]
        if masks and len(videos) > 1:
            raise ValueError("Masks in a multi-video folder must first be assigned to individual recordings")
        for video in videos:
            dataset = Dataset(root / video.stem)
            if dataset.directory.exists():
                raise FileExistsError(f"Dataset destination already exists: {dataset.directory}")
            datasets.append(dataset.directory)
            moves.append(FileMove(video, dataset.path(Path(folder_path("recording")) / video.name, write=True)))
        for path in entries:
            if path in videos or path in standard:
                continue
            if _is_mask(path):
                target = _mask_destination(Dataset(datasets[0]), path)
            else:
                target = root / folder_path("experiment") / path.name
            moves.append(FileMove(path, target))
        products.extend(existing_analysis(root))
    else:
        for path in entries:
            if path in standard:
                products.extend(existing_analysis(path))
                continue
            is_dataset = False
            if path.is_dir() and path.name not in data_folders:
                children = list(path.iterdir())
                direct_videos = [child for child in children if _is_video(child)]
                has_masks = any(_is_mask(child) for child in children) or (path / folder_path("masks")).is_dir()
                is_dataset = (len(direct_videos) == 1 or has_masks or
                              (path / file_name("description")).is_file() or (path / folder_path("recording")).is_dir())
                if len(direct_videos) > 1 and not is_dataset:
                    raise ValueError(f"Multiple videos in {path.name}: identify it as a dataset or a data folder")
            if is_dataset:
                plan = plan_layout(Dataset(path))
                datasets.extend(plan.datasets)
                moves.extend(plan.moves)
                products.extend(plan.existing_analysis)
            else:
                moves.append(FileMove(path, root / folder_path("experiment") / path.name))
                if path.is_dir():
                    products.extend(existing_analysis(path))
    if not datasets:
        raise ValueError("No videos or dataset folders found")
    validate_moves(moves)
    return LayoutPlan(tuple(datasets), tuple(moves), tuple(products))


def validate_moves(moves):
    """Check the whole plan before creating directories or moving files."""
    sources, destinations = set(), set()
    for move in moves:
        source, target = move.source, move.destination
        if source.is_symlink() or not (source.is_file() or source.is_dir()):
            raise ValueError(f"Expected a regular file or directory: {source}")
        if source in sources or target in destinations:
            raise ValueError(f"Repeated source or destination: {source} -> {target}")
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        if target.is_relative_to(source):
            raise ValueError(f"Cannot move a folder into itself: {source}")
        parent = target.parent
        while not parent.exists():
            if parent.is_symlink():
                raise ValueError(f"Unexpected symbolic link: {parent}")
            parent = parent.parent
        if not parent.is_dir() or any(path.is_symlink() for path in target.parents):
            raise ValueError(f"Unsafe destination directory: {target.parent}")
        if source.stat().st_dev != parent.stat().st_dev:
            raise ValueError(f"Cross-filesystem relocation requires an explicit copy: {source} -> {target}")
        sources.add(source)
        destinations.add(target)
    if sources & destinations:
        raise ValueError("Move destinations must not overlap source files")
    if any(parent in sources for source in sources for parent in source.parents):
        raise ValueError("Move sources must not overlap")


def _rename_no_replace(source, target):
    """Rename atomically without replacing a destination created concurrently."""
    if os.name == "nt":
        os.rename(source, target)
        return
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "linux":
        rename = library.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        args = (-100, os.fsencode(source), -100, os.fsencode(target), 1)
    elif sys.platform == "darwin":
        rename = library.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        args = (os.fsencode(source), os.fsencode(target), 4)
    else:
        raise OSError("No-overwrite rename is not implemented on this platform")
    rename.restype = ctypes.c_int
    if rename(*args):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(target))


def apply_moves(moves, *, before=None, after=None):
    """Apply a validated same-filesystem plan with optional per-file callbacks."""
    moves = tuple(moves)
    validate_moves(moves)
    for index, move in enumerate(moves, 1):
        if before is not None:
            before(move)
        move.destination.parent.mkdir(parents=True, exist_ok=True)
        _rename_no_replace(move.source, move.destination)
        if after is not None:
            after(move)
        if index % 100 == 0 or index == len(moves):
            logger.info("Moved %s/%s entries", index, len(moves))


def _inventory(path):
    """Record file counts, sizes and a digest of relative names and timestamps."""
    files, size = 0, 0
    digest = hashlib.sha256()
    for item in [path, *sorted(path.rglob("*"))] if path.is_dir() else [path]:
        if item.is_symlink() or not (item.is_file() or item.is_dir()):
            raise ValueError(f"Expected a regular file or directory: {item}")
        is_file = item.is_file()
        stat = item.stat()
        record = (item.relative_to(path).as_posix(), is_file,
                  stat.st_size if is_file else 0, stat.st_mtime_ns if is_file else 0)
        digest.update(json.dumps(record).encode("utf-8"))
        files += is_file
        size += stat.st_size if is_file else 0
    return {"files": files, "bytes": size, "metadata_sha256": digest.hexdigest()}


def apply_plan(plan, reorganization_file=None):
    """Move and verify each entry, optionally saving a reorganization record.

    No record is written unless reorganization_file is supplied. When requested,
    it stores paths relative to its folder and pre-move metadata.
    If interrupted, source/destination existence identifies completed moves.
    Verification checks names, sizes and timestamps, not file-content hashes.
    """
    if not plan.moves:
        return
    validate_moves(plan.moves)
    if reorganization_file is not None:
        reorganization_file = Path(reorganization_file).expanduser().absolute()
        if reorganization_file.exists() or reorganization_file.is_symlink():
            raise FileExistsError(reorganization_file)
        if any(reorganization_file.is_relative_to(move.source) or reorganization_file.is_relative_to(move.destination)
               for move in plan.moves):
            raise ValueError("The file reorganization record must be outside all moved entries")
    logger.info("Inventorying %s entries before moving", len(plan.moves))
    inventories = {move: _inventory(move.source) for move in plan.moves}
    if reorganization_file is not None:
        records = [{"source": os.path.relpath(move.source, reorganization_file.parent),
                    "destination": os.path.relpath(move.destination, reorganization_file.parent),
                    **inventories[move]} for move in plan.moves]
        reorganization_file.parent.mkdir(parents=True, exist_ok=True)
        with reorganization_file.open("x", encoding="utf-8") as stream:
            json.dump({"moves": records}, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        logger.info("Saved file reorganization record: %s", reorganization_file)

    def verify(move, path):
        if _inventory(path) != inventories[move]:
            raise RuntimeError(f"File inventory changed; stopped organizing: {path}")

    apply_moves(plan.moves, before=lambda move: verify(move, move.source),
                after=lambda move: verify(move, move.destination))
    logger.info("Verified %s files (%s bytes)", sum(item["files"] for item in inventories.values()),
                sum(item["bytes"] for item in inventories.values()))
