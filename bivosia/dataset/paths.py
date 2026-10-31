"""Resolve dataset folders, files and companions from the shared layout.

Examples
--------
>>> from bivosia.dataset.paths import companion_path, file_path, folder_path, format_paths
>>> file_path('key_frames').as_posix()
'masks/key_frame/key_frames.json'
>>> folder_path('job', job_id='abc').as_posix()
'analysis/jobs/abc'
>>> companion_path('custom.npy', 'mask_features', 'metadata').name
'custom.json'
>>> format_paths('masks', 'labels.npy', 'companions')['validity'].name
'labels.valid.npy'
>>> format_paths('masks', 'labels.npz', 'scratch')['buffer'].name
'labels.npz.partial.npy'
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path

from bivosia.config import DATASET_LAYOUT, FILE_PATTERNS
from bivosia.dataset.io.registry import format_spec, format_suffixes


def folder_entries(node=None, parent=Path(), storage=None):
    """Yield folder identifiers, relative paths, specifications and inherited storage."""
    node = DATASET_LAYOUT if node is None else node
    storage = node.get('storage', storage)
    for key, spec in node.get('folders', {}).items():
        path = parent / spec['name']
        selected = spec.get('storage', storage)
        yield key, path, spec, selected
        yield from folder_entries(spec, path, selected)


def file_entries():
    """Yield file identifiers, relative path templates and their specifications."""
    nodes = [(Path(), DATASET_LAYOUT)]
    nodes.extend((path, spec) for _, path, spec, _ in folder_entries())
    for parent, node in nodes:
        for key, spec in node.get('files', {}).items():
            yield key, parent / spec['name'], spec


def _entry(entries, key):
    for entry in entries:
        if entry[0] == key:
            return entry
    raise KeyError(key)


def folder_path(key, **parameters):
    """Return a configured dataset-relative folder, expanding its parameters."""
    return Path(str(_entry(folder_entries(), key)[1]).format(**parameters))


def file_path(key, **parameters):
    """Return a configured dataset-relative file, expanding its parameters."""
    return Path(str(_entry(file_entries(), key)[1]).format(**parameters))


def file_name(key, **parameters):
    """Return a configured filename without requiring parent-folder parameters."""
    return _entry(file_entries(), key)[2]['name'].format(**parameters)


def folder_suffixes(key):
    return format_suffixes(_entry(folder_entries(), key)[2]['data'])


def file_suffixes(key):
    return format_suffixes(_entry(file_entries(), key)[2]['data'])


def storage_for(relative):
    """Select storage from the deepest containing folder, including parameterized folders."""
    relative = Path(relative)
    matches = [(len(path.parts), storage) for _, path, _, storage in folder_entries()
               if len(relative.parts) >= len(path.parts) and all(
                   template == actual or template.startswith('{') and template.endswith('}')
                   for template, actual in zip(path.parts, relative.parts))]
    return max(matches)[1] if matches else DATASET_LAYOUT['storage']


def companion_path(path, key, companion):
    """Locate a product's companion beside its selected output filename."""
    path = Path(path)
    pattern = _entry(file_entries(), key)[2]['companions'][companion]
    return path.with_name(pattern.format(name=path.name, stem=path.stem))


def auxiliary_path(path, kind):
    """Locate partial or lock files using shared naming patterns."""
    path = Path(path)
    return path.with_name(FILE_PATTERNS[kind].format(name=path.name, stem=path.stem))


def format_paths(category, path, group, *, format=None):
    """Locate a format's named companion or scratch files beside its data file."""
    path = Path(path)
    patterns = format_spec(category, path, format=format).get(group, {})
    return {key: path.with_name(pattern.format(name=path.name, stem=path.stem))
            for key, pattern in patterns.items()}
