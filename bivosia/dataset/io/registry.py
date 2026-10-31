"""Resolve format adapters within a data category.

Examples
--------
>>> from bivosia.dataset.io.registry import format_spec, format_suffixes
>>> format_suffixes('arrays')
('.npy',)
>>> format_spec('keypoints', format='sleap')['reader'][1]
'SleapKeypointReader'
>>> format_spec('masks', 'labels.npy')['reader'][1]
'NpyMaskReader'
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from importlib import import_module
from pathlib import Path

from bivosia.config import DATA_FORMATS


def format_spec(category, path=None, *, format=None):
    """Select an explicit format, or infer a unique format from its suffix."""
    formats = DATA_FORMATS[category]
    if format is not None:
        if format not in formats:
            raise ValueError(f'Unknown {category} format: {format}')
        return formats[format]
    if path is None:
        raise ValueError('Supply a path or an explicit format')
    suffix = Path(path).suffix.lower()
    matches = [spec for spec in formats.values() if suffix in spec['suffixes']]
    if len(matches) != 1:
        reason = 'Ambiguous' if matches else 'Unsupported'
        raise ValueError(f'{reason} {category} format: {suffix}; select a registered format')
    return matches[0]


def format_handler(category, path=None, *, format=None, operation='reader'):
    """Import a registered reader or writer only when it is requested."""
    spec = format_spec(category, path, format=format)
    if operation not in spec:
        raise ValueError(f'{category} format {format or Path(path).suffix} has no {operation}')
    module, name = spec[operation]
    return getattr(import_module(module), name)


def format_suffixes(category):
    """Return the supported suffixes in registry order without duplicates."""
    return tuple(dict.fromkeys(suffix for spec in DATA_FORMATS[category].values()
                               for suffix in spec['suffixes']))


def file_inspector(path):
    """Find a shared header inspector, independent of a file's data category.

    Inspectors return FileMetadata. Return None for formats without an inspector.
    Categories using the same suffix must agree on its physical-format inspector.
    """
    suffix = Path(path).suffix.lower()
    handlers = {spec['inspector'] for formats in DATA_FORMATS.values()
                for spec in formats.values()
                if suffix in spec['suffixes'] and 'inspector' in spec}
    if not handlers:
        return None
    if len(handlers) != 1:
        raise ValueError(f'Ambiguous file inspector for {suffix}')
    module, name = handlers.pop()
    return getattr(import_module(module), name)
