"""Atomic JSON writing and annotation filenames."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path
import json
import os
import tempfile

from bivosia.dataset.paths import file_name
from bivosia.dataset.io.registry import format_handler, format_spec


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def annotation_filename(seed, recover, prefix=""):
    return file_name("annotation", seed=seed, recover=recover, prefix=prefix)


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    patterns = format_spec('documents', format='json')['temporary']
    descriptor, temporary = tempfile.mkstemp(
        prefix=patterns['prefix'].format(name=path.name, stem=path.stem),
        suffix=patterns['suffix'].format(name=path.name, stem=path.stem), dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path):
    """Read a structured document using the registered JSON reader."""
    return format_handler('documents', path, format='json')(path)


def atomic_json(path, payload):
    """Write JSON atomically, without exposing a partially written document."""
    return format_handler('documents', path, format='json', operation='writer')(path, payload)
