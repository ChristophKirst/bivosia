"""Read small UTF-8 text files for inspection.

Examples
--------
>>> from pathlib import Path
>>> from tempfile import TemporaryDirectory
>>> from bivosia.dataset.text import read_text
>>> with TemporaryDirectory() as directory:
...     path = Path(directory) / 'notes.txt'
...     _ = path.write_text('Recording notes', encoding='utf-8')
...     read_text(path)
'Recording notes'
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path

from bivosia.config import DATASET_VIEWER_DEFAULTS


def read_text(path, *, max_bytes=None):
    """Read bounded text, accepting a UTF-8 byte-order mark without altering the file."""
    limit = DATASET_VIEWER_DEFAULTS['max_text_bytes'] if max_bytes is None else max_bytes
    if limit < 1:
        raise ValueError('Text preview size must be positive')
    path = Path(path)
    message = f'{path.name} exceeds the text preview limit ({limit:,} bytes)'
    if path.stat().st_size > limit:
        raise ValueError(message)
    with path.open('rb') as stream:
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise ValueError(message)
    return content.decode('utf-8-sig')
