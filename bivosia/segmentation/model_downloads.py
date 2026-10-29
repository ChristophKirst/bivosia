"""Download and verify model checkpoints without replacing existing files.

Examples
--------
>>> from hashlib import sha256
>>> from pathlib import Path
>>> from tempfile import TemporaryDirectory
>>> from bivosia.segmentation.model_downloads import verify_file
>>> with TemporaryDirectory() as folder:
...     path = Path(folder) / 'weights'
...     _ = path.write_bytes(b'example')
...     verify_file(path, 'sha256', sha256(b'example').hexdigest())
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen
import hashlib
import os
import time


def verify_file(path, algorithm, expected):
    """Check a nonempty file against a published digest or digest prefix."""
    digest = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 ** 2):
            digest.update(chunk)
    if not Path(path).stat().st_size or not digest.hexdigest().startswith(expected):
        raise ValueError(f'Checkpoint checksum differs: {path}. Move it before installing again.')


def download_weight(path, url, algorithm, checksum, report):
    """Download a public checkpoint, verify it and publish without overwriting."""
    path = Path(path)
    if path.exists():
        verify_file(path, algorithm, checksum)
        report(f'Using {path.name}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='download-', dir=path.parent) as folder:
        temporary = Path(folder) / path.name
        report(f'Downloading {path.name}')
        with urlopen(url, timeout=30) as response, temporary.open('xb') as stream:
            total = int(response.headers.get('Content-Length', 0))
            done, last = 0, time.monotonic()
            while chunk := response.read(1024 ** 2):
                stream.write(chunk)
                done += len(chunk)
                now = time.monotonic()
                if now - last >= 1:
                    suffix = f' / {total / 1024 ** 2:.0f} MiB' if total else ''
                    report(f'{path.name}: {done / 1024 ** 2:.0f} MiB{suffix}')
                    last = now
            if total and done != total:
                raise IOError(f'Incomplete download: {path.name}')
        verify_file(temporary, algorithm, checksum)
        os.link(temporary, path)

