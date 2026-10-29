"""Persist background job settings, progress and cancellation requests.

Examples
--------
>>> from bivosia.dataset.jobs import valid_job_id
>>> valid_job_id('abc123')
'abc123'
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path
import pprint
import re
import time
import uuid

from bivosia.dataset.jsonio import atomic_json, read_json
from bivosia.dataset.paths import file_name


def valid_job_id(value):
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', value):
        raise ValueError('Invalid job ID')
    return value


class Job:
    def __init__(self, dataset, job_id):
        self.dataset, self.id = dataset, valid_job_id(job_id)
        self.directory = dataset.folder('job', job_id=self.id)

    def path(self, name):
        return self.directory / file_name('job_' + name)

    @classmethod
    def create(cls, dataset, operation, settings, job_id=None):
        job = cls(dataset, job_id or uuid.uuid4().hex)
        job.directory.mkdir(parents=True, exist_ok=False)
        atomic_json(job.path('settings'), {'operation': operation, 'settings': settings})
        job.update(state='queued', completed=0, total=0)
        return job

    @property
    def settings(self):
        return read_json(self.path('settings'))

    @property
    def status(self):
        return read_json(self.path('status'))

    @property
    def cancelled(self):
        return self.path('cancel').exists()

    def cancel(self):
        if not self.path('settings').is_file():
            raise FileNotFoundError('Unknown job')
        self.path('cancel').touch(exist_ok=True)

    def update(self, **values):
        status = self.status if self.path('status').is_file() else {}
        status.update(values, updated=time.time())
        atomic_json(self.path('status'), status)
        return status

    def read_log(self, offset=0, limit=65536):
        path = self.path('log')
        if not path.exists():
            return '', offset
        with path.open('rb') as stream:
            stream.seek(max(0, offset))
            data = stream.read(limit)
            return data.decode('utf-8', errors='replace'), stream.tell()


def list_jobs(dataset, *, operation=None):
    root = dataset.folder('jobs')
    return sorted((p.name for p in root.iterdir() if p.is_dir() and
                   (p / file_name('job_settings')).is_file() and
                   (operation is None or Job(dataset, p.name).settings['operation'] == operation)),
                  key=lambda name: (root / name).stat().st_mtime, reverse=True) if root.exists() else []


def write_script(path, dataset, settings, code_directory):
    """Write a portable, editable launcher that calls the installed processing code."""
    script = '''"""Run VOS with BiVOSIA; edit the paths and settings below before running."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path
import argparse
import sys

CODE = {code!r}  # The bivosia package folder.
DATASET = {dataset!r}
SETTINGS = {settings}

sys.path.insert(0, str(Path(CODE).expanduser().resolve().parent))

from bivosia.dataset.layout import Dataset
from bivosia.processing.jobs import submit, run
from bivosia.processing.log import configure_logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--background', action='store_true')
    args = parser.parse_args()
    configure_logging()
    job = submit(Dataset(DATASET), 'vos', SETTINGS, background=args.background)
    if not args.background:
        raise SystemExit(run(job))
'''.format(code=str(code_directory), dataset=str(dataset), settings=pprint.pformat(settings, sort_dicts=False))
    # File dialogs ask before replacement; this writer never overwrites silently.
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(script)
