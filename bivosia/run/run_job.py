"""Execute one saved processing job without a GUI.

Examples
--------
>>> from bivosia.run.run_job import main
>>> import sys
>>> sys.argv = ['run_job', '/path/to/dataset', 'job-id']  # doctest: +SKIP
>>> main()  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path
import argparse
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bivosia.dataset.jobs import Job
from bivosia.dataset.layout import Dataset
from bivosia.processing.jobs import run


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('dataset')
    parser.add_argument('job_id')
    args = parser.parse_args()
    return run(Job(Dataset(args.dataset), args.job_id), console=False)


if __name__ == '__main__':
    raise SystemExit(main())
