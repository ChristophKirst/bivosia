"""Collect a dataset's file inventory from mounted folders and registered SSH hosts."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path

from bivosia.dataset.inventory import FileInfo, scan_dataset
from bivosia.dataset.layout import Dataset
from bivosia.dataset.locations import remote_locations
from bivosia.processing.remote import Remote


def scan_locations(dataset, profiles):
    """Keep inaccessible remote locations visible alongside the available files."""
    dataset = Dataset(dataset.directory)
    entries = list(scan_dataset(dataset))
    for location in remote_locations(dataset):
        host, path = location['host'], location['path']
        try:
            profile = next((p for p in profiles if p['host'] == host), None)
            if profile is None:
                raise ValueError(f'Register a BiVOSIA installation for {host}')
            remote = Remote(**{key: profile[key] for key in ('host', 'python', 'code')}, dataset=path)
            response = remote.call('inventory')
            for record in response['files']:
                record['relative'], record['path'] = Path(record['relative']), Path(record['path'])
                record['source'] = f"{host}:{record['source']}"
                record['host'] = host
                entries.append(FileInfo(**record))
        except Exception as error:
            entries.append(FileInfo(Path(), Path(path), True, error=str(error),
                                     source=f'{host}:{path}', host=host))
    return tuple(entries)
