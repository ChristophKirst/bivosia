"""Resolve registered data locations for viewers without copying mask files."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import logging

from bivosia.dataset.storage.remote_reader import RemoteDatasetReader
from bivosia.dataset.workflow.status import inspect_locations
from bivosia.ui.remote_settings import remote_settings


def location_preferences(dataset):
    store = remote_settings()
    profiles = store.installations
    selected = store.dataset(dataset.directory).get('installation')
    return tuple(profiles.values()), profiles.get(selected, {}).get('host')


def open_remote_reader(dataset, *, optional=False):
    profiles, preferred = location_preferences(dataset)
    info = inspect_locations(dataset, profiles, preferred, allow_partial=optional)
    remote = info['analysis_remote'] or info['mask_remote']
    if remote is not None:
        return RemoteDatasetReader(remote)
    if info['remote_errors'] and not all(item['exists'] for item in info['masks'].values()):
        message = '\n'.join(info['remote_errors'])
        if not optional:
            raise ValueError(message)
        logging.getLogger(__name__).warning('Remote data unavailable; showing local files: %s', message)
    return None
