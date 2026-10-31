"""Compare dataset files and normalize the description of a consolidated copy."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path
import hashlib
import json

from bivosia.dataset.storage.file_operations import checked_file, file_digest
from bivosia.dataset.io.json import read_json
from bivosia.dataset.paths import file_path


def consolidated_settings(settings):
    """Use recording-relative video paths and resolve all files inside the new root."""
    result = {key: value for key, value in settings.items() if key not in ('storage_folders', 'remote_locations')}
    if result.get('video') and Path(result['video']).is_absolute():
        result['video'] = Path(result['video']).name
    return result


def file_signatures(dataset, files):
    """Hash selected files only, checking that their listing is still current."""
    result = {}
    for entry in files:
        path, state = checked_file(dataset, entry['path'], entry['expected'])
        description = Path(entry['relative']) == file_path('description')
        settings = consolidated_settings(read_json(path)) if description else None
        digest = (hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
                  if description else file_digest(path))
        checked_file(dataset, path, state)
        result[str(entry['path'])] = {'state': state, 'sha256': digest, 'settings': settings}
    return result
