"""Plan input synchronization using the dataset viewer's file-transfer machinery."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path

from bivosia.config import CONSENSUS_MASK_SOURCES
from bivosia.processing.workflow import required_products, step_inputs
from bivosia.dataset.jsonio import read_json
from bivosia.dataset.layout import Dataset
from bivosia.dataset.paths import file_path, folder_path, folder_suffixes, format_paths
from bivosia.processing.consolidation import plan_consolidation
from bivosia.processing.file_operations import _remote
from bivosia.processing.remote_inventory import inspect_locations, scan_locations


def plan_input_sync(dataset, location, profiles, operation, *, annotation=None, settings=None, progress=None):
    """Include only the inputs needed by the requested calculation, retaining target settings."""
    if annotation:
        dataset = Dataset(dataset.directory)
        dataset.settings['annotation'] = annotation
    entries = scan_locations(dataset, profiles)
    files = {entry.relative.as_posix() for entry in entries if not entry.is_directory and not entry.error}
    products = (required_products(settings['steps']) if operation == 'workflow' else
                step_inputs(operation, settings))
    if 'consensus_masks' in products:
        info = inspect_locations(dataset, profiles, location.get('host') or None, include_workflow=True)
        exported = info['export']
        source = ('export' if exported and exported['exists'] and not exported['error']
                  and info['export_current'] and not info['export_error'] else 'annotation')
        products.update(CONSENSUS_MASK_SOURCES[source]['inputs'])
    required = set()
    if 'video' in products:
        video = dataset.settings.get('video')
        if video:
            video = Path(video).name if Path(video).is_absolute() else video
            required.add(str(file_path('video', filename=video)))
        else:
            videos = [name for name in files if Path(name).parent == folder_path('recording')]
            if len(videos) != 1:
                raise ValueError('Select the experimental video in the dataset before synchronizing')
            required.add(videos[0])
    if 'key_frames' in products:
        required.update(str(file_path(key)) for key in ('key_frames', 'key_frame_masks'))
    if 'masks' in products:
        for direction in ('forward', 'reverse'):
            selected = dataset.settings.get('masks', {}).get(direction)
            names = ([str(folder_path('masks') / selected)] if selected else
                     [str(file_path(direction).with_suffix(suffix)) for suffix in folder_suffixes('masks')])
            present = [name for name in names if name in files]
            if len(present) != 1:
                raise ValueError(f'Select one available {direction} mask file in the dataset; found {len(present)}')
            name = present[0]
            required.add(name)
            if Path(name).suffix == '.npy':
                companion = str(format_paths('masks', name, 'companions', format='npy')['validity'])
                if companion in files:
                    required.add(companion)
    if 'analysis' in products:
        required.update(str(file_path(key)) for key in ('mask_intersections', 'mask_areas', 'mask_iou_match',
                        'mask_quality', 'mask_missing', 'mask_switch', 'mask_quality_metadata'))
        temporal = str(file_path('mask_temporal_quality'))
        if temporal in files:
            required.add(temporal)
    if 'keypoints' in products:
        required.add(dataset.relative(dataset.keypoint_file()))
    if {'annotations', 'export'} & products:
        selected = annotation or dataset.settings.get('annotation')
        if selected is None:
            annotations = [name for name in files if Path(name).parent == folder_path('annotations')
                           and name.endswith('.json')]
            if len(annotations) != 1:
                raise ValueError('Select an annotation before synchronizing analysis inputs')
            selected = annotations[0]
        selected = dataset.relative(selected)
        required.add(selected)
        if 'export' in products:
            found_export = False
            for entry in entries:
                if entry.relative.as_posix() != selected or entry.is_directory:
                    continue
                remote = _remote(dataset, entry.host, profiles)
                record = (remote.call('read_json', path=str(entry.path),
                                     expected={'size': entry.size, 'modified': entry.modified})['value']
                          if remote else read_json(entry.path))
                if record.get('export'):
                    found_export = True
                    required.add(str(folder_path('masks') / record['export']['path']))
            if not found_export:
                raise ValueError('The selected annotation has no exported masks to synchronize')
    missing = required - files
    if missing:
        raise FileNotFoundError('Required files are unavailable in the registered locations: ' + ', '.join(sorted(missing)))
    # A new location needs a portable description; existing location settings are preserved.
    required.add(str(file_path('description')))
    return plan_consolidation(dataset, location, profiles, progress, required=required, entries=entries)
