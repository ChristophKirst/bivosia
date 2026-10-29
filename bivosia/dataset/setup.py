"""Create dataset descriptions and select recordings without moving source data.

Examples
--------
>>> from pathlib import Path
>>> from tempfile import TemporaryDirectory
>>> from bivosia.dataset.setup import create_dataset
>>> with TemporaryDirectory() as directory:
...     dataset = create_dataset(Path(directory) / 'experiment')
...     dataset.description.is_file()
True
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path

from bivosia.dataset.formats import format_handler, format_spec
from bivosia.dataset.layout import Dataset


def create_dataset(directory):
    """Create a description in an empty folder, refusing existing dataset contents."""
    dataset = Dataset(directory)
    if dataset.directory.exists() and any(dataset.directory.iterdir()):
        raise FileExistsError('Choose an empty folder for a new dataset, or open the existing dataset.')
    dataset.directory.mkdir(parents=True, exist_ok=True)
    dataset.save()
    return dataset


def select_video(dataset, path):
    """Validate the recording header and save a reference, preserving annotations."""
    format_spec('video', path)
    metadata = format_handler('video', path, operation='inspector')(path)
    if not metadata.frames or metadata.frames < 1:
        raise ValueError('The video has no readable frames')
    name = dataset.video_name(path)
    try:
        same = dataset.video == Path(path).expanduser().resolve()
    except (ValueError, FileNotFoundError):
        same = False
    if not same and (dataset.file('key_frames').exists() or dataset.file('key_frame_masks').exists()
                     or any(dataset.mask(direction).exists() for direction in ('forward', 'reverse'))
                     or dataset.annotation_files or dataset.array('mask_intersections').exists()):
        raise ValueError('This dataset already contains masks or annotations. Create a new dataset for a different recording.')
    previous = dataset.settings.copy()
    try:
        dataset.settings['video'] = name
        dataset.save()
    except BaseException:
        dataset.settings = previous
        raise
    return metadata
