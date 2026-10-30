"""Validate and add existing keypoint tracks to a dataset.

Examples
--------
>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.dataset.keypoint_import import select_keypoints
>>> dataset = Dataset('/path/to/dataset')
>>> select_keypoints(dataset, '/path/to/tracks.h5', (1536, 2048))  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import closing
from pathlib import Path

from bivosia.dataset.file_operations import FileDestination, copy_bytes, file_state
from bivosia.dataset.formats import format_handler, format_spec
from bivosia.dataset.paths import folder_path
from bivosia.dataset.video import open_video


def keypoint_image_shape(dataset):
    """Default to saved keypoint dimensions or the selected video's reported size."""
    saved = dataset.settings.get('keypoints', {}).get('image_shape')
    if saved is not None:
        return tuple(saved)
    try:
        with closing(open_video(dataset.video)) as video:
            return video.shape
    except (OSError, ValueError):
        return (0, 0)


def select_keypoints(dataset, path, image_shape):
    """Validate SLEAP tracks, copy them if needed, and save their source resolution.

    Files already in a registered keypoints folder are used in place. Other files
    are copied into the configured keypoints folder without replacing any file.
    Existing alignment and displacement results protect their source selection.
    """
    path = Path(path).expanduser().resolve()
    format_spec('keypoints', path)
    previous = dataset.settings.get('keypoints', {})
    state = file_state(path)
    try:
        points = format_handler('keypoints', format='sleap')(
            path, image_shape, invalid_frames=previous.get('invalid_frames', ()))
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ValueError(f'Invalid SLEAP analysis file or source image dimensions: {error}') from error
    image_shape = [int(value) for value in points.source_image_shape]
    same = path == dataset.keypoint_file().resolve()
    changed = (not same or previous.get('image_shape') != image_shape
               or previous.get('format', 'sleap') != 'sleap')
    if changed and any(dataset.array(name).exists()
                       for name in ('keypoint_alignment', 'keypoint_displacement')):
        raise ValueError('Existing keypoint results use the current file and image dimensions. '
                         'Move those results out of the dataset before changing this selection.')

    folder = folder_path('keypoints')
    try:
        relative = Path(dataset.relative(path))
        name = relative.relative_to(folder)
    except ValueError:
        name = Path(path.name)
        relative = folder / name
    destination = dataset.path(relative)
    if destination.resolve() != path:
        if destination.exists():
            raise FileExistsError(f'A keypoint file already exists: {destination}. '
                                  'Select that file or rename the source file before importing.')
        destination = dataset.path(relative, write=True)
        with FileDestination(destination, state) as output:
            with path.open('rb') as source:
                checksum = copy_bytes(source, output.stream, state['size'])
            if file_state(path) != state:
                raise ValueError('Keypoint file changed during import; select it again.')
            output.publish(checksum)
    elif file_state(path) != state:
        raise ValueError('Keypoint file changed during validation; select it again.')

    settings = dataset.settings.copy()
    try:
        dataset.settings['keypoints'] = {
            **previous, 'format': 'sleap', 'file': name.as_posix(), 'image_shape': image_shape,
        }
        dataset.save()
    except BaseException:
        dataset.settings = settings
        raise
    return destination
