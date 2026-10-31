"""Import keypoint tracks and persist alignment and displacement results.

Examples
--------
With an existing dataset containing keypoint data:

>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.dataset.keypoints import load_keypoints
>>> dataset = Dataset("/path/to/dataset")
>>> data = load_keypoints(dataset)  # doctest: +SKIP
>>> frame = data.reader[0]  # doctest: +SKIP
"""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np

from bivosia.analysis.alignment import AlignmentResult, validate_point_overlaps
from bivosia.analysis.keypoints import validate_keypoint_alignment, validate_keypoint_displacement
from bivosia.dataset.io.containers import open_container, container_output
from bivosia.dataset.io.interfaces import KeypointReader
from bivosia.dataset.storage.file_operations import FileDestination, copy_bytes, file_state
from bivosia.dataset.io.registry import format_handler, format_spec
from bivosia.dataset.masks import open_masks
from bivosia.dataset.paths import folder_path
from bivosia.dataset.video import open_video


@dataclass
class KeypointInputs:
    """A coordinate reader with optional saved alignment and displacement results."""

    reader: KeypointReader
    alignment: AlignmentResult | None = None
    source: np.ndarray | None = None
    scores: np.ndarray | None = None
    mask_present: np.ndarray | None = None
    n_masks: int | None = None
    excluded_keypoints: tuple[str, ...] = ()
    displacement: np.ndarray | None = None

    @property
    def selected_node_ids(self):
        """Indices of nodes included in analysis measures and frame annotations."""
        return [i for i, name in enumerate(self.reader.node_names) if name not in self.excluded_keypoints]


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


def annotation_digest(path):
    """Fingerprint the annotation used to select the alignment masks."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def open_keypoints(dataset, *, source_coordinates=False, masks=None):
    """Open the selected keypoint adapter in mask or original image coordinates."""
    try:
        metadata = dataset.settings['keypoints']
        image_shape = metadata['image_shape']
    except KeyError as error:
        raise ValueError('Annotate the keypoint source image_shape in dataset.json under keypoints') from error
    kind = metadata.get('format')
    reader = format_handler('keypoints', dataset.keypoint_file(), format=kind)
    def open_points(mask):
        return reader(dataset.keypoint_file(), image_shape,
                      target_shape=None if source_coordinates else mask.image_shape,
                      frame_count=mask.shape[0], invalid_frames=metadata.get('invalid_frames', ()))
    if masks is not None:
        return open_points(masks)
    with open_masks(dataset.mask('forward'), require_valid=False, direction='forward') as masks:
        return open_points(masks)


def write_keypoint_alignment(path, result, source, scores, mask_present, overlaps, usable, n_masks, metadata):
    """Validate and atomically create alignment results without replacing an existing result."""
    validate_keypoint_alignment(result, source, scores, mask_present, n_masks, result.assignment.shape)
    nodes = json.loads(metadata['node_names'])
    excluded = json.loads(metadata['excluded_keypoints'])
    validate_point_overlaps(overlaps, usable, scores, mask_present, result.valid, nodes, excluded)
    with container_output(path, overwrite=False) as output:
        for name in ('assignment', 'valid', 'tied'):
            output.create_dataset(name, data=getattr(result, name), compression='lzf')
        output.create_dataset('source', data=source, compression='lzf')
        output.create_dataset('scores', data=scores, compression='lzf')
        output.create_dataset('mask_present', data=mask_present, compression='lzf')
        write_point_overlaps(output, overlaps, usable)
        output.attrs['n_masks'] = n_masks
        for name, value in metadata.items():
            output.attrs[name] = value


def write_point_overlaps(output, overlaps, usable):
    """Store per-node containment and point usability in an open alignment file."""
    array = output.create_dataset('overlaps', data=overlaps, compression='lzf')
    array.attrs['axes'] = 'frame, track, node, mask'
    array.attrs['mask_labels'] = 'column c refers to local mask label c+1 in source[frame]'
    array = output.create_dataset('point_usable', data=usable, compression='lzf')
    array.attrs['definition'] = 'finite xy, rounded inside image, on a valid alignment frame'


def load_keypoints(dataset, *, masks=None) -> KeypointInputs | None:
    """Load optional keypoints and the alignment for the selected annotation."""
    path = dataset.keypoint_file()
    if not path.is_file():
        return None
    data = KeypointInputs(open_keypoints(dataset, masks=masks))
    alignment_path = dataset.array('keypoint_alignment')
    if alignment_path.is_file():
        with open_container(alignment_path) as file:
            annotation = dataset.annotation
            if annotation is None or annotation_digest(annotation) != file.attrs['annotation_sha256']:
                raise ValueError('Keypoint alignment belongs to a different annotation; recalculate it before viewing')
            if data.reader.coordinate_scale != file.attrs['coordinate_scale']:
                raise ValueError('Keypoint coordinate annotation changed; recalculate alignment before viewing')
            data.alignment = AlignmentResult(*(file[name][:] for name in ('assignment', 'valid', 'tied')))
            data.source = file['source'][:]
            data.scores = file['scores'][:]
            data.mask_present = file['mask_present'][:]
            data.n_masks = int(file.attrs['n_masks'])
            data.excluded_keypoints = tuple(json.loads(file.attrs['excluded_keypoints']))
        validate_keypoint_alignment(data.alignment, data.source, data.scores, data.mask_present,
                                   data.n_masks, data.reader.shape[:2])
    data.displacement = load_keypoint_displacement(dataset, data.reader, exclude_keypoints=data.excluded_keypoints)
    return data


def write_keypoint_displacement(dataset, points, individual, maximal, *, exclude_keypoints, overwrite=False):
    """Save displacement with the selected nodes and coordinate metadata."""
    validate_keypoint_displacement(individual, maximal, *points.shape[:2])
    path = dataset.array('keypoint_displacement', write=True)
    with container_output(path, overwrite=overwrite) as file:
        file.create_dataset('individual', data=individual, compression='lzf')
        file.create_dataset('maximal', data=maximal, compression='lzf')
        file.attrs['units'] = points.source_units
        file.attrs['transition'] = 'index t compares frames t and t+1'
        file.attrs['frame_count'] = points.shape[0]
        file.attrs['image_shape'] = points.source_image_shape
        file.attrs['node_names'] = json.dumps(points.node_names)
        file.attrs['track_names'] = json.dumps(points.track_names)
        file.attrs['excluded_keypoints'] = json.dumps(list(exclude_keypoints))
        file.attrs['keypoints'] = 'selected nodes; finite xy in both frames'
    return path


def load_keypoint_displacement(dataset, points, *, exclude_keypoints):
    """Load the saved maximum, validating dimensions and keypoint coordinate metadata."""
    path = dataset.array('keypoint_displacement')
    if not path.is_file():
        return None
    with open_container(path) as file:
        if (file.attrs['units'] != points.source_units
                or tuple(file.attrs['image_shape']) != points.source_image_shape
                or json.loads(file.attrs['node_names']) != points.node_names
                or json.loads(file.attrs['track_names']) != points.track_names
                or tuple(json.loads(file.attrs['excluded_keypoints'])) != tuple(exclude_keypoints)
                or file.attrs['frame_count'] != points.shape[0]):
            raise ValueError('Displacement metadata does not match the recording keypoints')
        individual, maximal = file['individual'][:], file['maximal'][:]
    validate_keypoint_displacement(individual, maximal, *points.shape[:2])
    return maximal
