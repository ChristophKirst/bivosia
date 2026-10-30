"""Load keypoint data through its format adapter and persist alignment results.

Examples
--------
With an existing dataset containing keypoint data:

>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.dataset.keypoints import load_keypoints
>>> dataset = Dataset("/path/to/dataset")
>>> data = load_keypoints(dataset)  # doctest: +SKIP
>>> frame = data[0]  # doctest: +SKIP
"""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import hashlib
import json

from bivosia.analysis.alignment import Alignment, validate_point_overlaps
from bivosia.analysis.keypoints import validate_keypoint_alignment
from bivosia.dataset.formats import format_handler
from bivosia.dataset.masks import open_masks
from bivosia.dataset.containers import open_container, container_output
from bivosia.dataset.keypoint_displacement import load_keypoint_displacement


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


def load_keypoints(dataset, *, masks=None):
    """Load optional keypoints and the alignment for the selected annotation."""
    path = dataset.keypoint_file()
    if not path.is_file():
        return None
    data = open_keypoints(dataset, masks=masks)
    alignment_path = dataset.array('keypoint_alignment')
    if alignment_path.is_file():
        with open_container(alignment_path) as file:
            annotation = dataset.annotation
            if annotation is None or annotation_digest(annotation) != file.attrs['annotation_sha256']:
                raise ValueError('Keypoint alignment belongs to a different annotation; recalculate it before viewing')
            if data.coordinate_scale != file.attrs['coordinate_scale']:
                raise ValueError('Keypoint coordinate annotation changed; recalculate alignment before viewing')
            data.alignment = Alignment(*(file[name][:] for name in ('assignment', 'valid', 'tied')))
            data.source = file['source'][:]
            data.scores = file['scores'][:]
            data.mask_present = file['mask_present'][:]
            data.n_masks = int(file.attrs['n_masks'])
            data.excluded_keypoints = tuple(json.loads(file.attrs['excluded_keypoints']))
        validate_keypoint_alignment(data.alignment, data.source, data.scores, data.mask_present,
                                   data.n_masks, data.shape[:2])
    data.displacement = load_keypoint_displacement(dataset, data)
    return data
