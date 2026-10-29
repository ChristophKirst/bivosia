"""Persist per-skeleton and maximal keypoint displacement in original image pixels.

Examples
--------
With an existing dataset containing keypoints and saved displacement results:

>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.dataset.keypoints import open_keypoints
>>> from bivosia.dataset.keypoint_displacement import load_keypoint_displacement
>>> dataset = Dataset("/path/to/dataset")
>>> points = open_keypoints(dataset, source_coordinates=True)  # doctest: +SKIP
>>> maximal = load_keypoint_displacement(dataset, points)  # doctest: +SKIP
"""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import json

from bivosia.analysis.keypoint_displacement import validate_keypoint_displacement
from bivosia.dataset.containers import open_container, container_output


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


def load_keypoint_displacement(dataset, points):
    """Load the saved maximum, validating dimensions and keypoint coordinate metadata."""
    path = dataset.array('keypoint_displacement')
    if not path.is_file():
        return None
    with open_container(path) as file:
        if (file.attrs['units'] != points.source_units
                or tuple(file.attrs['image_shape']) != points.source_image_shape
                or json.loads(file.attrs['node_names']) != points.node_names
                or json.loads(file.attrs['track_names']) != points.track_names
                or tuple(json.loads(file.attrs['excluded_keypoints'])) != points.excluded_keypoints
                or file.attrs['frame_count'] != points.shape[0]):
            raise ValueError('Displacement metadata does not match the recording keypoints')
        individual, maximal = file['individual'][:], file['maximal'][:]
    validate_keypoint_displacement(individual, maximal, *points.shape[:2])
    return maximal
