"""Calculate recording keypoint counts and displacement in bounded blocks.

Examples
--------
>>> import numpy as np
>>> from bivosia.processing.keypoints import calculate_keypoint_counts
>>> calculate_keypoint_counts(np.zeros((2, 1, 3, 2)), block_size=1).tolist()
[[3], [3]]
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import logging

import numpy as np

from bivosia.config import PROCESSING_DEFAULTS
from bivosia.analysis.keypoints import keypoint_counts, keypoint_displacement
from bivosia.dataset.keypoints import open_keypoints
from bivosia.dataset.keypoint_displacement import write_keypoint_displacement
from bivosia.processing.log import Progress


logger = logging.getLogger(__name__)


def calculate_keypoint_counts(points, *, node_ids=None,
                              block_size=PROCESSING_DEFAULTS['keypoints']['block_size'],
                              progress=None, cancelled=None):
    """Read selected finite-point counts per frame and track in bounded blocks.

    >>> import numpy as np
    >>> from bivosia.processing.keypoints import calculate_keypoint_counts
    >>> points = np.array([[[[np.nan, np.nan], [1., 2.], [3., 4.]]]])
    >>> calculate_keypoint_counts(points, node_ids=[0], block_size=1).tolist()
    [[0]]
    """
    if block_size < 1:
        raise ValueError('Block size must be positive')
    count, tracks = points.shape[:2]
    counts = np.empty((count, tracks), dtype=np.intp)
    report = Progress('Keypoint counts', count, logger=logger, callback=progress, cancelled=cancelled)
    for start in range(0, count, block_size):
        report.check_cancelled()
        stop = min(start + block_size, count)
        block = points[start:stop]
        if node_ids is not None:
            block = block[:, :, node_ids, :]
        counts[start:stop] = keypoint_counts(block)
        report.update(stop - start)
    report.finish()
    return counts


def calculate_keypoint_displacement(points, *, node_ids=None,
                                    block_size=PROCESSING_DEFAULTS['keypoints']['block_size'],
                                    progress=None, cancelled=None):
    """Calculate consecutive-frame displacements, retaining block-boundary pairs."""
    if block_size < 1:
        raise ValueError('Block size must be positive')
    count, tracks = points.shape[:2]
    transitions = max(0, count - 1)
    individual = np.empty((transitions, tracks), dtype=np.float64)
    maximal = np.empty(transitions, dtype=np.float64)
    report = Progress('Keypoint displacement', transitions, logger=logger,
                      callback=progress, cancelled=cancelled)
    for start in range(0, transitions, block_size):
        report.check_cancelled()
        stop = min(start + block_size, transitions)
        block = points[start:stop + 1]
        if node_ids is not None:
            block = block[:, :, node_ids, :]
        individual[start:stop], maximal[start:stop] = keypoint_displacement(block)
        report.update(stop - start)
    report.finish()
    return individual, maximal


def process_keypoint_displacement(dataset, *, exclude_keypoints, overwrite=False,
                                 block_size=PROCESSING_DEFAULTS['keypoints']['block_size'],
                                 progress=None, cancelled=None):
    """Save selected-node displacement for the recording's frame range in source pixels."""
    points = open_keypoints(dataset, source_coordinates=True)
    exclude_keypoints = tuple(exclude_keypoints)
    for name in exclude_keypoints:
        if name not in points.node_names:
            raise ValueError(f'Unknown keypoint to exclude: {name}')
    points.excluded_keypoints = exclude_keypoints
    individual, maximal = calculate_keypoint_displacement(
        points, node_ids=points.selected_node_ids, block_size=block_size, progress=progress, cancelled=cancelled)
    path = write_keypoint_displacement(dataset, points, individual, maximal,
                                      exclude_keypoints=exclude_keypoints, overwrite=overwrite)
    logger.info('Saved keypoint displacement: %s', path)
    return path
