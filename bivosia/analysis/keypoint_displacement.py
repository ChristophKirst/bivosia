"""Displacement of shared keypoints between consecutive keypoint skeletons.

Examples
--------
>>> import numpy as np
>>> from bivosia.analysis.keypoint_displacement import keypoint_displacement
>>> points = np.array([[[[0., 0.], [np.nan, np.nan]]],
...                    [[[3., 4.], [1., 1.]]],
...                    [[[np.nan, np.nan], [np.nan, np.nan]]]])
>>> individual, maximal = keypoint_displacement(points)
>>> individual[:, 0].tolist(), maximal.tolist()
([5.0, inf], [5.0, inf])
"""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import numpy as np


def keypoint_displacement(points):
    """Return per-track means and their maximum for each adjacent frame pair.

    Points have shape (frames, tracks, nodes, xy). A shared node must have
    finite x and y at both endpoints. An empty intersection gives infinity.
    Output index t describes the transition from frame t to frame t+1.
    All supplied nodes contribute, independently of point-to-mask alignment.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 4 or points.shape[-1] != 2 or points.shape[1] == 0:
        raise ValueError('Expected points with shape (frames, tracks, nodes, 2) and at least one track')
    finite = np.isfinite(points).all(axis=-1)
    shared = finite[:-1] & finite[1:]
    delta = np.zeros_like(points[1:])
    np.subtract(points[1:], points[:-1], out=delta, where=shared[..., None])
    distances = np.linalg.norm(delta, axis=-1)
    counts = shared.sum(axis=-1)
    individual = np.divide(distances.sum(axis=-1), counts,
                           out=np.full(counts.shape, np.inf), where=counts > 0)
    return individual, individual.max(axis=1)


def validate_keypoint_displacement(individual, maximal, frame_count, track_count):
    """Check transition dimensions, nonnegative scores and track maxima."""
    individual, maximal = np.asarray(individual), np.asarray(maximal)
    if individual.shape != (max(0, frame_count - 1), track_count) or maximal.shape != individual.shape[:1]:
        raise ValueError('Displacement arrays must cover every consecutive frame pair and track')
    if np.any(np.isnan(individual)) or np.any(individual < 0):
        raise ValueError('Displacements must be nonnegative or positive infinity')
    if not np.array_equal(maximal, individual.max(axis=1)):
        raise ValueError('Maximal displacement differs from the per-track maximum')
