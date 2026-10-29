"""Read SLEAP analysis HDF5 coordinates through the shared point interface.

Examples
--------
>>> from bivosia.dataset.sleap import SleapPoints
>>> points = SleapPoints('tracks.h5', image_shape=(1536, 2048))  # doctest: +SKIP
>>> points[0].shape  # doctest: +SKIP
(2, 11, 2)
"""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path

import h5py
import numpy as np

from bivosia.analysis.alignment import alignment_valid_frames
from bivosia.analysis.keypoints import validate_keypoints
from bivosia.dataset.frame_data import PointData


class SleapPoints(PointData):
    """Adapt SLEAP's (track, xy, node, frame) storage to frame-major points."""

    source_units = 'original SLEAP image pixels'

    def __init__(self, path, image_shape, *, target_shape=None, frame_count=None, invalid_frames=()):
        super().__init__(image_shape, target_shape)
        self.path = Path(path)
        with h5py.File(self.path, 'r') as file:
            tracks = file['tracks']
            if tracks.ndim != 4 or tracks.shape[1] != 2:
                raise ValueError('SLEAP tracks must have shape (tracks, 2, nodes, frames)')
            count = tracks.shape[3] if frame_count is None else frame_count
            if not isinstance(count, (int, np.integer)) or not 0 < count <= tracks.shape[3]:
                raise ValueError('SLEAP data must cover the requested frame count')
            self.shape = (count, tracks.shape[0], tracks.shape[2], 2)
            self.dtype = tracks.dtype
            self.node_names = file['node_names'].asstr()[:].tolist()
            self.track_names = file['track_names'].asstr()[:].tolist()
        self.valid = alignment_valid_frames(np.ones(self.shape[0], dtype=bool), invalid_frames)
        validate_keypoints(self, self.valid, self.image_shape)
        if len(self.node_names) != self.shape[2] or len(self.track_names) != self.shape[1]:
            raise ValueError('SLEAP names do not match the node and track dimensions')

    def _read(self, frames):
        if isinstance(frames, slice):
            frames = slice(*frames.indices(self.shape[0]))
            if frames.step <= 0:
                raise ValueError('Point frame slices must have a positive step')
        else:
            if not isinstance(frames, (int, np.integer)):
                raise TypeError('Point frames require an integer or slice')
            frames = frames + self.shape[0] if frames < 0 else frames
            if not 0 <= frames < self.shape[0]:
                raise IndexError('Point frame outside the selected recording interval')
        with h5py.File(self.path, 'r') as file:
            points = file['tracks'][..., frames]
        return points.transpose(0, 2, 1) if points.ndim == 3 else points.transpose(3, 0, 2, 1)
