"""Interfaces for frame-indexed point coordinates and labeled mask images."""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from abc import ABC, abstractmethod

import numpy as np

from bivosia.analysis.keypoints import point_coordinate_scale

class PointData(ABC):
    """Points in image coordinates, with shape (frames, tracks, nodes, xy).

    Single points use one node per track. ``valid`` gives frame validity;
    individual missing points use NaN. ``image_shape`` is (height, width).
    Integer and slice indexing return NumPy arrays in frame order. Readers
    preserve native dimensions in ``source_image_shape`` and can convert to
    a target image size before returning coordinates.
    """

    shape: tuple[int, int, int, int]
    dtype: np.dtype
    image_shape: tuple[int, int]
    valid: np.ndarray
    node_names: list[str]
    track_names: list[str]
    source_units = 'original image pixels'

    def __init__(self, image_shape, target_shape=None):
        self.source_image_shape = tuple(image_shape)
        self.image_shape = tuple(image_shape if target_shape is None else target_shape)
        self.coordinate_scale = point_coordinate_scale(self.source_image_shape, self.image_shape)
        self.alignment = None
        self.source = None
        self.scores = None
        self.mask_present = None
        self.n_masks = None
        self.excluded_keypoints = ()
        self.displacement = None

    @property
    def selected_node_ids(self):
        """Indices of nodes included in analysis measures and frame annotations."""
        return [i for i, name in enumerate(self.node_names) if name not in self.excluded_keypoints]

    def __getitem__(self, frames):
        """Read coordinates in the pixel space declared by the dataset metadata."""
        points = self._read(frames)
        return points if self.coordinate_scale == 1 else points * self.coordinate_scale

    @abstractmethod
    def _read(self, frames):
        """Read native coordinates for one frame or a frame slice."""


class MaskData(ABC):
    """Integer labels with shape (frames, height, width) and frame validity.

    Background is 0; object labels are positive integers. Integer and slice
    indexing return NumPy arrays in frame order. Standalone readers own their
    resources; streams sharing a container are limited to that container's lifetime.
    """

    shape: tuple[int, int, int]
    dtype: np.dtype
    valid: np.ndarray

    @property
    def image_shape(self):
        return self.shape[1:]

    @abstractmethod
    def __getitem__(self, frames):
        """Read one frame or a frame slice."""

    @abstractmethod
    def close(self):
        """Release reader resources."""

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
