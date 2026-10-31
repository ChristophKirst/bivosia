"""Interfaces for frame-indexed masks, keypoints, videos and file metadata."""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from bivosia.analysis.keypoints import point_coordinate_scale


@dataclass(frozen=True)
class FileMetadata:
    """Array headers or video metadata; duration is in seconds."""

    arrays: tuple = ()
    frames: int | None = None
    duration: float | None = None


class KeypointReader(ABC):
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

    def __getitem__(self, frames):
        """Read coordinates in the pixel space declared by the dataset metadata."""
        points = self._read(frames)
        return points if self.coordinate_scale == 1 else points * self.coordinate_scale

    @abstractmethod
    def _read(self, frames):
        """Read native coordinates for one frame or a frame slice."""


class MaskReader(ABC):
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


class VideoReader(ABC):
    """Decoded frames with source frame counts, image dimensions and frame rate."""

    count: int
    shape: tuple[int, int]
    fps: float

    @abstractmethod
    def read(self, frame):
        """Read one frame in the color order selected when opening the reader."""

    @abstractmethod
    def close(self):
        """Release decoder resources."""

    def frames(self, start, end, direction, cancelled=None, buffer_bytes=128 * 1024**2):
        """Yield inclusive source-index/image pairs, without re-encoding the video."""
        if direction not in ('forward', 'reverse') or not 0 <= start <= end < self.count:
            raise ValueError('Invalid video interval or direction')
        step = max(1, buffer_bytes // (self.shape[0] * self.shape[1] * 3))
        if direction == 'forward':
            for frame in range(start, end + 1):
                if cancelled and cancelled():
                    raise InterruptedError('Video reading cancelled')
                yield frame, self.read(frame)
        else:
            last = end
            while last >= start:
                first = max(start, last - step + 1)
                block = []
                for frame in range(first, last + 1):
                    if cancelled and cancelled():
                        raise InterruptedError('Video reading cancelled')
                    block.append((frame, self.read(frame)))
                yield from reversed(block)
                last = first - 1
