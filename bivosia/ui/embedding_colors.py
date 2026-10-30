"""Consistent sample and animal colors for embeddings and their frame timelines."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import numpy as np
from PySide6.QtGui import QLinearGradient

from bivosia.ui.colors import identity_colors


def embedding_bounds(points):
    low, high = points.min(axis=0), points.max(axis=0)
    padding = np.maximum(high - low, 1.) * .03
    return low - padding, high + padding


def sample_gradient():
    gradient = QLinearGradient()
    for position, color in zip(np.linspace(0, 1, 6), identity_colors(6)):
        gradient.setColorAt(float(position), color)
    return gradient


def embedding_colors(points, identities, mode):
    """Use animal identity colors or the same UMAP-2 gradient as Qt's 3D scatter."""
    if not len(points):
        return np.empty((0, 4), dtype=np.uint8)
    if mode == 'animal':
        palette = np.array([color.getRgb() for color in identity_colors(int(identities.max()))], dtype=np.uint8)
        return palette[identities - 1]
    low, high = embedding_bounds(points)
    values = (points[:, 1] - low[1]) / (high[1] - low[1])
    stops = sample_gradient().stops()
    positions, colors = zip(*stops)
    rgba = np.array([color.getRgb() for color in colors])
    return np.rint(np.column_stack([np.interp(values, positions, rgba[:, channel])
                                   for channel in range(4)])).astype(np.uint8)
