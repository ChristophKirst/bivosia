"""Render keypoint disks using original labels, assigned labels or both.

Examples
--------
>>> import numpy as np
>>> from bivosia.ui.keypoints import overlay_keypoints
>>> image = np.zeros((30, 30, 3), dtype=np.uint8)
>>> overlay_keypoints(image, np.array([[[15., 15.]]]), [1], (30, 30), 'both').shape
(30, 30, 3)
"""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import numpy as np
import cv2

from bivosia.ui.colors import MASK_COLORS_BGR
from bivosia.ui.plot_models import SeriesSpec
from bivosia.ui.plot_styles import METRIC_COLORS


def configure_displacement_plot(graph, maximal):
    """Plot D_t at frame t for the transition t to t+1, retaining infinity markers."""
    graph.set_series((SeriesSpec(
        'keypoint_displacement', 'Maximal displacement (source pixels); ∞ = no common keypoints',
        maximal, frames=np.arange(len(maximal)),
        color=METRIC_COLORS['mean_centroid_displacement_px'],
        temporal_extrema=True, show_infinity=True),))
    graph.fit_shared_y_range()
    graph.setToolTip('At frame t: maximum across skeletons of mean displacement from t to t+1. '
                     'Only analysis-selected keypoints with finite xy in both frames contribute. '
                     'Distances are in original image pixels. ∞ marks no common keypoints '
                     'for at least one skeleton. The final frame has no following-frame score.')


def overlay_keypoints(image, points, assignment, image_shape, mode, label_colors=None):
    """Draw an original-label inner disk and an assigned-label outer disk.

    Original mode draws every finite point. Assigned mode draws only assigned
    tracks. Both mode retains the original disk when no assignment exists.
    Coordinates use the source image size, independently of mask resolution.
    Supplied mask-label colors affect assigned disks only; original disks
    always retain their keypoint track colors.
    """
    result = image.copy()
    height, width = image.shape[:2]
    scale = np.array([width / image_shape[1], height / image_shape[0]])
    radius = max(2, round(min(height, width) / 256))
    colors = MASK_COLORS_BGR if label_colors is None else label_colors
    for track, nodes in enumerate(points):
        original = tuple(int(c) for c in MASK_COLORS_BGR[track % len(MASK_COLORS_BGR)])
        label = int(assignment[track])
        assigned = (tuple(int(c) for c in colors[label % len(colors)])
                    if label >= 0 else None)
        if mode == 'assigned' and assigned is None:
            continue
        for x, y in np.rint(nodes * scale):
            if not np.isfinite((x, y)).all() or not (0 <= x < width and 0 <= y < height):
                continue
            center = int(x), int(y)
            if mode == 'both' and assigned is not None:
                cv2.circle(result, center, radius + max(2, radius // 2), assigned, -1, cv2.LINE_AA)
            cv2.circle(result, center, radius, assigned if mode == 'assigned' else original, -1, cv2.LINE_AA)
    return result
