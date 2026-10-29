"""Shared animal identity, linked-zone and key-frame colors."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import colorsys

import numpy as np
from PySide6.QtGui import QColor


KEY_FRAME_COLORS = {"proposed": "#e4ae42", "labeled": "#45c47c", "boundary": "#a783d5"}
WORKFLOW_COLORS = {"available": "#45c47c", "missing": "#e4ae42", "review": "#a783d5", "checking": "#888888"}
MASK_COLORS_BGR = np.asarray(
    # Original six-vole rendering palette, in label order.
    ((120, 80, 225), (210, 85, 25), (25, 200, 100),
     (200, 25, 200), (25, 200, 200), (30, 215, 255)), dtype=np.uint8)


def identity_colors(n, offset=0.):
    return [QColor.fromRgbF(*colorsys.hsv_to_rgb((offset + i / max(n, 1)) % 1., .68, .95)) for i in range(n)]


def component_colors(n, component):
    return identity_colors(n, .27 + component * .38196601125)


def linked_zone_colors(n):
    """Distribute distinct blue shades across neighboring linked components.

    >>> from bivosia.ui.colors import linked_zone_colors
    >>> [round(color.lightnessF(), 3) for color in linked_zone_colors(4)]
    [0.9, 0.667, 0.55, 0.783]
    >>> linked_zone_colors(0)
    []
    """
    lightness = np.linspace(.9, .55, n)
    # Golden-ratio spacing keeps nearby component numbers apart in the gradient.
    ranks = np.argsort(np.argsort((np.arange(n) * .38196601125) % 1.))
    return [QColor.fromHslF(.58, .5, float(value)) for value in lightness[ranks]]


def bgr_colors(colors):
    return np.asarray([(c.blue(), c.green(), c.red()) for c in colors], dtype=np.uint8)
