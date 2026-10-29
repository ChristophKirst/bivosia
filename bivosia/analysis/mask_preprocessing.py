"""Select body fragments and remove thin protrusions before mask analysis.

Examples
--------
>>> import numpy as np
>>> from bivosia.analysis.mask_preprocessing import preprocess_mask
>>> mask = np.zeros((12, 24), dtype=bool)
>>> mask[2:6, 2:6] = mask[2:5, 8:11] = mask[9, 21] = True
>>> cleaned = preprocess_mask(mask, main_body_area=16, fragment_distance=8, opening_radius=0)
>>> int(cleaned.sum()), int(mask.sum())
(25, 26)
>>> int(preprocess_mask(mask, main_body_area=15, fragment_distance=8, opening_radius=0).sum())
16
>>> int(preprocess_mask(mask, main_body_area=16, fragment_distance=2, opening_radius=0).sum())
16
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import operator

import numpy as np
import cv2

from bivosia.config import ANALYSIS_DEFAULTS


def preprocessing_settings(enabled, **parameters):
    """Return validated settings, or None when preprocessing is disabled."""
    if not enabled:
        return None
    settings = dict(ANALYSIS_DEFAULTS['mask_preprocessing'])
    if parameters.keys() - settings.keys():
        raise ValueError('Unknown mask preprocessing parameter')
    settings.update(parameters)
    for name in ('filter_by_area', 'filter_by_distance', 'apply_opening'):
        if not isinstance(settings[name], (bool, np.bool_)):
            raise ValueError(f'{name} must be boolean')
        settings[name] = bool(settings[name])
    for name in ('main_body_area', 'fragment_distance'):
        if not np.isfinite(settings[name]) or settings[name] < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    try:
        radius = operator.index(settings['opening_radius'])
    except TypeError:
        raise ValueError('Opening radius must be a nonnegative integer') from None
    if isinstance(settings['opening_radius'], (bool, np.bool_)) or radius < 0:
        raise ValueError('Opening radius must be a nonnegative integer')
    settings['opening_radius'] = radius
    return settings


def preprocess_mask(mask, *,
                    filter_by_area=ANALYSIS_DEFAULTS['mask_preprocessing']['filter_by_area'],
                    main_body_area=ANALYSIS_DEFAULTS['mask_preprocessing']['main_body_area'],
                    filter_by_distance=ANALYSIS_DEFAULTS['mask_preprocessing']['filter_by_distance'],
                    fragment_distance=ANALYSIS_DEFAULTS['mask_preprocessing']['fragment_distance'],
                    apply_opening=ANALYSIS_DEFAULTS['mask_preprocessing']['apply_opening'],
                    opening_radius=ANALYSIS_DEFAULTS['mask_preprocessing']['opening_radius']):
    """Return a separate body mask at the supplied image resolution.

    Area filtering retains only the largest 8-connected component when its
    foreground pixel count exceeds ``main_body_area``. Distance filtering
    retains the two largest remaining components when their centroid distance
    is at most ``fragment_distance``; otherwise it retains only the largest.
    Equal areas retain component scan order. Each disabled filter leaves all
    current components intact. Opening applies erosion followed by dilation
    with a disk of ``opening_radius`` pixels (zero skips opening). Pixels
    outside the image are background. Each step can be disabled independently.
    Holes remain unless altered by opening; source pixels are never modified.
    """
    settings = preprocessing_settings(True, filter_by_area=filter_by_area, main_body_area=main_body_area,
                                      filter_by_distance=filter_by_distance, fragment_distance=fragment_distance,
                                      apply_opening=apply_opening, opening_radius=opening_radius)
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != np.bool_ or not all(mask.shape):
        raise ValueError('Expected a nonempty two-dimensional boolean image')
    result = mask.copy()
    if settings['filter_by_area'] or settings['filter_by_distance']:
        count, labels, statistics, centers = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8)
        if count == 1:
            return result
        areas = statistics[1:, cv2.CC_STAT_AREA]
        keep = np.argsort(-areas, kind='stable') + 1
        if settings['filter_by_area'] and areas[keep[0] - 1] > settings['main_body_area']:
            keep = keep[:1]
        if settings['filter_by_distance'] and len(keep) > 1:
            nearby = np.linalg.norm(centers[keep[0]] - centers[keep[1]]) <= settings['fragment_distance']
            keep = keep[:2] if nearby else keep[:1]
        result = np.isin(labels, keep)
    radius = settings['opening_radius']
    if settings['apply_opening'] and radius:
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        disk = (x * x + y * y <= radius * radius).astype(np.uint8)
        result = cv2.morphologyEx(result.astype(np.uint8), cv2.MORPH_OPEN, disk,
                                  borderType=cv2.BORDER_CONSTANT, borderValue=0) != 0
    return result
