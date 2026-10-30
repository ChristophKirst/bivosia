"""Transparent mask thumbnails for embedding plots."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import closing

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage

from bivosia.config import ANALYSIS_DEFAULTS
from bivosia.analysis.masks import preprocessing_settings
from bivosia.analysis.mask_features import prepare_shape_mask
from bivosia.dataset.shape_features import sample_masks


def mask_images(dataset, frames, identities, colors, settings, source, cancelled):
    """Crop selected masks at the resolution and preprocessing used for their features."""
    preprocessing = preprocessing_settings(settings['preprocess_masks'], **{
        key: settings[key] for key in ANALYSIS_DEFAULTS['masks']['preprocessing']})
    images = {}
    with closing(sample_masks(dataset, frames, identities,
                 source=source['mask_source'], expected=source)) as masks:
        for index, mask in masks:
            if cancelled():
                return images
            mask = prepare_shape_mask(mask, downsample_factor=settings['downsample_factor'],
                                      preprocessing=preprocessing)
            y, x = np.nonzero(mask)
            if not len(x):
                raise ValueError('A selected sample no longer contains a mask')
            crop = mask[y.min():y.max() + 1, x.min():x.max() + 1]
            pixels = np.zeros((*crop.shape, 4), dtype=np.uint8)
            pixels[crop] = colors[index]
            image = QImage(pixels.data, crop.shape[1], crop.shape[0], pixels.strides[0],
                           QImage.Format.Format_RGBA8888).copy()
            images[index] = image.scaled(128, 128, Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation)
    return images
