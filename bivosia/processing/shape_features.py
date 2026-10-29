"""Extract consensus-mask Zernike features and run standard UMAP.

Examples
--------
>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.processing.shape_features import shape_features_job
>>> dataset = Dataset('/path/to/dataset')
>>> outputs = shape_features_job(dataset, radius=100, z_score=True)  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from importlib.metadata import version
import logging

import numpy as np

from bivosia.config import ANALYSIS_DEFAULTS, PROCESSING_DEFAULTS
from bivosia.analysis.embedding import umap_embedding
from bivosia.analysis.mask_preprocessing import preprocessing_settings
from bivosia.analysis.shape_features import frame_zernike_features, validate_shape_metadata, zernike_indices
from bivosia.dataset.analysis_inputs import require_new_outputs
from bivosia.dataset.containers import container_output
from bivosia.dataset.masks import open_linked_masks
from bivosia.dataset.shape_features import (
    append_zernike, consensus_source, create_zernike, embedding_path,
    read_embedding, read_zernike, write_embedding, zernike_path,
)
from bivosia.processing.log import Progress


logger = logging.getLogger(__name__)


def extract_zernike(dataset, radius, *, degree=ANALYSIS_DEFAULTS['shape_features']['degree'],
                    block_size=PROCESSING_DEFAULTS['shape_features']['block_size'],
                    preprocess_masks=ANALYSIS_DEFAULTS['shape_features']['preprocess_masks'],
                    progress=None, cancelled=None, **preprocessing_parameters):
    """Stream present animals from valid consensus frames into a descriptor file."""
    if block_size < 1:
        raise ValueError('block_size must be positive')
    zernike_indices(degree)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Choose a positive Zernike radius in mask pixels')
    preprocessing = preprocessing_settings(preprocess_masks, **preprocessing_parameters)
    path, source = consensus_source(dataset)
    metadata = dict(source, radius=radius, degree=degree, preprocessing=preprocessing,
                    mahotas_version=version('mahotas'))
    destination = zernike_path(dataset, preprocess_masks=preprocess_masks, write=True)
    require_new_outputs([destination])
    with open_linked_masks(path) as streams:
        masks = streams['preferred']
        count = masks.shape[0]
        report = Progress('Zernike features', count, logger=logger, callback=progress, cancelled=cancelled)
        with container_output(destination, overwrite=False) as output:
            create_zernike(output, dict(metadata, frame_count=count, image_shape=list(masks.image_shape)))
            for start in range(0, count, block_size):
                end = min(count, start + block_size)
                features, frames, identities = [], [], []
                for frame in range(start, end):
                    report.check_cancelled()
                    if masks.valid[frame]:
                        try:
                            labels, values = frame_zernike_features(masks[frame], radius, degree,
                                                                   preprocessing=preprocessing)
                        except ValueError as error:
                            raise ValueError(f'Frame {frame}: {error}') from error
                        features.append(values)
                        frames.append(np.full(len(labels), frame, dtype=np.int64))
                        identities.append(labels)
                if features:
                    append_zernike(output, np.concatenate(features), np.concatenate(frames),
                                   np.concatenate(identities), degree)
                report.update(end - start)
            if len(output['features']) < 4:
                raise ValueError('At least four valid animal masks are needed for standard UMAP')
            _, current_source = consensus_source(dataset)
            validate_shape_metadata(current_source, source)
            report.check_cancelled()
    report.finish(dataset.relative(destination))
    return destination


def shape_features_job(dataset, *, radius, degree=ANALYSIS_DEFAULTS['shape_features']['degree'],
                       z_score=ANALYSIS_DEFAULTS['shape_features']['z_score'],
                       preprocess_masks=ANALYSIS_DEFAULTS['shape_features']['preprocess_masks'],
                       progress=None, cancelled=None, **preprocessing_parameters):
    """Reuse matching descriptors and save raw or z-scored UMAP without overwrites."""
    # Fail before creating files if either optional analysis dependency is absent.
    versions = {name: version(name) for name in ('mahotas', 'umap-learn', 'scikit-learn')}
    callback = (None if progress is None else
                lambda completed, total: progress({'completed': completed, 'total': total}))
    preprocessing = preprocessing_settings(preprocess_masks, **preprocessing_parameters)
    _, source = consensus_source(dataset)
    expected = dict(source, degree=degree, radius=radius, preprocessing=preprocessing)
    features = zernike_path(dataset, preprocess_masks=preprocess_masks)
    if not features.is_file():
        features = extract_zernike(dataset, radius, degree=degree, preprocess_masks=preprocess_masks,
                                  progress=callback, cancelled=cancelled, **preprocessing_parameters)
    report = Progress('UMAP', 1, logger=logger, callback=callback, cancelled=cancelled)
    if progress is not None:
        progress({'completed': 0, 'total': 1})
    values, frames, identities, metadata = read_zernike(features, expected)
    expected = dict(expected, feature_id=metadata['feature_id'], z_score=z_score)
    destination = embedding_path(dataset, z_score=z_score, preprocess_masks=preprocess_masks, write=True)
    if destination.is_file():
        read_embedding(destination, expected)
        logger.info('Using saved UMAP: %s', dataset.relative(destination))
    else:
        require_new_outputs([destination])
        logger.info('Fitting standard UMAP to %s samples; z-score=%s', len(values), z_score)
        coordinates, mean, scale, parameters = umap_embedding(values, z_score=z_score)
        report.check_cancelled()
        _, current_source = consensus_source(dataset)
        validate_shape_metadata(current_source, source)
        write_embedding(destination, coordinates, mean, scale, frames, identities,
                        dict(expected, feature_count=values.shape[1], versions=versions,
                             umap_parameters=parameters))
    report.update(1)
    report.finish(dataset.relative(destination))
    return [str(features), str(destination)]
