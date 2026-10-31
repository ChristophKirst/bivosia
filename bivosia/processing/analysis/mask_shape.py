"""Extract consensus-mask Zernike features and run standard UMAP.

Examples
--------
>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.processing.analysis.mask_shape import shape_features_job
>>> dataset = Dataset('/path/to/dataset')
>>> outputs = shape_features_job(dataset, radius=100, z_score=True)  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from importlib.metadata import version
import logging

import numpy as np

from bivosia.config import ANALYSIS_DEFAULTS, CONSENSUS_MASK_DEFAULTS, PROCESSING_DEFAULTS
from bivosia.analysis.embedding import umap_embedding
from bivosia.analysis.keyframes import output_shape
from bivosia.analysis.mask_features import frame_zernike_features, sampled_frames, validate_shape_metadata, zernike_indices
from bivosia.dataset.workflow.processing import require_new_outputs
from bivosia.dataset.io.containers import container_output
from bivosia.dataset.masks import ConsensusMaskReader, consensus_metadata
from bivosia.dataset.mask_features import (
    MaskShapeResults, append_zernike, create_zernike, read_embedding, read_zernike, write_embedding,
)
from bivosia.processing.log import Progress


logger = logging.getLogger(__name__)


def extract_zernike(dataset, radius, *, degree=ANALYSIS_DEFAULTS['mask_features']['zernike']['degree'],
                    downsample_factor=ANALYSIS_DEFAULTS['mask_features']['zernike']['downsample_factor'],
                    block_size=PROCESSING_DEFAULTS['analysis']['mask_shape']['block_size'],
                    preprocess_masks=ANALYSIS_DEFAULTS['mask_features']['zernike']['preprocess_masks'],
                    mask_source=CONSENSUS_MASK_DEFAULTS['source'],
                    start_frame=PROCESSING_DEFAULTS['analysis']['mask_shape']['start_frame'],
                    end_frame=PROCESSING_DEFAULTS['analysis']['mask_shape']['end_frame'],
                    frame_step=PROCESSING_DEFAULTS['analysis']['mask_shape']['frame_step'],
                    progress=None, cancelled=None, **preprocessing_parameters):
    """Stream present animals from valid consensus frames into a descriptor file."""
    if block_size < 1:
        raise ValueError('block_size must be positive')
    zernike_indices(degree)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Choose a positive Zernike radius in mask pixels')
    with ConsensusMaskReader(dataset, source=mask_source) as masks:
        results = MaskShapeResults(dataset, dict(radius=radius, degree=degree, downsample_factor=downsample_factor,
            preprocess_masks=preprocess_masks, mask_source=mask_source, start_frame=start_frame,
            end_frame=end_frame, frame_step=frame_step, **preprocessing_parameters), source=masks.metadata)
        destination = results.path('mask_zernike', write=True)
        require_new_outputs([destination])
        downsample_factor = results.factor
        metadata = dict(results.feature_expected, mahotas_version=version('mahotas'))
        count = masks.shape[0]
        selected_frames = sampled_frames(count, **results.sampling)
        report = Progress('Zernike features', len(selected_frames), logger=logger, callback=progress, cancelled=cancelled)
        with container_output(destination, overwrite=False) as output:
            create_zernike(output, dict(metadata, frame_count=count, source_image_shape=list(masks.image_shape),
                                       image_shape=list(output_shape(masks.image_shape, 1 / downsample_factor))))
            for start in range(0, len(selected_frames), block_size):
                batch = selected_frames[start:start + block_size]
                features, frames, identities = [], [], []
                for frame in batch:
                    report.check_cancelled()
                    if masks.valid[frame]:
                        try:
                            labels, values, excluded = frame_zernike_features(
                                masks[frame], radius, degree, downsample_factor=downsample_factor,
                                preprocessing=results.preprocessing)
                        except ValueError as error:
                            raise ValueError(f'Frame {frame}: {error}') from error
                        for identity, required_radius in excluded:
                            logger.warning('Frame %s, animal %s: excluded from shape analysis; '
                                           'Zernike radius %g is too small (requires %.6g pixels)',
                                           frame, identity, radius, required_radius)
                        features.append(values)
                        frames.append(np.full(len(labels), frame, dtype=np.int64))
                        identities.append(labels)
                if features:
                    append_zernike(output, np.concatenate(features), np.concatenate(frames),
                                   np.concatenate(identities), degree)
                report.update(len(batch))
            if len(output['features']) < 4:
                raise ValueError('At least four valid animal masks are needed for standard UMAP')
            current_source = consensus_metadata(dataset, source=mask_source)
            validate_shape_metadata(current_source, results.source)
            report.check_cancelled()
    report.finish(dataset.relative(destination))
    return destination


def embed_zernike(dataset, settings, *, progress=None, cancelled=None):
    """Fit or load UMAP from saved descriptors without extracting masks."""
    versions = {name: version(name) for name in ('umap-learn', 'scikit-learn')}
    results = MaskShapeResults(dataset, settings)
    features = results.path('mask_zernike')
    descriptors = read_zernike(features, results.feature_expected)
    versions['mahotas'] = descriptors.metadata['mahotas_version']
    expected = results.embedding_expected(descriptors.metadata)
    destination = results.path('mask_umap', write=True)
    report = Progress('UMAP', 1, logger=logger, callback=progress, cancelled=cancelled)
    if progress is not None:
        progress(0, 1)
    if destination.is_file():
        read_embedding(destination, expected)
        logger.info('Using saved UMAP: %s', dataset.relative(destination))
    else:
        require_new_outputs([destination])
        logger.info('Fitting %sD UMAP to %s samples; z-score=%s',
                    results.dimensions, len(descriptors.values), results.settings['z_score'])
        embedding = umap_embedding(descriptors.values, n_components=results.dimensions,
                                   z_score=results.settings['z_score'])
        report.check_cancelled()
        current_source = consensus_metadata(dataset, source=results.settings['mask_source'])
        validate_shape_metadata(current_source, results.source)
        write_embedding(destination, embedding, descriptors.frames, descriptors.identities,
                        dict(expected, feature_count=descriptors.values.shape[1], versions=versions))
    report.update(1)
    report.finish(dataset.relative(destination))
    return destination


def shape_features_job(dataset, *, radius, degree=ANALYSIS_DEFAULTS['mask_features']['zernike']['degree'],
                       downsample_factor=ANALYSIS_DEFAULTS['mask_features']['zernike']['downsample_factor'],
                       n_components=ANALYSIS_DEFAULTS['embedding']['n_components'],
                       z_score=ANALYSIS_DEFAULTS['embedding']['z_score'],
                       preprocess_masks=ANALYSIS_DEFAULTS['mask_features']['zernike']['preprocess_masks'],
                       mask_source=CONSENSUS_MASK_DEFAULTS['source'],
                       start_frame=PROCESSING_DEFAULTS['analysis']['mask_shape']['start_frame'],
                       end_frame=PROCESSING_DEFAULTS['analysis']['mask_shape']['end_frame'],
                       frame_step=PROCESSING_DEFAULTS['analysis']['mask_shape']['frame_step'],
                       progress=None, cancelled=None, **preprocessing_parameters):
    """Reuse matching descriptors and save raw or z-scored UMAP without overwrites."""
    # Fail before creating files if either optional analysis dependency is absent.
    for name in ('mahotas', 'umap-learn', 'scikit-learn'):
        version(name)
    results = MaskShapeResults(dataset, dict(radius=radius, degree=degree, downsample_factor=downsample_factor,
        n_components=n_components, z_score=z_score, preprocess_masks=preprocess_masks, mask_source=mask_source,
        start_frame=start_frame, end_frame=end_frame, frame_step=frame_step, **preprocessing_parameters))
    downsample_factor = results.factor
    callback = (None if progress is None else
                lambda completed, total: progress({'completed': completed, 'total': total}))
    features = results.path('mask_zernike')
    if not features.is_file():
        features = extract_zernike(dataset, radius, degree=degree, preprocess_masks=preprocess_masks, mask_source=mask_source,
                                  downsample_factor=downsample_factor,
                                  progress=callback, cancelled=cancelled, **results.sampling, **preprocessing_parameters)
    destination = embed_zernike(dataset, results.settings, progress=callback, cancelled=cancelled)
    return [str(features), str(destination)]
