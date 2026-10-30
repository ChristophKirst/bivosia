"""Standard UMAP embeddings with optional column-wise z-scoring.

Examples
--------
>>> import numpy as np
>>> from bivosia.analysis.embedding import embedding_features
>>> values = np.array([[1., 7.], [3., 7.]])
>>> scaled, mean, scale = embedding_features(values, z_score=True)
>>> scaled.tolist()
[[-1.0, 0.0], [1.0, 0.0]]
>>> values.tolist()
[[1.0, 7.0], [3.0, 7.0]]
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import operator

import numpy as np

from bivosia.config import ANALYSIS_DEFAULTS


def embedding_features(features, *, z_score=ANALYSIS_DEFAULTS['embedding']['z_score']):
    """Copy features, optionally centering and scaling each column over samples.

    StandardScaler uses population variance (ddof=0). Constant columns have
    scale one and become zero after centering. The input is never changed.
    """
    values = np.array(features, dtype=np.float64, copy=True)
    if values.ndim != 2 or not all(values.shape) or not np.isfinite(values).all():
        raise ValueError('Expected a nonempty, finite samples-by-features matrix')
    if z_score:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        return scaler.fit_transform(values), scaler.mean_, scaler.scale_
    return values, np.zeros(values.shape[1]), np.ones(values.shape[1])


def embedding_dimensions(n_components):
    """Validate the requested number of UMAP dimensions.

    >>> from bivosia.analysis.embedding import embedding_dimensions
    >>> embedding_dimensions(3)
    3
    """
    dimensions = operator.index(n_components)
    if dimensions not in (2, 3):
        raise ValueError('Choose two or three UMAP dimensions')
    return dimensions


def umap_embedding(features, *, n_components=ANALYSIS_DEFAULTS['embedding']['n_components'],
                   z_score=ANALYSIS_DEFAULTS['embedding']['z_score']):
    """Fit a 2D or 3D UMAP, keeping the library's other defaults."""
    from umap import UMAP

    n_components = embedding_dimensions(n_components)
    values, mean, scale = embedding_features(features, z_score=z_score)
    if len(values) < n_components + 2:
        raise ValueError(f'{n_components}D UMAP requires at least {n_components + 2} samples')
    if not np.any(values != values[0]):
        raise ValueError('UMAP requires variation between feature vectors')
    model = UMAP(n_components=n_components)
    coordinates = model.fit_transform(values)
    validate_embedding(coordinates, mean, scale, len(values), values.shape[1], n_components=n_components)
    return coordinates, mean, scale, model.get_params()


def validate_embedding(coordinates, mean, scale, samples, features, *,
                       n_components=ANALYSIS_DEFAULTS['embedding']['n_components']):
    """Validate saved UMAP coordinates and their feature-scaling parameters."""
    n_components = embedding_dimensions(n_components)
    if (coordinates.shape != (samples, n_components) or mean.shape != (features,)
            or scale.shape != mean.shape or not np.isfinite(coordinates).all()
            or not np.isfinite(mean).all() or not np.isfinite(scale).all()
            or np.any(scale <= 0)):
        raise ValueError('Invalid UMAP coordinates or feature-scaling parameters')
