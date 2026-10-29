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

import numpy as np

from bivosia.config import ANALYSIS_DEFAULTS


def embedding_features(features, *, z_score=ANALYSIS_DEFAULTS['shape_features']['z_score']):
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


def umap_embedding(features, *, z_score=ANALYSIS_DEFAULTS['shape_features']['z_score']):
    """Fit a two-dimensional UMAP using the library's unmodified defaults."""
    from umap import UMAP

    values, mean, scale = embedding_features(features, z_score=z_score)
    if len(values) < 4:
        raise ValueError('Standard two-dimensional UMAP requires at least four samples')
    if not np.any(values != values[0]):
        raise ValueError('UMAP requires variation between feature vectors')
    model = UMAP()
    coordinates = model.fit_transform(values)
    validate_embedding(coordinates, mean, scale, len(values), values.shape[1])
    return coordinates, mean, scale, model.get_params()


def validate_embedding(coordinates, mean, scale, samples, features):
    """Validate saved UMAP coordinates and their feature-scaling parameters."""
    if (coordinates.shape != (samples, 2) or mean.shape != (features,)
            or scale.shape != mean.shape or not np.isfinite(coordinates).all()
            or not np.isfinite(mean).all() or not np.isfinite(scale).all()
            or np.any(scale <= 0)):
        raise ValueError('Invalid UMAP coordinates or feature-scaling parameters')
