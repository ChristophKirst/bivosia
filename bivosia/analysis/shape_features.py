"""Centered Zernike descriptors for individual consensus masks.

Examples
--------
>>> import numpy as np
>>> from bivosia.analysis.shape_features import zernike_features
>>> mask = np.zeros((12, 12), dtype=bool)
>>> mask[3:9, 4:8] = True
>>> zernike_features(mask, radius=5, degree=2).shape
(4,)
>>> bool(np.allclose(zernike_features(mask, 5), zernike_features(np.rot90(mask), 5)))
True
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import numpy as np

from bivosia.config import ANALYSIS_DEFAULTS
from bivosia.analysis.mask_preprocessing import preprocess_mask
from bivosia.analysis.revision import revision_fingerprint


def zernike_indices(degree):
    """Return Mahotas' ordered (radial degree, angular order) pairs."""
    if not isinstance(degree, (int, np.integer)) or degree < 0:
        raise ValueError('Zernike degree must be a nonnegative integer')
    return np.array([(n, m) for n in range(degree + 1)
                     for m in range(n + 1) if (n - m) % 2 == 0], dtype=np.int32)


def zernike_features(mask, radius, degree=ANALYSIS_DEFAULTS['shape_features']['degree']):
    """Return absolute Mahotas moments centered at the foreground centroid.

    Use one fixed radius in mask pixels for every sample. Mahotas normalizes
    foreground weights by their sum; coordinates retain their scale relative
    to the supplied radius. No per-mask spatial resizing is performed.
    Reject a radius that would exclude any foreground pixel.
    """
    from mahotas.features import zernike_moments

    zernike_indices(degree)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Choose a positive Zernike radius in mask pixels')
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != np.bool_ or not mask.any():
        raise ValueError('Expected a nonempty two-dimensional boolean mask')
    y, x = np.nonzero(mask)
    center = (float(y.mean()), float(x.mean()))
    extent = float(np.hypot(y - center[0], x - center[1]).max())
    if extent > radius:
        raise ValueError(f'Zernike radius {radius:g} is too small; this mask needs at least {extent:.6g} pixels')
    y0, x0, y1, x1 = y.min(), x.min(), y.max() + 1, x.max() + 1
    crop = mask[y0:y1, x0:x1]
    return zernike_moments(crop, radius, degree=degree,
                           cm=(center[0] - y0, center[1] - x0))


def frame_zernike_features(labels, radius, degree=ANALYSIS_DEFAULTS['shape_features']['degree'], *,
                           preprocessing=None):
    """Return positive identities and descriptors, optionally preprocessing each body.

    Animals with no remaining foreground after preprocessing are omitted.
    Preprocessing settings use input-mask pixels and never resize the image.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError('Expected a nonnegative integer label image')
    identities = np.unique(labels)
    identities = identities[identities > 0]
    values = np.empty((len(identities), len(zernike_indices(degree))), dtype=np.float64)
    present = np.ones(len(identities), dtype=bool)
    for row, identity in enumerate(identities):
        mask = labels == identity
        if preprocessing is not None:
            mask = preprocess_mask(mask, **preprocessing)
        if not mask.any():
            present[row] = False
            continue
        values[row] = zernike_features(mask, radius, degree)
    return identities[present], values[present]


def validate_consensus_export(annotation, fingerprint):
    """Require an export to match the current saved annotation."""
    current = revision_fingerprint(annotation['header'], annotation['zones'],
                                   annotation['links'], annotation.get('link_frames', []))
    if annotation['export']['fingerprint'] != current or fingerprint != current:
        raise ValueError('Consensus masks do not match the current annotation; export again')


def validate_shape_metadata(saved, expected):
    """Require saved features and embeddings to match their source and settings."""
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError('Saved shape analysis uses different masks or settings; preserve or move it before recalculating')


def validate_zernike_data(values, frames, identities, degree):
    """Validate descriptor dimensions, sample indices and finite moment values."""
    validate_sample_indices(frames, identities)
    if (values.ndim != 2 or values.shape[1] != len(zernike_indices(degree))
            or frames.shape != (len(values),)
            or not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError('Invalid Zernike feature vectors or sample indices')


def validate_sample_indices(frames, identities):
    """Require unique, ordered frame and positive global-identity pairs."""
    if (frames.ndim != 1 or identities.shape != frames.shape
            or not np.issubdtype(frames.dtype, np.integer)
            or not np.issubdtype(identities.dtype, np.integer)
            or np.any(frames < 0) or np.any(identities <= 0)):
        raise ValueError('Invalid frame or identity indices')
    if len(frames) > 1 and np.any((frames[1:] < frames[:-1]) |
            ((frames[1:] == frames[:-1]) & (identities[1:] <= identities[:-1]))):
        raise ValueError('Samples must have unique, ordered frame and identity indices')
