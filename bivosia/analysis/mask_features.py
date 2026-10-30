"""Mask geometry, contour and Zernike descriptors, temporal features and summaries.

Examples
--------
>>> import numpy as np
>>> from bivosia.analysis.mask_features import frame_features, mask_areas, mask_centers, mask_presence
>>> labels = np.array([[0, 1], [2, 2]], dtype=np.uint8)
>>> mask_areas(labels, 2).tolist()
[1, 2]
>>> mask_centers(labels, 2)
((1.0, 0.0), (0.5, 1.0))
>>> mask_presence(labels, 3).tolist()
[True, True, False]
>>> values, previous, components, holes = frame_features(labels, 2, None)
>>> values.shape[0]
2

Zernike descriptors require the optional Mahotas dependency:

>>> from bivosia.analysis.mask_features import zernike_features
>>> mask = np.zeros((12, 12), dtype=bool)
>>> mask[3:9, 4:8] = True
>>> zernike_features(mask, radius=5, degree=2).shape
(4,)
>>> bool(np.allclose(zernike_features(mask, 5), zernike_features(np.rot90(mask), 5)))
True
"""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import math

import numpy as np
import cv2

from bivosia.config import ANALYSIS_DEFAULTS
from bivosia.analysis.masks import _mask_labels, downsample_mask, preprocess_mask, validate_downsample_factor


# Mask geometry and presence


def mask_areas(mask: np.ndarray, n: int) -> np.ndarray:
    """Count foreground pixels for raw object IDs 1..n, excluding background."""
    labels = _mask_labels(mask, n)
    return np.bincount(labels, minlength=n + 1)[1:n + 1].astype(np.int64, copy=False)


def mask_centers(masks, n):
    ys, xs = np.nonzero((masks > 0) & (masks <= n))
    labels = masks[ys, xs].astype(np.intp)
    counts = np.bincount(labels, minlength=n + 1)
    sum_x = np.bincount(labels, weights=xs, minlength=n + 1)
    sum_y = np.bincount(labels, weights=ys, minlength=n + 1)
    return tuple((float(sum_x[i] / counts[i]), float(sum_y[i] / counts[i])) if counts[i] else None
                 for i in range(1, n + 1))


def mask_presence(mask, n):
    """Return whether each object ID 1..n occurs in the mask."""
    return np.isin(np.arange(1, n + 1), np.unique(mask))


def missing_mask_counts(present: np.ndarray) -> np.ndarray:
    """Count zeros along the identity axis of Boolean presence or pixel areas."""
    present = np.asarray(present, dtype=bool)
    return present.shape[-1] - np.count_nonzero(present, axis=-1)


# Contour and temporal features


FEATURE_NAMES = (
    "area_px",
    "centroid_x",
    "centroid_y",
    "external_perimeter_px",
    "component_count",
    "hole_count",
    "largest_component_area_px",
    "total_hole_area_px",
    "temporal_intersection_px",
    "shape_change_i1",
    "convex_hull_perimeter_excess",
    "fft_boundary_high_frequency_energy",
    "multiscale_perimeter_excess",
    "normalized_bending_energy",
    "hu_log_1",
    "hu_log_2",
    "hu_log_3",
    "hu_log_4",
    "hu_log_5",
    "hu_log_6",
    "hu_log_7",
)


FFT_FEATURE_NAMES = (
    "fft_boundary_high_frequency_energy",
    "multiscale_perimeter_excess",
    "normalized_bending_energy",
)


HU_FEATURE_NAMES = (
    'shape_change_i1',
    *(f'hu_log_{index}' for index in range(1, 8)),
)


COMPONENT_DTYPE = np.dtype(
    [
        ("area_px", "<u4"),
        ("centroid_x", "<f4"),
        ("centroid_y", "<f4"),
        ("external_perimeter_px", "<f4"),
        ("bbox_x", "<u2"),
        ("bbox_y", "<u2"),
        ("bbox_width", "<u2"),
        ("bbox_height", "<u2"),
    ]
)


HOLE_DTYPE = np.dtype(
    [
        ("area_px", "<f4"),
        ("perimeter_px", "<f4"),
        ("parent_component", "<i4"),
    ]
)


def feature_names(
    *,
    include_convex_hull: bool = ANALYSIS_DEFAULTS['mask_features']['include_convex_hull'],
    include_hu_features: bool = ANALYSIS_DEFAULTS['mask_features']['include_hu_features'],
    include_fft_features: bool = ANALYSIS_DEFAULTS['mask_features']['include_fft_features'],
) -> tuple[str, ...]:
    """Return the ordered output columns for the enabled feature groups.

    >>> from bivosia.analysis.mask_features import feature_names
    >>> 'hu_log_1' in feature_names(include_hu_features=False)
    False
    >>> 'shape_change_i1' in feature_names(include_hu_features=True)
    True
    >>> 'fft_boundary_high_frequency_energy' in feature_names(include_fft_features=False)
    False
    """
    return tuple(
        name for name in FEATURE_NAMES
        if (include_convex_hull or name != 'convex_hull_perimeter_excess')
        and (include_hu_features or name not in HU_FEATURE_NAMES)
        and (include_fft_features or name not in FFT_FEATURE_NAMES)
    )


def _resample_closed_contour(
    contour: np.ndarray | None,
    sample_count: int = ANALYSIS_DEFAULTS['mask_features']['fft_features']['sample_count'],
) -> np.ndarray | None:
    """Return equally arc-length-spaced complex samples of a closed contour."""
    if contour is None:
        return None
    points = np.asarray(contour[:, 0, :], dtype=np.float64)
    if len(points) < 3:
        return None
    closed = np.vstack((points, points[0]))
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    keep = np.r_[True, lengths > 0]
    closed = closed[keep]
    if len(closed) < 4:
        return None
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
    perimeter = float(cumulative[-1])
    if not math.isfinite(perimeter) or perimeter <= 0:
        return None
    targets = np.arange(sample_count, dtype=np.float64) * perimeter / sample_count
    x = np.interp(targets, cumulative, closed[:, 0])
    y = np.interp(targets, cumulative, closed[:, 1])
    return x + 1j * y


def _spectral_boundary_features(contour: np.ndarray | None) -> tuple[float, float, float]:
    """Calculate scale-invariant FFT, multiscale-perimeter, and bending roughness."""
    parameters = ANALYSIS_DEFAULTS['mask_features']['fft_features']
    samples = _resample_closed_contour(contour, parameters['sample_count'])
    if samples is None:
        return math.nan, math.nan, math.nan
    samples = samples - np.mean(samples)
    spectrum = np.fft.fft(samples)
    frequencies = np.abs(np.fft.fftfreq(len(samples)) * len(samples))
    power = np.abs(spectrum) ** 2
    nonconstant = frequencies > 0
    total_power = float(np.sum(power[nonconstant]))
    if not math.isfinite(total_power) or total_power <= 0:
        return math.nan, math.nan, math.nan

    high_frequency_energy = float(
        np.sum(power[frequencies > parameters['low_frequency_cutoff']]) / total_power
    )
    raw_perimeter = float(np.sum(np.abs(np.roll(samples, -1) - samples)))
    perimeter_excesses = []
    bending_samples = None
    for cutoff in parameters['perimeter_cutoffs']:
        filtered = spectrum.copy()
        filtered[frequencies > cutoff] = 0
        reconstructed = np.fft.ifft(filtered)
        smooth_perimeter = float(
            np.sum(np.abs(np.roll(reconstructed, -1) - reconstructed))
        )
        if smooth_perimeter > 0:
            perimeter_excesses.append(max(0.0, raw_perimeter / smooth_perimeter - 1.0))
        if cutoff == parameters['bending_cutoff']:
            bending_samples = reconstructed
    multiscale_excess = (
        float(np.mean(perimeter_excesses)) if perimeter_excesses else math.nan
    )

    bending_energy = math.nan
    if bending_samples is not None:
        segments = np.roll(bending_samples, -1) - bending_samples
        segment_lengths = np.abs(segments)
        valid = segment_lengths > 1e-12
        if np.all(valid):
            angles = np.angle(segments)
            turns = np.angle(np.exp(1j * (angles - np.roll(angles, 1))))
            local_lengths = 0.5 * (segment_lengths + np.roll(segment_lengths, 1))
            length = float(np.sum(segment_lengths))
            raw_bending = length * float(np.sum(turns * turns / local_lengths))
            # A circle has scale-invariant bending energy 4*pi^2.
            bending_energy = max(0.0, raw_bending / (4.0 * math.pi * math.pi) - 1.0)
    return high_frequency_energy, multiscale_excess, bending_energy


def _signed_log_hu(contour: np.ndarray | None) -> np.ndarray:
    result = np.full(7, np.nan, dtype=np.float64)
    if contour is None:
        return result
    hu = cv2.HuMoments(cv2.moments(contour)).reshape(-1)
    nonzero = hu != 0
    result[nonzero] = np.sign(hu[nonzero]) * np.log10(np.abs(hu[nonzero]))
    return result


def _one_identity(
    mask_bool: np.ndarray,
    include_convex_hull: bool = ANALYSIS_DEFAULTS['mask_features']['include_convex_hull'],
    include_hu_features: bool = ANALYSIS_DEFAULTS['mask_features']['include_hu_features'],
    include_fft_features: bool = ANALYSIS_DEFAULTS['mask_features']['include_fft_features'],
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Return dense raw values plus every foreground-component and hole record."""
    mask = mask_bool.astype(np.uint8, copy=False)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    component_count = max(0, count - 1)
    if component_count == 0:
        state = {
            "area_px": 0.0,
            "mask": None,
            "contour": None,
        }
        return state, np.empty(0, COMPONENT_DTYPE), np.empty(0, HOLE_DTYPE)

    component_records = np.zeros(component_count, dtype=COMPONENT_DTYPE)
    component_records["area_px"] = stats[1:, cv2.CC_STAT_AREA]
    component_records["centroid_x"] = centroids[1:, 0]
    component_records["centroid_y"] = centroids[1:, 1]
    component_records["bbox_x"] = stats[1:, cv2.CC_STAT_LEFT]
    component_records["bbox_y"] = stats[1:, cv2.CC_STAT_TOP]
    component_records["bbox_width"] = stats[1:, cv2.CC_STAT_WIDTH]
    component_records["bbox_height"] = stats[1:, cv2.CC_STAT_HEIGHT]

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hole_records: list[tuple[float, float, int]] = []
    contour_for_component: dict[int, np.ndarray] = {}
    if hierarchy is not None:
        relations = hierarchy[0]
        for contour, relation in zip(contours, relations):
            perimeter = float(cv2.arcLength(contour, True))
            if relation[3] < 0:
                x, y = contour[0, 0]
                component_label = int(labels[int(y), int(x)])
                if component_label < 1:
                    continue
                local_index = component_label - 1
                component_records["external_perimeter_px"][local_index] += perimeter
                current = contour_for_component.get(local_index)
                if current is None or cv2.contourArea(contour) > cv2.contourArea(current):
                    contour_for_component[local_index] = contour
            else:
                parent_contour = contours[int(relation[3])]
                x, y = parent_contour[0, 0]
                component_label = int(labels[int(y), int(x)])
                hole_records.append(
                    (
                        float(cv2.contourArea(contour)),
                        perimeter,
                        component_label - 1,
                    )
                )

    largest_component = int(np.argmax(component_records["area_px"]))
    largest_contour = contour_for_component.get(largest_component)
    if include_convex_hull:
        largest_perimeter = (
            float(cv2.arcLength(largest_contour, True))
            if largest_contour is not None else math.nan
        )
        hull_perimeter = math.nan
        if largest_contour is not None and len(largest_contour) >= 3:
            hull = cv2.convexHull(largest_contour)
            hull_perimeter = float(cv2.arcLength(hull, True))
        hull_excess = (
            max(0.0, largest_perimeter / hull_perimeter - 1.0)
            if hull_perimeter > 0 else math.nan
        )
    else:
        hull_excess = math.nan
    if include_fft_features:
        fft_energy, multiscale_excess, bending_energy = _spectral_boundary_features(largest_contour)
    else:
        fft_energy = multiscale_excess = bending_energy = math.nan
    areas = component_records["area_px"].astype(np.float64)
    total_area = float(np.sum(areas))
    centroid_x = float(np.sum(areas * component_records["centroid_x"]) / total_area)
    centroid_y = float(np.sum(areas * component_records["centroid_y"]) / total_area)
    holes = np.asarray(hole_records, dtype=HOLE_DTYPE)
    state = {
        "area_px": total_area,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "external_perimeter_px": float(
            np.sum(component_records["external_perimeter_px"], dtype=np.float64)
        ),
        "component_count": float(component_count),
        "hole_count": float(len(holes)),
        "largest_component_area_px": float(component_records["area_px"][largest_component]),
        "total_hole_area_px": float(np.sum(holes["area_px"], dtype=np.float64)),
        "convex_hull_perimeter_excess": hull_excess,
        "fft_boundary_high_frequency_energy": fft_energy,
        "multiscale_perimeter_excess": multiscale_excess,
        "normalized_bending_energy": bending_energy,
        "mask": mask_bool,
        "contour": largest_contour,
    }
    if include_hu_features:
        state['hu_log'] = _signed_log_hu(largest_contour)
    return state, component_records, holes


def frame_features(
    labels: np.ndarray,
    n_animals: int,
    previous: list[dict] | None,
    include_convex_hull: bool = ANALYSIS_DEFAULTS['mask_features']['include_convex_hull'],
    include_hu_features: bool = ANALYSIS_DEFAULTS['mask_features']['include_hu_features'],
    include_fft_features: bool = ANALYSIS_DEFAULTS['mask_features']['include_fft_features'],
) -> tuple[np.ndarray, list[dict], list[np.ndarray], list[np.ndarray]]:
    """Return dense raw values, private state, and ragged geometry records.

    Hu features include seven signed-log Hu moments and the Hu-based I1
    comparison of consecutive largest-component contours.
    Convex-hull excess and the FFT features (energy, multiscale perimeter,
    and bending energy) are controlled independently.
    """
    extracted = [
        _one_identity(labels == animal, include_convex_hull, include_hu_features, include_fft_features)
        for animal in range(1, n_animals + 1)
    ]
    state = [item[0] for item in extracted]
    components = [item[1] for item in extracted]
    holes = [item[2] for item in extracted]
    names = feature_names(include_convex_hull=include_convex_hull,
                          include_hu_features=include_hu_features,
                          include_fft_features=include_fft_features)
    output = np.full((n_animals, len(names)), np.nan, dtype=np.float64)
    index = {name: position for position, name in enumerate(names)}
    for animal, current in enumerate(state):
        for name in (
            "area_px",
            "centroid_x",
            "centroid_y",
            "external_perimeter_px",
            "component_count",
            "hole_count",
            "largest_component_area_px",
            "total_hole_area_px",
            "convex_hull_perimeter_excess",
            "fft_boundary_high_frequency_energy",
            "multiscale_perimeter_excess",
            "normalized_bending_energy",
        ):
            if name in current and name in index:
                output[animal, index[name]] = current[name]
        if "hu_log" in current:
            output[animal, index["hu_log_1"] : index["hu_log_7"] + 1] = current["hu_log"]
        if previous is None or current["mask"] is None or previous[animal]["mask"] is None:
            continue
        old = previous[animal]
        output[animal, index["temporal_intersection_px"]] = float(
            np.count_nonzero(current["mask"] & old["mask"])
        )
        if include_hu_features and current["contour"] is not None and old["contour"] is not None:
            value = float(
                cv2.matchShapes(
                    old["contour"], current["contour"], cv2.CONTOURS_MATCH_I1, 0
                )
            )
            # OpenCV returns DBL_MAX for degenerate contour comparisons.
            if math.isfinite(value) and value < np.finfo(np.float64).max:
                output[animal, index["shape_change_i1"]] = value
    return output, state, components, holes


# Zernike descriptors


class ZernikeRadiusError(ValueError):
    """An animal mask extends beyond the requested Zernike disk."""

    def __init__(self, radius, required_radius):
        self.required_radius = required_radius
        super().__init__(f'Zernike radius {radius:g} is too small; this mask needs at least {required_radius:.6g} pixels')


def zernike_indices(degree):
    """Return Mahotas' ordered (radial degree, angular order) pairs."""
    if not isinstance(degree, (int, np.integer)) or degree < 0:
        raise ValueError('Zernike degree must be a nonnegative integer')
    return np.array([(n, m) for n in range(degree + 1)
                     for m in range(n + 1) if (n - m) % 2 == 0], dtype=np.int32)


def zernike_features(mask, radius, degree=ANALYSIS_DEFAULTS['mask_features']['zernike']['degree']):
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
        raise ZernikeRadiusError(radius, extent)
    y0, x0, y1, x1 = y.min(), x.min(), y.max() + 1, x.max() + 1
    crop = mask[y0:y1, x0:x1]
    return zernike_moments(crop, radius, degree=degree,
                           cm=(center[0] - y0, center[1] - x0))


def prepare_shape_mask(mask, *, downsample_factor=ANALYSIS_DEFAULTS['mask_features']['zernike']['downsample_factor'],
                       preprocessing=None):
    """Downsample an animal mask, then preprocess it in the resulting pixel grid."""
    mask = downsample_mask(mask, downsample_factor)
    return preprocess_mask(mask, **preprocessing) if preprocessing is not None else mask


def frame_zernike_features(labels, radius, degree=ANALYSIS_DEFAULTS['mask_features']['zernike']['degree'], *,
                           downsample_factor=ANALYSIS_DEFAULTS['mask_features']['zernike']['downsample_factor'],
                           preprocessing=None):
    """Return identities, descriptors and radius exclusions for one frame.

    Downsampling precedes preprocessing. Radius, distances, areas and opening
    use the downsampled pixel grid. Animals with no remaining pixels are omitted.
    Masks outside the radius are excluded individually; the third return value
    lists their (identity, required radius) pairs. Other calculation errors propagate.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError('Expected a nonnegative integer label image')
    downsample_factor = validate_downsample_factor(downsample_factor)
    identities = np.unique(labels)
    identities = identities[identities > 0]
    values = np.empty((len(identities), len(zernike_indices(degree))), dtype=np.float64)
    present = np.ones(len(identities), dtype=bool)
    excluded = []
    for row, identity in enumerate(identities):
        mask = prepare_shape_mask(labels == identity, downsample_factor=downsample_factor,
                                  preprocessing=preprocessing)
        if not mask.any():
            present[row] = False
            continue
        try:
            values[row] = zernike_features(mask, radius, degree)
        except ZernikeRadiusError as error:
            present[row] = False
            excluded.append((int(identity), error.required_radius))
    return identities[present], values[present], excluded


# Frame summaries


MEASURE_NAMES = (
    "missing_masks",
    "max_centroid_displacement_px",
    "max_temporal_nonoverlap",
    "max_symmetric_relative_area_change",
    "max_absolute_log_area_ratio",
    "max_symmetric_relative_circumference_change",
    "max_component_count",
    "max_hole_count",
    "max_outside_largest_component_fraction",
    "max_fragmentation_index",
    "max_shape_change_i1",
    "max_isoperimetric_deficit",
    "max_convex_hull_perimeter_excess",
    "max_fft_boundary_high_frequency_energy",
    "max_multiscale_perimeter_excess",
    "max_normalized_bending_energy",
)


def _row_max(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    maximum = np.max(np.where(finite, values, -np.inf), axis=1)
    maximum[~np.any(finite, axis=1)] = np.nan
    return maximum.astype(np.float32, copy=False)


def mask_measures(source, feature_names):
    """Return per-pass, per-frame measures and their names from raw features."""
    names = list(feature_names)
    index = {name: position for position, name in enumerate(names)}
    required = {
        "area_px",
        "centroid_x",
        "centroid_y",
        "external_perimeter_px",
        "component_count",
        "hole_count",
        "largest_component_area_px",
        "temporal_intersection_px",
    }
    missing = sorted(required - set(index))
    if missing:
        raise ValueError(f"Raw feature array is missing fields: {', '.join(missing)}")
    source = np.asarray(source)
    if source.ndim != 4 or source.shape[0] != 2 or source.shape[-1] != len(names):
        raise ValueError("Expected raw feature shape (2, frames, animals, features)")
    optional_sources = {
        "max_shape_change_i1": "shape_change_i1",
        "max_convex_hull_perimeter_excess": "convex_hull_perimeter_excess",
        "max_fft_boundary_high_frequency_energy": "fft_boundary_high_frequency_energy",
        "max_multiscale_perimeter_excess": "multiscale_perimeter_excess",
        "max_normalized_bending_energy": "normalized_bending_energy",
    }
    measure_names = tuple(
        name
        for name in MEASURE_NAMES
        if name not in optional_sources or optional_sources[name] in index
    )
    destination = np.full((2, source.shape[1], len(measure_names)), np.nan, dtype=np.float32)
    measure_index = {name: position for position, name in enumerate(measure_names)}

    for pass_index in range(2):
        area = np.asarray(source[pass_index, :, :, index["area_px"]])
        present = np.isfinite(area) & (area > 0)
        destination[pass_index, :, measure_index["missing_masks"]] = missing_mask_counts(present)

        previous_present = np.zeros_like(present)
        previous_present[1:] = present[:-1]
        transition_valid = present & previous_present

        centroid_x = np.asarray(source[pass_index, :, :, index["centroid_x"]])
        centroid_y = np.asarray(source[pass_index, :, :, index["centroid_y"]])
        displacement = np.full_like(area, np.nan, dtype=np.float64)
        displacement[1:] = np.hypot(
            centroid_x[1:] - centroid_x[:-1],
            centroid_y[1:] - centroid_y[:-1],
        )
        displacement[~transition_valid] = np.nan
        destination[
            pass_index, :, measure_index["max_centroid_displacement_px"]
        ] = _row_max(displacement)

        intersection = np.asarray(
            source[pass_index, :, :, index["temporal_intersection_px"]]
        )
        previous_area = np.full_like(area, np.nan, dtype=np.float64)
        previous_area[1:] = area[:-1]
        union = area + previous_area - intersection
        nonoverlap = np.divide(
            union - intersection,
            union,
            out=np.full_like(area, np.nan, dtype=np.float64),
            where=transition_valid & np.isfinite(intersection) & (union > 0),
        )
        destination[
            pass_index, :, measure_index["max_temporal_nonoverlap"]
        ] = _row_max(nonoverlap)

        area_change = np.divide(
            2.0 * np.abs(area - previous_area),
            area + previous_area,
            out=np.full_like(area, np.nan, dtype=np.float64),
            where=transition_valid & ((area + previous_area) > 0),
        )
        destination[
            pass_index, :, measure_index["max_symmetric_relative_area_change"]
        ] = _row_max(area_change)

        absolute_log_area_ratio = np.full_like(area, np.nan, dtype=np.float64)
        absolute_log_area_ratio[transition_valid] = np.abs(
            np.log(area[transition_valid]) - np.log(previous_area[transition_valid])
        )
        destination[
            pass_index, :, measure_index["max_absolute_log_area_ratio"]
        ] = _row_max(absolute_log_area_ratio)

        perimeter = np.asarray(
            source[pass_index, :, :, index["external_perimeter_px"]]
        )
        previous_perimeter = np.full_like(perimeter, np.nan, dtype=np.float64)
        previous_perimeter[1:] = perimeter[:-1]
        perimeter_sum = perimeter + previous_perimeter
        perimeter_change = np.full_like(perimeter, np.nan, dtype=np.float64)
        regular = transition_valid & (perimeter_sum > 0)
        perimeter_change[regular] = (
            2.0
            * np.abs(perimeter[regular] - previous_perimeter[regular])
            / perimeter_sum[regular]
        )
        both_zero = transition_valid & (perimeter_sum == 0)
        perimeter_change[both_zero] = 0.0
        destination[
            pass_index,
            :,
            measure_index["max_symmetric_relative_circumference_change"],
        ] = _row_max(perimeter_change)

        component_count = np.asarray(
            source[pass_index, :, :, index["component_count"]]
        ).copy()
        component_count[~present] = np.nan
        destination[
            pass_index, :, measure_index["max_component_count"]
        ] = _row_max(component_count)

        hole_count = np.asarray(source[pass_index, :, :, index["hole_count"]]).copy()
        hole_count[~present] = np.nan
        destination[pass_index, :, measure_index["max_hole_count"]] = _row_max(
            hole_count
        )

        largest = np.asarray(
            source[pass_index, :, :, index["largest_component_area_px"]]
        )
        outside_largest = np.divide(
            area - largest,
            area,
            out=np.full_like(area, np.nan, dtype=np.float64),
            where=present,
        )
        destination[
            pass_index,
            :,
            measure_index["max_outside_largest_component_fraction"],
        ] = _row_max(outside_largest)

        # Count all 8-connected components, including single-pixel fragments.
        fragmentation = (
            np.maximum(0.0, component_count - 1.0)
            + hole_count
            + outside_largest
        )
        destination[
            pass_index, :, measure_index["max_fragmentation_index"]
        ] = _row_max(fragmentation)

        if "shape_change_i1" in index:
            shape = np.asarray(source[pass_index, :, :, index["shape_change_i1"]]).copy()
            shape[shape >= np.finfo(np.float64).max] = np.nan
            shape[~transition_valid] = np.nan
            destination[pass_index, :, measure_index["max_shape_change_i1"]] = _row_max(
                shape
            )

        isoperimetric_deficit = np.divide(
            perimeter * perimeter,
            4.0 * np.pi * area,
            out=np.full_like(area, np.nan, dtype=np.float64),
            where=present,
        ) - 1.0
        isoperimetric_deficit[isoperimetric_deficit < 0] = 0.0
        destination[
            pass_index, :, measure_index["max_isoperimetric_deficit"]
        ] = _row_max(isoperimetric_deficit)

        for raw_name, measure_name in (
            ("convex_hull_perimeter_excess", "max_convex_hull_perimeter_excess"),
            (
                "fft_boundary_high_frequency_energy",
                "max_fft_boundary_high_frequency_energy",
            ),
            ("multiscale_perimeter_excess", "max_multiscale_perimeter_excess"),
            ("normalized_bending_energy", "max_normalized_bending_energy"),
        ):
            if raw_name not in index:
                continue
            values = np.asarray(source[pass_index, :, :, index[raw_name]]).copy()
            values[~present] = np.nan
            destination[pass_index, :, measure_index[measure_name]] = _row_max(values)
    return destination, measure_names


# Sample selection and feature validation


def frame_sampling(start_frame, end_frame, frame_step):
    """Validate an inclusive frame range; -1 denotes the recording's last frame."""
    values = (start_frame, end_frame, frame_step)
    if any(not isinstance(value, (int, np.integer)) for value in values):
        raise ValueError('Frame selection requires integer values')
    if start_frame < 0 or end_frame < -1 or frame_step < 1 or (end_frame != -1 and end_frame < start_frame):
        raise ValueError('Choose a valid start/end frame and a positive frame interval')
    return dict(zip(('start_frame', 'end_frame', 'frame_step'), map(int, values)))


def sampled_frames(frame_count, *, start_frame, end_frame, frame_step):
    """Select source frames without changing their indices.

    >>> from bivosia.analysis.mask_features import sampled_frames
    >>> list(sampled_frames(10, start_frame=2, end_frame=8, frame_step=3))
    [2, 5, 8]
    """
    frame_sampling(start_frame, end_frame, frame_step)
    end = frame_count - 1 if end_frame == -1 else end_frame
    if not 0 <= start_frame <= end < frame_count:
        raise ValueError('Selected frame range is outside the recording')
    return range(start_frame, end + 1, frame_step)


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
