"""Mask-dependent raw geometry and temporal feature calculations.

Examples
--------
>>> from bivosia.analysis.features import frame_features
>>> import numpy as np
>>> labels = np.array([[0, 1], [2, 2]], dtype=np.uint8)
>>> values, previous, components, holes = frame_features(labels, 2, None)
>>> values.shape[0]
2
"""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import math

import numpy as np
import cv2

from bivosia.config import ANALYSIS_DEFAULTS


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


def feature_names(
    *,
    include_convex_hull: bool = ANALYSIS_DEFAULTS['features']['include_convex_hull'],
    include_hu_features: bool = ANALYSIS_DEFAULTS['features']['include_hu_features'],
    include_fft_features: bool = ANALYSIS_DEFAULTS['features']['include_fft_features'],
) -> tuple[str, ...]:
    """Return the ordered output columns for the enabled feature groups.

    >>> from bivosia.analysis.features import feature_names
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
    sample_count: int = ANALYSIS_DEFAULTS['features']['fft_features']['sample_count'],
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
    parameters = ANALYSIS_DEFAULTS['features']['fft_features']
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
    include_convex_hull: bool = ANALYSIS_DEFAULTS['features']['include_convex_hull'],
    include_hu_features: bool = ANALYSIS_DEFAULTS['features']['include_hu_features'],
    include_fft_features: bool = ANALYSIS_DEFAULTS['features']['include_fft_features'],
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
    include_convex_hull: bool = ANALYSIS_DEFAULTS['features']['include_convex_hull'],
    include_hu_features: bool = ANALYSIS_DEFAULTS['features']['include_hu_features'],
    include_fft_features: bool = ANALYSIS_DEFAULTS['features']['include_fft_features'],
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
