"""Per-frame mask measures and forward/reverse comparisons.

Examples
--------
>>> from bivosia.analysis.measures import direction_difference
>>> direction_difference([0, 1], [0.2, 0.8], [1, 2], [0.3, 0.1])[1].tolist()
[0.5]
"""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import numpy as np

from bivosia.analysis.masks import missing_mask_counts


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


def direction_difference(forward_frames, forward_values, reverse_frames, reverse_values):
    """Return aligned frame indices and forward-minus-reverse measure values."""
    frames, forward, reverse = np.intersect1d(forward_frames, reverse_frames, return_indices=True)
    return frames, np.asarray(forward_values)[forward] - np.asarray(reverse_values)[reverse]


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


def recording_measures(matrices, evidence, forward, reverse, missing):
    """Return combined missing flags, margin quality and mean assigned IoU."""
    frame_count, n, _ = matrices.shape
    missing = missing | forward | reverse
    quality = np.array(evidence.quality, copy=True)
    quality[missing] = np.nan
    assigned = matrices[np.arange(frame_count)[:, None], np.arange(n), evidence.assignment]
    assigned_iou = np.mean(assigned, axis=1)
    assigned_iou[missing] = np.nan
    return missing, quality, assigned_iou
