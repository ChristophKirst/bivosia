"""Shared viewer colors, labels and display defaults."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import numpy as np


METRIC_COLORS = {
    "missing_masks": "#ff6b6b",
    "mean_mask_area_px": "#55bde8",
    "mean_centroid_displacement_px": "#f29d49",
    "mean_temporal_iou": "#62c47a",
    "mean_relative_area_change": "#d9bd4a",
    "mean_fragmentation_index": "#d477d6",
    "mean_hole_count": "#9787e5",
    "max_centroid_displacement_px": "#f29d49",
    "max_temporal_nonoverlap": "#62c47a",
    "max_symmetric_relative_area_change": "#d9bd4a",
    "max_absolute_log_area_ratio": "#b6c957",
    "max_symmetric_relative_circumference_change": "#55bde8",
    "max_component_count": "#d477d6",
    "max_hole_count": "#9787e5",
    "max_outside_largest_component_fraction": "#e879a8",
    "max_fragmentation_index": "#d477d6",
    "max_shape_change_i1": "#9ad0f5",
    "max_isoperimetric_deficit": "#f5b971",
    "max_convex_hull_perimeter_excess": "#f08a8a",
    "max_fft_boundary_high_frequency_energy": "#7bdff2",
    "max_multiscale_perimeter_excess": "#b2f7a6",
    "max_normalized_bending_energy": "#cdb4db",
    "mean_raw_assigned_iou": "#4fd1c5",
    "zod_local_quality": "#f2df5c",
}

HAND_BASE_COLORS = {
    "FORWARD": "#8bd17c",
    "REVERSE": "#2f8f5b",
    "UNKNOWN": "#9b78cf",
}
MISSING_COLORS = {
    "FORWARD": "#ff8994",
    "REVERSE": "#d84a5c",
    "BOTH": "#9b233c",
    "UNKNOWN": "#b37c86",
}
HAND_SWAP_COLOR = "#143f70"
HAND_SWAP_COLORS = {
    "FORWARD": "#8fb8ef",
    "REVERSE": "#4775bf",
    "BOTH": HAND_SWAP_COLOR,
}
HAND_UNKNOWN_COLOR = '#8b929c'
HAND_BACKGROUND_COLOR = '#252a33'
SWITCH_COLOR = "#5ca9e6"
KEYPOINT_SWAP_BORDER_COLOR = "#143f70"

REGION_COLORS = {
    "Combined seeds": "#ffd43b",
    "Missing forward only": MISSING_COLORS["FORWARD"],
    "Missing reverse only": MISSING_COLORS["REVERSE"],
    "Missing both": MISSING_COLORS["BOTH"],
    "Missing direction unknown": MISSING_COLORS["UNKNOWN"],
    "Switch frames": SWITCH_COLOR,
    "Missing mask + switch": "#9655a7",
    "Auto zones": "#e59b45",
    "Tie": "#ffd43b",
    "Swap": SWITCH_COLOR,
    "Swap over invalid zone": "#c5b1e5",
    "Partial resolved swap": "#28679b",
    "Partial swap over invalid zone": HAND_BASE_COLORS['UNKNOWN'],
    "No overlap": MISSING_COLORS['BOTH'],
    "Unassigned": "#9655a7",
    "Tie + unassigned": "#cba56e",
    "Missing skeleton": "#8b929c",
}

QUALITY_METRIC_LABELS = {
    "mean_raw_assigned_iou": "Raw assigned mIoU",
    "zod_local_quality": "Margin quality",
}

STANDARD_QUALITY_PLOT_LABELS = {
    key: QUALITY_METRIC_LABELS[key]
    for key in (
        "zod_local_quality",
        "mean_raw_assigned_iou",
    )
}

QUALITY_METRIC_RANGES = {
    "mean_raw_assigned_iou": (0.0, 1.0),
    "zod_local_quality": (-1.0, 1.0),
}

QUALITY_METRIC_HIGHER_IS_BETTER = {
    "mean_raw_assigned_iou": True,
    "zod_local_quality": True,
}


ROW_COLORS = {
    "auto": REGION_COLORS['Auto zones'], "invalid": "#9b233c",
}


MEASURE_LABELS = {
    "missing_masks": "Missing",
    "max_centroid_displacement_px": "Displacement",
    "max_temporal_nonoverlap": "1 − IoU",
    "max_symmetric_relative_area_change": "Area Δsym",
    "max_absolute_log_area_ratio": "Area |log ratio|",
    "max_symmetric_relative_circumference_change": "Circumference Δsym",
    "max_component_count": "Components",
    "max_hole_count": "Holes",
    "max_outside_largest_component_fraction": "Outside largest",
    "max_fragmentation_index": "Fragmentation",
    "max_shape_change_i1": "Shape",
    "max_isoperimetric_deficit": "Compactness deficit",
    "max_convex_hull_perimeter_excess": "Hull excess",
    "max_fft_boundary_high_frequency_energy": "FFT roughness",
    "max_multiscale_perimeter_excess": "Multiscale roughness",
    "max_normalized_bending_energy": "Bending roughness",
}


def measure_labels(data):
    """Names for the finite measure series available in either direction."""
    return {name: MEASURE_LABELS[name]
            for name in dict.fromkeys(name for values in data.values() for name in values)
            if any(np.isfinite(values[name]).any() for values in data.values() if name in values)}
