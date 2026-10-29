"""Feature traces, envelopes and display ranges."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from dataclasses import dataclass

import numpy as np

from bivosia.analysis.measures import direction_difference


METRIC_ENVELOPES = {
    "mean_mask_area_px": ("min_mask_area_px", "max_mask_area_px"),
    "mean_centroid_displacement_px": (
        "min_centroid_displacement_px", "max_centroid_displacement_px"
    ),
    "mean_temporal_iou": ("min_temporal_iou", "max_temporal_iou"),
    "mean_relative_area_change": (
        "min_relative_area_change", "max_relative_area_change"
    ),
    "mean_fragmentation_index": (
        "min_fragmentation_index", "max_fragmentation_index"
    ),
    "mean_hole_count": ("min_hole_count", "max_hole_count"),
    "mean_raw_assigned_iou": (
        "min_raw_assigned_iou", "max_raw_assigned_iou"
    ),
}


@dataclass(frozen=True)
class FeatureCurve:
    frames: np.ndarray
    values: np.ndarray
    minimum: np.ndarray | None = None
    maximum: np.ndarray | None = None


def feature_curves(sparse, dense, metrics):
    """Return aligned F, R and F−R values once; missing frames remain gaps."""
    columns = {}
    for direction in ("FORWARD", "REVERSE"):
        if dense:
            data = dense.get(direction, {})
            columns[direction] = {
                key: FeatureCurve(np.arange(len(values)), np.asarray(values))
                for key, values in data.items()
            }
        else:
            data = sparse.get(direction, {})
            frames = np.asarray(sorted(data), dtype=np.int64)
            keys = set(metrics)
            for metric in metrics:
                keys.update(METRIC_ENVELOPES.get(metric, ()))
            columns[direction] = {
                key: FeatureCurve(frames, np.asarray([data[int(f)].get(key, np.nan) for f in frames]))
                for key in keys
            }
    result = {}
    for metric in metrics:
        for direction in ("FORWARD", "REVERSE"):
            values = columns[direction].get(metric)
            if values is None or not values.frames.size:
                continue
            bounds = METRIC_ENVELOPES.get(metric)
            low, high = (columns[direction].get(k) for k in bounds) if bounds else (None, None)
            result[direction, metric] = FeatureCurve(
                values.frames, values.values,
                low.values if low is not None and high is not None else None,
                high.values if low is not None and high is not None else None,
            )
        forward, reverse = (result.get((d, metric)) for d in ("FORWARD", "REVERSE"))
        if forward is None or reverse is None:
            continue
        frames, difference = direction_difference(forward.frames, forward.values, reverse.frames, reverse.values)
        if frames.size:
            result["DIFFERENCE", metric] = FeatureCurve(frames, difference)
    return result


def feature_ranges(curves, metrics, difference_scale="p95"):
    """Shared F/R scales; symmetric robust scales for the raw F−R difference."""
    if difference_scale not in {"p95", "3sigma", "6sigma", "raw_max"}:
        raise ValueError(f"Unknown difference scale: {difference_scale}")
    result = {}
    for metric in metrics:
        values = [
            a[np.isfinite(a)]
            for direction in ("FORWARD", "REVERSE")
            if (curve := curves.get((direction, metric))) is not None
            for a in (curve.values, curve.minimum, curve.maximum) if a is not None
        ]
        values = [a for a in values if a.size]
        if values:
            bounds = (min(float(a.min()) for a in values), max(float(a.max()) for a in values))
            for direction in ("FORWARD", "REVERSE"):
                result[direction, metric] = bounds
        difference = curves.get(("DIFFERENCE", metric))
        if difference is None:
            continue
        finite = difference.values[np.isfinite(difference.values)]
        magnitude = 0.
        if finite.size:
            if difference_scale == "p95":
                magnitude = float(np.percentile(np.abs(finite), 95))
            elif difference_scale in {"3sigma", "6sigma"}:
                magnitude = float(np.std(finite)) * (3 if difference_scale == "3sigma" else 6)
            elif values:
                magnitude = max(float(np.max(np.abs(a))) for a in values)
            if magnitude <= 0:
                magnitude = float(np.max(np.abs(finite)))
        magnitude = max(1e-12, magnitude)
        result["DIFFERENCE", metric] = (-magnitude, magnitude)
    return result
