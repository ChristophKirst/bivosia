"""Extract recording-scale mask features and derived measures.

Examples
--------
>>> from bivosia.processing.features import calculate_mask_features, calculate_mask_measures
>>> calculate_mask_features(  # doctest: +SKIP
...     "masks/masks_forward.h5", "masks/masks_reverse.h5",
...     "analysis/mask_features.npy", n_animals=6)
>>> calculate_mask_measures("analysis/mask_features.npy")  # doctest: +SKIP
"""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from typing import Sequence

from pathlib import Path
import logging

import numpy as np

from bivosia.analysis.zone_detection import merge_intervals
from bivosia.config import ANALYSIS_DEFAULTS, PROCESSING_DEFAULTS
from bivosia.analysis.features import COMPONENT_DTYPE, FFT_FEATURE_NAMES, HOLE_DTYPE, feature_names, frame_features
from bivosia.analysis.measures import mask_measures
from bivosia.dataset.arrays import array_output, read_array, write_array
from bivosia.dataset.features import create_geometry, write_geometry_block
from bivosia.dataset.jsonio import atomic_json, read_json
from bivosia.dataset.masks import mask_shape, open_masks
from bivosia.dataset.containers import container_output
from bivosia.dataset.paths import companion_path, file_name
from bivosia.processing.log import Progress
from bivosia.processing.parallel import unordered_blocks, worker_count


logger = logging.getLogger(__name__)
_MASKS: dict[int, object] = {}
_FEATURE_NAMES: tuple[str, ...] = feature_names()
_INCLUDE_CONVEX_HULL = ANALYSIS_DEFAULTS['features']['include_convex_hull']
_INCLUDE_HU_FEATURES = ANALYSIS_DEFAULTS['features']['include_hu_features']
_INCLUDE_FFT_FEATURES = ANALYSIS_DEFAULTS['features']['include_fft_features']


def calculate_dataset_features(dataset, n_animals, *, annotations=None, progress=None,
                               cancelled=None, **settings):
    """Calculate raw features, geometry and measures at the dataset's output paths."""
    outputs = {name: dataset.array(name, write=True)
               for name in ('mask_features', 'mask_geometry', 'mask_measures')}
    for name in ('mask_features', 'mask_measures'):
        dataset.file(name, companion='metadata', write=True)
    intervals = None
    if annotations is not None:
        annotation = read_json(dataset.annotation_path(annotations))
        if annotation['schema'] != 1:
            raise ValueError('Expected a current revision/linking annotation')
        intervals = [(zone['start'], zone['end']) for zone in annotation['zones']]
    features = calculate_mask_features(
        dataset.mask('forward'), dataset.mask('reverse'), outputs['mask_features'],
        n_animals, geometry_output=outputs['mask_geometry'], intervals=intervals,
        progress=progress, cancelled=cancelled, **settings)
    calculate_mask_measures(features, outputs['mask_measures'], progress=progress, cancelled=cancelled)
    return outputs


def _open_worker(
    forward: str, reverse: str, dataset: str,
    include_convex_hull: bool, include_hu_features: bool, include_fft_features: bool,
) -> None:
    global _MASKS, _FEATURE_NAMES, _INCLUDE_CONVEX_HULL, _INCLUDE_HU_FEATURES
    global _INCLUDE_FFT_FEATURES
    _MASKS = {
        0: open_masks(forward, dataset, direction='forward'),
        1: open_masks(reverse, dataset, direction='reverse'),
    }
    _INCLUDE_CONVEX_HULL = include_convex_hull
    _INCLUDE_HU_FEATURES = include_hu_features
    _INCLUDE_FFT_FEATURES = include_fft_features
    _FEATURE_NAMES = feature_names(include_convex_hull=include_convex_hull,
                                   include_hu_features=include_hu_features,
                                   include_fft_features=include_fft_features)


def _concatenate(records: list[np.ndarray], dtype: np.dtype) -> np.ndarray:
    nonempty = [record for record in records if len(record)]
    return np.concatenate(nonempty) if nonempty else np.empty(0, dtype=dtype)


def _calculate_block(
    task: tuple[int, int, int, int],
) -> tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pass_index, start, end, n_animals = task
    source = _MASKS[pass_index]
    read_start = max(0, start - 1)
    labels = np.asarray(source[read_start:end])
    values = np.empty((end - start, n_animals, len(_FEATURE_NAMES)), dtype=np.float64)
    component_counts = np.zeros((end - start, n_animals), dtype=np.int32)
    hole_counts = np.zeros((end - start, n_animals), dtype=np.int32)
    component_records: list[np.ndarray] = []
    hole_records: list[np.ndarray] = []
    previous = None
    for offset, frame_number in enumerate(range(read_start, end)):
        current, previous, components, holes = frame_features(
            labels[offset], n_animals, previous,
            include_convex_hull=_INCLUDE_CONVEX_HULL,
            include_hu_features=_INCLUDE_HU_FEATURES,
            include_fft_features=_INCLUDE_FFT_FEATURES,
        )
        if frame_number < start:
            continue
        destination_offset = frame_number - start
        values[destination_offset] = current
        for animal in range(n_animals):
            component_counts[destination_offset, animal] = len(components[animal])
            hole_counts[destination_offset, animal] = len(holes[animal])
            component_records.append(components[animal])
            hole_records.append(holes[animal])
    return (
        pass_index,
        start,
        end,
        values,
        component_counts,
        _concatenate(component_records, COMPONENT_DTYPE),
        hole_counts,
        _concatenate(hole_records, HOLE_DTYPE),
    )



def calculate_mask_features(
    forward_path: str | Path,
    reverse_path: str | Path,
    output: str | Path,
    n_animals: int,
    *,
    workers: int | None = PROCESSING_DEFAULTS['features']['workers'],
    batch_size: int = PROCESSING_DEFAULTS['features']['batch_size'],
    dataset: str = "masks",
    intervals: Sequence[tuple[int, int]] | None = None,
    geometry_output: str | Path | None = None,
    include_convex_hull: bool = ANALYSIS_DEFAULTS['features']['include_convex_hull'],
    include_hu_features: bool = ANALYSIS_DEFAULTS['features']['include_hu_features'],
    include_fft_features: bool = ANALYSIS_DEFAULTS['features']['include_fft_features'],
    progress=None,
    cancelled=None,
) -> Path:
    """Write dense raw values and a ragged HDF5 geometry sidecar."""
    forward = Path(forward_path).expanduser().resolve()
    reverse = Path(reverse_path).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    geometry_path = (
        Path(geometry_output).expanduser().resolve()
        if geometry_output is not None
        else output_path.with_name(file_name("mask_geometry"))
    )
    if n_animals < 1 or batch_size < 1:
        raise ValueError("n_animals and batch_size must be positive")
    frame_count, frame_height, frame_width = mask_shape(forward, reverse, dataset)

    if intervals is None:
        intervals = [(0, frame_count - 1)]
    for start, end in intervals:
        if not 0 <= start <= end < frame_count:
            raise ValueError(f"Interval {start}–{end} is outside the recording")
    intervals = merge_intervals(tuple(intervals), 0)
    if not intervals:
        raise ValueError("No intervals were selected")
    tasks = [
        (
            pass_index,
            start,
            min(start + batch_size, interval_end + 1),
            n_animals,
        )
        for pass_index in range(2)
        for interval_start, interval_end in intervals
        for start in range(interval_start, interval_end + 1, batch_size)
    ]
    selected_pass_frames = 2 * sum(end - start + 1 for start, end in intervals)
    count = worker_count(workers, len(tasks))
    names = feature_names(include_convex_hull=include_convex_hull,
                          include_hu_features=include_hu_features,
                          include_fft_features=include_fft_features)

    report = Progress("Mask features", selected_pass_frames, logger=logger, callback=progress, cancelled=cancelled)
    with array_output(output_path, (2, frame_count, n_animals, len(names)),
                      np.float64, fill=np.nan) as destination, container_output(geometry_path) as geometry:
        create_geometry(geometry, frame_count, n_animals, frame_height, frame_width)
        for (
            pass_index,
            start,
            end,
            values,
            block_component_counts,
            block_components,
            block_hole_counts,
            block_holes,
        ) in unordered_blocks(
            _calculate_block,
            tasks,
            workers=count,
            initializer=_open_worker,
            initargs=(str(forward), str(reverse), dataset,
                      include_convex_hull, include_hu_features, include_fft_features),
        ):
            report.check_cancelled()
            destination[pass_index, start:end] = values
            write_geometry_block(geometry, pass_index, start, end,
                                 block_component_counts, block_components, block_hole_counts, block_holes)
            report.update(end - start)

    fft = ANALYSIS_DEFAULTS['features']['fft_features']
    metadata = {
        "schema_version": 4,
        "array": output_path.name,
        "geometry": geometry_path.name,
        "shape": [2, frame_count, n_animals, len(names)],
        "dtype": "float64",
        "axes": ["pass", "frame", "animal", "feature"],
        "pass_names": ["FORWARD", "REVERSE"],
        "feature_names": list(names),
        "convex_hull_included": include_convex_hull,
        "hu_features_included": include_hu_features,
        "fft_features_included": include_fft_features,
        "dataset": dataset,
        "intervals_inclusive": intervals,
        "definitions": {
            "area_px": "sum of every 8-connected foreground component area",
            "external_perimeter_px": "sum of external contour arc lengths; hole boundaries excluded",
            "temporal_intersection_px": "same-pass identity intersection at frames t-1 and t",
        },
    }
    if include_convex_hull:
        metadata['definitions']['convex_hull_perimeter_excess'] = 'largest-contour perimeter / convex-hull perimeter - 1'
    if include_fft_features:
        metadata['definitions'].update({
            'fft_boundary_high_frequency_energy': f"fraction of equally spaced largest-contour coordinate FFT energy above harmonic {fft['low_frequency_cutoff']}",
            'multiscale_perimeter_excess': 'mean of P_raw/P_lowpass-1 for largest-contour Fourier cutoffs ' + ', '.join(map(str, fft['perimeter_cutoffs'])),
            'normalized_bending_energy': f"scale-invariant bending energy of the harmonic-{fft['bending_cutoff']} largest contour, divided by 4*pi^2 and minus 1",
        })
    if include_hu_features:
        metadata['definitions'].update({
            'hu_log_1..7': 'signed base-10-log Hu moments of the largest component contour',
            'shape_change_i1': 'OpenCV matchShapes I1 on consecutive largest-component contours; degenerate failures are NaN',
        })
    atomic_json(companion_path(output_path, "mask_features", "metadata"), metadata)
    report.finish()
    return output_path



def calculate_mask_measures(features_path, output=None, *, progress=None, cancelled=None):
    """Write a (pass, frame, measure) worst-animal array."""
    source_path = Path(features_path).expanduser().resolve()
    output_path = Path(output).expanduser().resolve() if output is not None else source_path.with_name(file_name("mask_measures"))
    metadata = read_json(companion_path(source_path, "mask_features", "metadata"))
    source = read_array(source_path)
    report = Progress("Mask measures", 2, logger=logger, callback=progress, cancelled=cancelled)
    values, measure_names = mask_measures(source, metadata["feature_names"])
    report.update(2)
    write_array(output_path, values)
    output_metadata = {
        "schema_version": 4,
        "array": output_path.name,
        "shape": [2, source.shape[1], len(measure_names)],
        "dtype": "float32",
        "axes": ["pass", "frame", "measure"],
        "pass_names": list(metadata.get("pass_names", ["FORWARD", "REVERSE"])),
        "measure_names": list(measure_names),
        "higher_is_worse": True,
        "animal_aggregation": "X[p,t,k] = max_i x[p,t,i,k], independently for each pass and frame",
        "definitions": {
            "missing_masks": "expected animals minus animals with nonempty component lists",
            "max_temporal_nonoverlap": "max over animals of 1 - temporal IoU",
            "max_symmetric_relative_area_change": "max 2*abs(A_t-A_t-1)/(A_t+A_t-1)",
            "max_absolute_log_area_ratio": "max abs(log(A_t/A_t-1)) over animals with positive area in both frames",
            "max_symmetric_relative_circumference_change": "max 2*abs(P_t-P_t-1)/(P_t+P_t-1), external perimeter",
            "max_component_count": "maximum count of every 8-connected component, without size filtering",
            "max_outside_largest_component_fraction": "max 1-largest_component_area/total_area",
            "max_fragmentation_index": "max ((component_count-1) + hole_count + 1-largest_component_area/total_area), with component_count >= 1",
            "max_isoperimetric_deficit": "max P^2/(4*pi*A)-1 over animals",
            "max_convex_hull_perimeter_excess": "max P/P_convex_hull-1 over largest animal contours",
        },
    }
    for name in FFT_FEATURE_NAMES:
        measure = 'max_' + name
        if measure in measure_names:
            output_metadata['definitions'][measure] = 'max over animals of ' + metadata['definitions'][name]

    atomic_json(companion_path(output_path, "mask_measures", "metadata"), output_metadata)
    report.finish()
    return output_path
