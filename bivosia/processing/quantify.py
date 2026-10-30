"""Calculate recording-level mask intersections, areas, assignments and quality.

Examples
--------
With existing datasets containing forward and reverse masks:

>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.processing.quantify import quantify, quantify_many
>>> dataset = Dataset("/path/to/dataset")
>>> outputs = quantify(dataset, n_animals=6, workers=2)  # doctest: +SKIP
>>> datasets = [Dataset("/path/to/first"), Dataset("/path/to/second")]
>>> outputs = quantify_many(datasets, n_animals=2, workers=4)  # doctest: +SKIP
"""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import ExitStack, closing
from pathlib import Path
import logging

import numpy as np

from bivosia.config import ANALYSIS_DEFAULTS, PROCESSING_DEFAULTS
from bivosia.analysis.mask_features import mask_areas
from bivosia.analysis.quality import (
    frame_quality_measures, mask_intersections, optimal_assignments,
    permutation_change_measures, temporal_assignment_quality,
)
from bivosia.dataset.paths import file_name
from bivosia.dataset.arrays import MaskOverlaps, array_output, write_array
from bivosia.dataset.jsonio import atomic_json
from bivosia.dataset.masks import mask_shape, open_masks
from bivosia.processing.log import Progress
from bivosia.processing.parallel import unordered_blocks, worker_count


logger = logging.getLogger(__name__)
_FORWARD = _REVERSE = _MASK_PATHS = None


def _close_masks():
    global _FORWARD, _REVERSE, _MASK_PATHS
    for masks in (_FORWARD, _REVERSE):
        if masks is not None:
            masks.close()
    _FORWARD = _REVERSE = _MASK_PATHS = None


def _open_masks(forward, reverse, dataset):
    global _FORWARD, _REVERSE, _MASK_PATHS
    paths = (forward, reverse, dataset)
    if paths != _MASK_PATHS:
        _close_masks()
        try:
            _FORWARD = open_masks(forward, dataset, direction='forward')
            _REVERSE = open_masks(reverse, dataset, direction='reverse')
        except BaseException:
            _close_masks()
            raise
        _MASK_PATHS = paths


def _overlap_block(task):
    recording, forward_path, reverse_path, start, end, n, temporal_intersections, dataset, count_dtype = task
    _open_masks(forward_path, reverse_path, dataset)
    read_start = max(0, start - 1)
    forward, reverse = np.asarray(_FORWARD[read_start:end]), np.asarray(_REVERSE[read_start:end])
    result = np.empty((end - start, n, n), dtype=count_dtype)
    areas = np.empty((2, end - read_start, n), dtype=count_dtype)
    temporal = np.empty((2, end - max(1, start), n, n), dtype=count_dtype)
    previous = None
    for offset, frame in enumerate(range(read_start, end)):
        current = tuple(masks[offset] for masks in (forward, reverse))
        areas[:, offset] = tuple(mask_areas(mask, n) for mask in current)
        if frame >= start:
            result[frame - start] = mask_intersections(*current, n)
            if previous is not None:
                for direction in range(2):
                    temporal[direction, frame - max(1, start)] = mask_intersections(
                        previous[direction], current[direction], n)
        previous = current
    quality = temporal_assignment_quality(temporal, areas)
    return (recording, start, end, result, areas[:, start - read_start:],
            temporal if temporal_intersections else None, quality)


def calculate_mask_overlaps(forward_path, reverse_path, output, n_animals, *,
                            workers=PROCESSING_DEFAULTS['quantify']['workers'],
                            batch_size=PROCESSING_DEFAULTS['quantify']['overlap_batch_size'],
                            dataset="masks",
                            temporal_intersections=ANALYSIS_DEFAULTS['quality']['temporal_intersections'],
                            progress=None, cancelled=None):
    """Write pixel counts and temporal quality, optionally saving temporal counts.

    Output intersections have shape (frames, animals, animals). The sibling
    area array has shape (2, frames, animals), ordered forward then reverse.
    Temporal intersections have shape (2, frames-1, animals, animals):
    [d, t, i, j] compares ID i+1 at t with ID j+1 at t+1 in direction d.
    """
    _calculate_mask_overlaps([(forward_path, reverse_path, output)], n_animals,
                             workers=workers, batch_size=batch_size, dataset=dataset,
                             temporal_intersections=temporal_intersections,
                             progress=progress, cancelled=cancelled)
    return Path(output)


def _calculate_mask_overlaps(jobs, n_animals, *,
                             workers=PROCESSING_DEFAULTS['quantify']['workers'],
                             batch_size=PROCESSING_DEFAULTS['quantify']['overlap_batch_size'],
                             dataset="masks",
                             temporal_intersections=ANALYSIS_DEFAULTS['quality']['temporal_intersections'],
                             progress=None, cancelled=None):
    """Share one worker pool across recording blocks; commit completed recordings."""
    if n_animals < 2 or batch_size < 1:
        raise ValueError("n_animals must be at least two and batch_size positive")
    jobs = [tuple(map(Path, job)) for job in jobs]
    if not jobs:
        raise ValueError("No recordings selected")
    if len({output.resolve().parent for _, _, output in jobs}) != len(jobs):
        raise ValueError("Each recording must have a distinct output directory")
    count_dtype = np.dtype(ANALYSIS_DEFAULTS['quality']['count_dtype'])
    shapes = [mask_shape(forward, reverse, dataset) for forward, reverse, _ in jobs]
    if any(height * width > np.iinfo(count_dtype).max for _, height, width in shapes):
        raise ValueError(f"Frame pixel count exceeds {count_dtype} capacity")
    frames = [shape[0] for shape in shapes]
    if min(frames) < 1:
        raise ValueError("Mask recordings must contain at least one frame")
    # Group blocks by recording to reuse open masks and retain sequential reads.
    tasks = [(index, str(forward), str(reverse), start, min(start + batch_size, frames[index]),
              n_animals, temporal_intersections, dataset, count_dtype)
             for index, (forward, reverse, _) in enumerate(jobs)
             for start in range(0, frames[index], batch_size)]
    count = worker_count(workers, len(tasks))
    report = Progress("Mask intersections and areas", sum(frames), logger=logger,
                      callback=progress, cancelled=cancelled)
    logger.info("Processing %s recording(s) with %s worker(s)", len(jobs), count)
    remaining, targets = frames.copy(), {}
    try:
        with ExitStack() as stack:
            with closing(unordered_blocks(_overlap_block, tasks, workers=count)) as results:
                for index, start, end, values, block_areas, block_temporal, block_quality in results:
                    report.check_cancelled()
                    if index not in targets:
                        output = jobs[index][2]
                        outputs = stack.enter_context(ExitStack())
                        destination = outputs.enter_context(array_output(
                            output, (frames[index], n_animals, n_animals), count_dtype))
                        areas = outputs.enter_context(array_output(output.with_name(file_name("mask_areas")),
                            (2, frames[index], n_animals), count_dtype))
                        quality = outputs.enter_context(array_output(
                            output.with_name(file_name("mask_temporal_quality")),
                            (2, frames[index] - 1), np.float32))
                        temporal = None
                        if temporal_intersections:
                            temporal = outputs.enter_context(array_output(
                                output.with_name(file_name("mask_temporal_intersections")),
                                (2, frames[index] - 1, n_animals, n_animals), count_dtype))
                        targets[index] = outputs, destination, areas, temporal, quality
                    outputs, destination, areas, temporal, quality = targets[index]
                    destination[start:end] = values
                    areas[:, start:end] = block_areas
                    quality[:, max(0, start - 1):end - 1] = block_quality
                    if temporal is not None:
                        temporal[:, max(0, start - 1):end - 1] = block_temporal
                    remaining[index] -= end - start
                    report.update(end - start)
                    if remaining[index] == 0:
                        outputs.close()
                        del targets[index]
                        logger.info("Saved overlaps: %s", jobs[index][2])
    finally:
        _close_masks()
    report.finish()


def calculate_quality_measures(intersection_path, area_path, output_directory, *,
                               batch_size=PROCESSING_DEFAULTS['quantify']['quality_batch_size'],
                               progress=None, cancelled=None):
    """Derive assignments, signed margin quality, missing masks and switch frames."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output = Path(output_directory)
    paths = {key: output / file_name(key) for key in ("mask_iou_match", "mask_quality", "mask_missing", "mask_switch")}
    paths["metadata"] = output / file_name("mask_quality_metadata")
    with closing(MaskOverlaps(intersection_path, area_path)) as overlaps, ExitStack() as stack:
        frame_count, n_animals, _ = overlaps.shape
        if n_animals < 2:
            raise ValueError("Assignment quality requires at least two animals")
        report = Progress("Assignments and quality", frame_count, logger=logger,
                          callback=progress, cancelled=cancelled)
        matches = stack.enter_context(array_output(paths["mask_iou_match"], (frame_count, n_animals), np.int16))
        quality = stack.enter_context(array_output(paths["mask_quality"], (frame_count,), np.float32))
        missing = stack.enter_context(array_output(paths["mask_missing"], (frame_count,), bool))
        for start in range(0, frame_count, batch_size):
            report.check_cancelled()
            end = min(start + batch_size, frame_count)
            matrices = overlaps.iou(slice(start, end))
            assignment = optimal_assignments(matrices)
            values, flags, _assigned, _margin = frame_quality_measures(matrices, assignment)
            matches[start:end], quality[start:end], missing[start:end] = assignment, values, flags
            report.update(end - start)
        # Compare the complete sequence so changes across block edges are retained.
        _changes, switches = permutation_change_measures(matches, missing)
        report.check_cancelled()
        write_array(paths["mask_switch"], switches)
    atomic_json(paths["metadata"], {
        "frame_count": frame_count, "n_animals": n_animals,
        "quality_definition": "minimum signed local assignment margin across animals",
        "missing_definition": "any forward or reverse mask has zero area",
        "switch_definition": "both frames adjacent to a valid change in the forward-to-reverse mask_iou_match permutation",
    })
    report.finish()
    return paths


def quantify_job(dataset, *, progress=None, cancelled=None, **settings):
    """Run mask analysis with persisted job progress and serializable output paths."""
    callback = (None if progress is None else
                lambda completed, total: progress({'completed': completed, 'total': total}))
    outputs = quantify(dataset, progress=callback, cancelled=cancelled, **settings)
    return [str(path) for path in outputs.values()]


def quantify(dataset, n_animals, *, workers=PROCESSING_DEFAULTS['quantify']['workers'],
             temporal_intersections=ANALYSIS_DEFAULTS['quality']['temporal_intersections'],
             progress=None, cancelled=None):
    """Run basic mask quantification using the dataset's standard output paths."""
    return quantify_many([dataset], n_animals, workers=workers,
                         temporal_intersections=temporal_intersections,
                         progress=progress, cancelled=cancelled)[0]


def quantify_many(datasets, n_animals, *, workers=PROCESSING_DEFAULTS['quantify']['workers'],
                  temporal_intersections=ANALYSIS_DEFAULTS['quality']['temporal_intersections'],
                  progress=None, cancelled=None):
    """Quantify recordings with a shared mask-processing pool and per-recording outputs."""
    if n_animals < 2:
        raise ValueError("Assignment quality requires at least two animals")
    datasets = list(datasets)
    options = {"progress": progress, "cancelled": cancelled}
    names = ["mask_intersections", "mask_areas", "mask_temporal_quality", "mask_iou_match", "mask_quality", "mask_missing", "mask_switch", "mask_quality_metadata"]
    if temporal_intersections:
        names.append("mask_temporal_intersections")
    outputs = [{name: item.array(name, write=True) for name in names} for item in datasets]
    jobs = [(item.mask("forward"), item.mask("reverse"), paths["mask_intersections"])
            for item, paths in zip(datasets, outputs)]
    _calculate_mask_overlaps(jobs, n_animals, workers=workers,
                             temporal_intersections=temporal_intersections, **options)
    results = []
    for item, output in zip(datasets, outputs):
        logger.info("Assignments and quality: %s", item.directory)
        quality = calculate_quality_measures(output["mask_intersections"], output["mask_areas"],
                                             output["mask_quality"].parent, **options)
        paths = {"mask_intersections": output["mask_intersections"], "mask_areas": output["mask_areas"],
                 "mask_temporal_quality": output["mask_temporal_quality"], **quality}
        if temporal_intersections:
            paths["mask_temporal_intersections"] = output["mask_temporal_intersections"]
        results.append(paths)
    return results
