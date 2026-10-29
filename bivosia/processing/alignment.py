"""Align recording keypoints to annotation-selected masks in bounded blocks.

Examples
--------
With an existing dataset containing masks, a selected annotation and keypoints:

>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.processing.alignment import calculate_keypoint_alignment
>>> dataset = Dataset("/path/to/dataset")
>>> result = calculate_keypoint_alignment(dataset, 2, workers=4)  # doctest: +SKIP
"""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import closing
import json
import logging

import numpy as np

from bivosia.config import ANALYSIS_DEFAULTS, PROCESSING_DEFAULTS
from bivosia.analysis.alignment import Alignment, iter_align_scores, alignment_valid_frames
from bivosia.analysis.masks import mask_presence
from bivosia.analysis.point_masks import point_mask_overlaps, point_overlap_scores
from bivosia.analysis.revision import RevisionZone, preferred_sources
from bivosia.dataset.jsonio import read_json
from bivosia.dataset.keypoints import (annotation_digest, open_keypoints,
                               write_keypoint_alignment)
from bivosia.dataset.masks import mask_pair
from bivosia.dataset.paths import auxiliary_path
from bivosia.processing.log import Progress
from bivosia.processing.parallel import unordered_blocks, worker_count


logger = logging.getLogger(__name__)


def alignment_job(dataset, *, progress=None, cancelled=None, **settings):
    """Run keypoint alignment with saved job progress and a serializable result."""
    callback = (None if progress is None else
                lambda completed, total: progress({'completed': completed, 'total': total}))
    calculate_keypoint_alignment(dataset, progress=callback, cancelled=cancelled, **settings)
    return [str(dataset.array('keypoint_alignment'))]


def _overlap_block(task):
    (forward_path, reverse_path, reader, start, stop, sources, valid, n, kernel) = task
    points = reader[start:stop]
    overlaps = np.zeros((*points.shape[:3], n), dtype=bool)
    usable = np.zeros(points.shape[:3], dtype=bool)
    present = np.zeros((stop - start, n), dtype=bool)
    with mask_pair(forward_path, reverse_path, require_valid=False) as masks:
        for offset in np.flatnonzero(valid):
            mask = masks[sources[offset]][start + offset]
            overlaps[offset], usable[offset] = point_mask_overlaps(
                points[offset], mask, n, dilation_kernel=kernel)
            present[offset] = mask_presence(mask, n)
    return start, stop, overlaps, usable, present


def calculate_keypoint_overlaps(dataset, points, sources, valid, n_masks, *,
                                dilation_kernel=ANALYSIS_DEFAULTS['point_masks']['dilation_kernel'],
                                workers=PROCESSING_DEFAULTS['alignment']['workers'],
                                block_size=PROCESSING_DEFAULTS['alignment']['block_size'],
                                progress=None, cancelled=None):
    """Collect containment from a supplied PointData reader on valid frames.

    Readers passed to multiple workers must be serializable. The SLEAP reader
    keeps only file metadata between reads and supports this directly.
    """
    count, tracks, nodes = points.shape[:3]
    overlaps = np.zeros((count, tracks, nodes, n_masks), dtype=bool)
    usable = np.zeros((count, tracks, nodes), dtype=bool)
    present = np.zeros((count, n_masks), dtype=bool)
    report = Progress('Keypoint containment', count, logger=logger, callback=progress, cancelled=cancelled)
    tasks = [(dataset.mask('forward'), dataset.mask('reverse'), points, start,
              min(start + block_size, count), sources[start:start + block_size],
              valid[start:start + block_size], n_masks, dilation_kernel)
             for start in range(0, count, block_size)]
    with closing(unordered_blocks(_overlap_block, tasks, workers=worker_count(workers, len(tasks)))) as blocks:
        for start, stop, contained, in_image, mask_present in blocks:
            overlaps[start:stop], usable[start:stop], present[start:stop] = contained, in_image, mask_present
            report.update(stop - start)
    report.finish()
    return overlaps, usable, present


def calculate_keypoint_alignment(dataset, n_animals, *, invalid_point_frames=(),
                                 exclude_keypoints=(),
                                 fallback_previous=ANALYSIS_DEFAULTS['alignment']['fallback_previous'],
                                 dilation_kernel=ANALYSIS_DEFAULTS['point_masks']['dilation_kernel'],
                                 workers=PROCESSING_DEFAULTS['alignment']['workers'],
                                 block_size=PROCESSING_DEFAULTS['alignment']['block_size'],
                                 progress=None, cancelled=None, points=None):
    """Save assignments against the current annotation's preferred raw mask labels.

    Invalid annotation zones and invalid point frames are excluded. Outside
    zones the dataset selects forward masks. Source codes accompany assignments
    so a viewer can distinguish forward labels from reverse labels. Temporal
    fallback requires consistent mask identities across the selected sequence.
    exclude_keypoints lists node names omitted from both C and K in C**2/K.
    Per-node overlaps retain all nodes, including excluded ones, for rescoring.
    Supply a PointData reader in mask coordinates, or load the dataset's reader.
    """
    if n_animals < 1 or block_size < 1:
        raise ValueError('Animal count and block size must be positive')
    annotation = dataset.annotation
    if annotation is None:
        raise ValueError('Select a mask annotation before aligning keypoints')
    destination = dataset.array('keypoint_alignment', write=True)
    if destination.exists() or auxiliary_path(destination, 'partial').exists():
        raise ValueError('Keypoint alignment output already exists')
    fingerprint = annotation_digest(annotation)
    record = read_json(annotation)
    points = open_keypoints(dataset) if points is None else points
    exclude_keypoints = tuple(exclude_keypoints)
    for name in exclude_keypoints:
        if name not in points.node_names:
            raise ValueError(f'Unknown keypoint to exclude: {name}')
    points.excluded_keypoints = exclude_keypoints
    node_ids = points.selected_node_ids
    count, n_tracks = points.shape[:2]
    sources = preferred_sources([RevisionZone(**zone) for zone in record['zones']], count)
    with mask_pair(dataset.mask('forward'), dataset.mask('reverse'), require_valid=False) as (forward, reverse):
        if forward.shape[0] != count:
            raise ValueError('Keypoints and masks must cover the same frame interval')
        if forward.image_shape != points.image_shape:
            raise ValueError('Point and mask image dimensions differ')
        image_shape = forward.image_shape
        scale = points.coordinate_scale
        mask_valid = ((sources == 0) & forward.valid) | ((sources == 1) & reverse.valid)
    valid = alignment_valid_frames(mask_valid & points.valid, invalid_point_frames)
    overlaps, usable, present = calculate_keypoint_overlaps(dataset, points, sources, valid, n_animals,
        dilation_kernel=dilation_kernel, workers=workers, block_size=block_size,
        progress=progress, cancelled=cancelled)
    scores = point_overlap_scores(overlaps[:, :, node_ids], usable[:, :, node_ids])
    report = Progress('Keypoint assignment', count, logger=logger, callback=progress, cancelled=cancelled)
    assignment = np.full((count, n_tracks), -1, dtype=np.int32)
    tied = np.zeros(count, dtype=bool)
    for frame, labels, is_tied in iter_align_scores(scores, valid, fallback_previous=fallback_previous):
        report.check_cancelled()
        assignment[frame], tied[frame] = labels, is_tied
        if frame == report.done:
            report.update(1)
    result = Alignment(assignment, valid, tied)
    if annotation_digest(annotation) != fingerprint:
        raise ValueError('Annotation changed while aligning keypoints; result was not saved')
    write_keypoint_alignment(destination, result, sources, scores, present, overlaps, usable, n_animals, {
        'annotation': dataset.relative(annotation), 'annotation_sha256': fingerprint,
        'coordinate_scale': scale, 'dilation_kernel': dilation_kernel,
        'fallback_previous': fallback_previous,
        'excluded_keypoints': json.dumps(exclude_keypoints),
        'node_names': json.dumps(points.node_names), 'track_names': json.dumps(points.track_names),
        'image_shape': image_shape, 'source_image_shape': points.source_image_shape,
        'coordinate_rounding': 'nearest integer; half-integers to even',
    })
    report.finish(dataset.relative(destination))
    return result
