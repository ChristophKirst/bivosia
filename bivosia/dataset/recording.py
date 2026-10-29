"""Load one recording's masks, cached evidence and video streams.

Examples
--------
>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.dataset.recording import Recording
>>> recording = Recording(Dataset("experiment"))  # doctest: +SKIP
>>> mask = recording.mask("forward", 0)  # doctest: +SKIP
>>> recording.close()  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import closing
from functools import lru_cache

import numpy as np

from bivosia.dataset.analysis import load_mask_measures, load_quality_evidence
from bivosia.dataset.arrays import MaskOverlaps, quality_cache_available, read_array
from bivosia.dataset.layout import Dataset
from bivosia.dataset.keypoints import load_keypoints
from bivosia.dataset.video import VideoSet
from bivosia.analysis.zone_detection import mask_count_seed_signal, prepare_zod_evidence
from bivosia.analysis.masks import missing_mask_counts
from bivosia.analysis.measures import recording_measures


class Recording:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.directory = self.dataset.directory
        with closing(MaskOverlaps(dataset.array("mask_intersections"), dataset.array("mask_areas"))) as overlaps:
            self.count, self.n, _ = overlaps.shape
            if self.n < 2:
                raise ValueError("Assignment quality requires at least two animals")
            matrices = overlaps.iou()
            counts = missing_mask_counts(overlaps.areas)
        match_path = self.dataset.array("mask_iou_match")
        if match_path.is_file() and quality_cache_available(self.dataset.analysis_file):
            self.evidence = load_quality_evidence(self.dataset.analysis_file, match_path)
        else:
            assignments = read_array(match_path) if match_path.is_file() else None
            self.evidence = prepare_zod_evidence(matrices, assignments)
        if self.evidence.assignment.shape != (self.count, self.n):
            raise ValueError("Cached assignments and mask-count dimensions differ")
        self.paths = {"original": self.dataset.video,
                      **{s: self.dataset.mask(s) for s in ("forward", "reverse")}}
        self.videos = VideoSet(self.paths)
        self.streams = {}
        try:
            if (any(self.videos.counts[source] != self.count for source in ("forward", "reverse"))
                    or self.videos.counts["original"] < self.count):
                raise ValueError(f"Masks must match the evidence length, and the video must cover it: {self.videos.counts}, evidence={self.count}")
            if self.mask("forward", 0).shape != self.mask("reverse", 0).shape:
                raise ValueError("Forward/reverse masks must share image coordinates")
            forward, reverse = counts[0] > 0, counts[1] > 0
            missing = mask_count_seed_signal(self.evidence, *counts)
            self.missing_forward, self.missing_reverse = forward, reverse
            self.missing, self.quality, self.assigned_iou = recording_measures(
                matrices, self.evidence, forward, reverse, missing)
            self.measures = load_mask_measures(self.count, self.dataset.array("mask_measures"),
                self.dataset.file("mask_measures", companion="metadata"))
            self.keypoints = load_keypoints(dataset)
            if self.keypoints is not None and self.keypoints.shape[0] != self.count:
                raise ValueError('Keypoints must cover the same frame interval as the masks')
        except BaseException:
            self.close()
            raise

    @lru_cache(maxsize=32)
    def mask(self, source, frame):
        mask = self.videos.read(source, int(frame))
        if not np.issubdtype(mask.dtype, np.integer) or mask.min() < 0 or mask.max() > self.n:
            raise ValueError(f"Invalid mask IDs in {source} at frame {frame}; expected 0–{self.n}")
        return mask

    def video_stream(self, name):
        if name is None:
            return self.videos
        if name not in self.streams:
            self.streams[name] = VideoSet({"original": self.paths["original"]})
        return self.streams[name]

    @lru_cache(maxsize=8)
    def image(self, frame, stream=None):
        return self.video_stream(stream).read("original", int(frame))

    def prefetch(self, frames, stream=None):
        self.video_stream(stream).prefetch("original", frames)

    def stop_prefetch(self, stream):
        if stream in self.streams:
            self.streams[stream].stop_prefetch()

    def close(self):
        self.mask.cache_clear()
        self.image.cache_clear()
        for stream in self.streams.values():
            stream.close()
        self.streams.clear()
        self.videos.close()
