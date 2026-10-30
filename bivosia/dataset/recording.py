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
from bivosia.analysis.zones_of_disagreement import mask_count_seed_signal, prepare_zod_evidence
from bivosia.analysis.mask_features import missing_mask_counts
from bivosia.analysis.quality import recording_measures


class Recording:
    def __init__(self, dataset: Dataset, *, remote_reader=None, require_analysis=True):
        self.remote_reader = remote_reader
        self.streams = {}
        self.videos = None
        try:
            self._load(dataset, require_analysis)
        except BaseException:
            self.close()
            raise

    def _load(self, dataset, require_analysis):
        self.dataset, self.directory = dataset, dataset.directory
        self.paths = {"original": dataset.video}
        masks = {}
        remote_paths = self.remote_reader.mask_paths() if self.remote_reader else {}
        for direction in ('forward', 'reverse'):
            path = dataset.mask(direction)
            if self.remote_reader and direction in remote_paths:
                self.paths[direction] = remote_paths[direction]
                masks[direction] = self.remote_reader.open_masks(direction)
            elif path.is_file() or require_analysis:
                self.paths[direction] = path
        self.videos = VideoSet(self.paths, masks=masks)
        if not require_analysis:
            self.videos.frame_count = self.videos.counts['original']
        self.count = self.videos.frame_count
        self.n = self.evidence = self.quality = self.assigned_iou = None
        self.missing = self.missing_forward = self.missing_reverse = None
        analysis = self.remote_reader.analysis_dataset() if self.remote_reader else dataset
        if self.remote_reader and not any(analysis.analysis.glob('*')):
            analysis = dataset
        self.annotation = dataset.annotation or analysis.annotation
        self._load_analysis(analysis, require_analysis)
        if require_analysis:
            if any(self.videos.counts[source] != self.count for source in ('forward', 'reverse')):
                raise ValueError('Masks must match the evidence length')
            if self.mask('forward', 0).shape != self.mask('reverse', 0).shape:
                raise ValueError('Forward/reverse masks must share image coordinates')
        self.temporal_quality = self._array(analysis, 'mask_temporal_quality', (2, self.count - 1))
        self.measures = load_mask_measures(self.count, analysis.array('mask_measures'),
                                           analysis.file('mask_measures', companion='metadata'))
        reference = next(iter(self.videos.masks.values()), None)
        self.keypoints = load_keypoints(dataset, masks=reference) if reference is not None else None
        if self.keypoints is not None and self.keypoints.shape[0] != self.count:
            raise ValueError('Keypoints must cover the same frame interval as the recording')

    def _load_analysis(self, analysis, required):
        matrices = None
        intersections, areas = analysis.array('mask_intersections'), analysis.array('mask_areas')
        if required or (intersections.is_file() and areas.is_file()):
            with closing(MaskOverlaps(intersections, areas)) as overlaps:
                count, self.n, _ = overlaps.shape
                if count != self.count:
                    raise ValueError('Mask analysis and recording frame counts differ')
                if required and self.n < 2:
                    raise ValueError('Assignment quality requires at least two animals')
                matrices = overlaps.iou()
                self.missing_forward, self.missing_reverse = missing_mask_counts(overlaps.areas)
        match_path = analysis.array('mask_iou_match')
        if match_path.is_file() and quality_cache_available(analysis.analysis_file):
            self.evidence = load_quality_evidence(analysis.analysis_file, match_path)
        elif required:
            assignments = read_array(match_path) if match_path.is_file() else None
            self.evidence = prepare_zod_evidence(matrices, assignments)
        if self.evidence is not None:
            if self.n is None:
                self.n = self.evidence.assignment.shape[1]
            if self.evidence.assignment.shape != (self.count, self.n):
                raise ValueError('Cached assignments and recording dimensions differ')
            self.quality, self.missing = self.evidence.quality, self.evidence.missing
            if matrices is not None:
                missing = mask_count_seed_signal(self.evidence, self.missing_forward, self.missing_reverse)
                self.missing, self.quality, self.assigned_iou = recording_measures(
                    matrices, self.evidence, self.missing_forward > 0, self.missing_reverse > 0, missing)
        else:
            self.quality = self._array(analysis, 'mask_quality', (self.count,))
            self.missing = self._array(analysis, 'mask_missing', (self.count,))

    @staticmethod
    def _array(dataset, name, shape):
        path = dataset.array(name)
        if not path.is_file():
            return None
        values = read_array(path)
        if values.shape != shape:
            raise ValueError(f'{name} and recording dimensions differ')
        return values

    @lru_cache(maxsize=32)
    def mask(self, source, frame):
        mask = self.videos.read(source, int(frame))
        if not np.issubdtype(mask.dtype, np.integer) or mask.min() < 0 or (self.n is not None and mask.max() > self.n):
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
        if self.videos is not None:
            self.videos.close()
        if self.remote_reader is not None:
            self.remote_reader.close()
