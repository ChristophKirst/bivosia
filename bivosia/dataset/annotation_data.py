"""Load saved mask-analysis evidence needed for annotation and export jobs."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import closing

from bivosia.analysis.mask_features import missing_mask_counts
from bivosia.dataset.analysis import load_quality_evidence
from bivosia.dataset.arrays import MaskOverlaps, read_array
from bivosia.dataset.session import RevisionSession


class AnnotationData:
    def __init__(self, dataset):
        self.arrays = []
        try:
            with closing(MaskOverlaps(dataset.array('mask_intersections'), dataset.array('mask_areas'))) as counts:
                self.count, self.n, _ = counts.shape
                self.missing_forward, self.missing_reverse = missing_mask_counts(counts.areas)
            self.evidence = load_quality_evidence(dataset.analysis_file, dataset.array('mask_iou_match'))
            self.arrays.extend((self.evidence.assignment, self.evidence.quality, self.evidence.missing,
                                self.evidence.switch_frames))
            if self.evidence.assignment.shape != (self.count, self.n):
                raise ValueError('Mask analysis dimensions differ')
            path = dataset.array('mask_temporal_quality')
            self.temporal_quality = read_array(path) if path.is_file() else None
            if self.temporal_quality is not None:
                self.arrays.append(self.temporal_quality)
                if self.temporal_quality.shape != (2, self.count - 1):
                    raise ValueError('Temporal quality and mask frame counts differ')
        except BaseException:
            self.close()
            raise

    def session(self, path):
        return RevisionSession.load(path, self.missing_forward, self.missing_reverse, self.n,
                                    temporal_quality=self.temporal_quality)

    def close(self):
        for array in self.arrays:
            if getattr(array, '_mmap', None) is not None:
                array._mmap.close()
        self.arrays.clear()
