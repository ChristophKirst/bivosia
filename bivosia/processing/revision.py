"""Detection and linking access for a loaded recording.

Examples
--------
>>> from bivosia.processing.revision import RevisionData
>>> from bivosia.dataset.layout import Dataset
>>> data = RevisionData(Dataset("experiment"))  # doctest: +SKIP
>>> zones = data.detect(0.6, 0.9, 0)  # doctest: +SKIP
>>> data.close()  # doctest: +SKIP
"""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from functools import lru_cache

from bivosia.config import ANALYSIS_DEFAULTS
from bivosia.analysis.linking import Linker
from bivosia.analysis.mask_features import mask_centers
from bivosia.analysis.zones_of_disagreement import detect_zods, detection_parameters
from bivosia.dataset.recording import Recording


class RevisionData(Recording):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linking = (Linker(self.mask, self.n, self.quality, self.missing, self.evidence.switch_frames)
                        if self.evidence is not None and {'forward', 'reverse'} <= self.videos.masks.keys()
                        else None)

    @lru_cache(maxsize=32)
    def centers(self, source, frame):
        return mask_centers(self.mask(source, frame), self.n)

    def detect(self, seed=ANALYSIS_DEFAULTS['zones_of_disagreement']['theta_quality'],
               recover=ANALYSIS_DEFAULTS['zones_of_disagreement']['theta_recovery'], gap=ANALYSIS_DEFAULTS['zones_of_disagreement']['theta_gap']):
        return detect_zods(self.evidence, seed, theta_recovery=recover,
                           theta_gap=gap, missing_signal=self.missing)

    def header(self, seed, recover, gap, *,
               direction_method=ANALYSIS_DEFAULTS['zones_of_disagreement']['direction_method']):
        return detection_parameters(seed, recover, gap, direction_method=direction_method)

    def close(self):
        self.centers.cache_clear()
        if getattr(self, "linking", None) is not None:
            self.linking.clear()
        super().close()
