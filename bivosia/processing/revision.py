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
from bivosia.analysis.masks import mask_centers
from bivosia.dataset.recording import Recording
from bivosia.processing.zones import detect_zones, detection_parameters


class RevisionData(Recording):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linking = Linker(self.mask, self.n, self.quality, self.missing, self.evidence.switch_frames)

    @lru_cache(maxsize=32)
    def centers(self, source, frame):
        return mask_centers(self.mask(source, frame), self.n)

    def detect(self, seed=ANALYSIS_DEFAULTS['zone_detection']['theta_quality'],
               recover=ANALYSIS_DEFAULTS['zone_detection']['theta_recovery'], gap=ANALYSIS_DEFAULTS['zone_detection']['theta_gap']):
        return detect_zones(self.evidence, seed, recover, gap, missing=self.missing)

    def header(self, seed, recover, gap):
        return detection_parameters(seed, recover, gap)

    def close(self):
        self.centers.cache_clear()
        if hasattr(self, "linking"):
            self.linking.clear()
        super().close()
