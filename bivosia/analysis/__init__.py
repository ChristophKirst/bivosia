"""Generalized Zone-of-Disagreement analysis algorithms.

Examples
--------
>>> from bivosia.analysis import detect_zods, prepare_zod_evidence
>>> import numpy as np
>>> evidence = prepare_zod_evidence(np.eye(2)[None, :, :])
>>> detect_zods(evidence, theta_quality=0.6).final_zods
()
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from bivosia.analysis.quality import frame_quality_measures, permutation_change_measures
from bivosia.analysis.zones_of_disagreement import (
    ZodEvidence,
    ZodResult,
    detect_zods,
    mask_count_seed_signal,
    prepare_zod_evidence,
)


__all__ = (
    "ZodEvidence", "ZodResult", "detect_zods", "frame_quality_measures",
    "mask_count_seed_signal", "permutation_change_measures", "prepare_zod_evidence",
)
