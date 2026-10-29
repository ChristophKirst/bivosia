"""Apply detection parameters to prepared recording evidence.

Examples
--------
>>> from bivosia.processing.zones import detection_parameters
>>> detection_parameters(0.6, 0.9)["seed"]
0.6
"""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from bivosia.config import ANALYSIS_DEFAULTS
from bivosia.analysis.zone_detection import detect_zods


def detection_parameters(seed, recover, gap=ANALYSIS_DEFAULTS['zone_detection']['theta_gap']):
    return {"seed": float(seed), "recover": float(recover), "gap": int(gap),
            "detector": "local-margin-hungarian-v1"}


def detect_zones(evidence, seed=ANALYSIS_DEFAULTS['zone_detection']['theta_quality'],
                 recover=ANALYSIS_DEFAULTS['zone_detection']['theta_recovery'],
                 gap=ANALYSIS_DEFAULTS['zone_detection']['theta_gap'], *, missing=None):
    return detect_zods(evidence, seed, theta_recovery=recover,
                       theta_gap=gap, missing_signal=missing)
