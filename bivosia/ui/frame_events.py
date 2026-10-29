"""Missing-mask and switch-frame tracks for the zone timeline."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import numpy as np

from bivosia.analysis.zone_detection import boolean_runs


def frame_event_intervals(missing_signal=None, forward_missing_signal=None,
                          reverse_missing_signal=None, missing_direction_known=None,
                          switch_frames=None):
    """Classify missing-mask directions and switch frames as inclusive runs."""
    signals = [value for value in (
        missing_signal, forward_missing_signal, reverse_missing_signal,
        missing_direction_known, switch_frames,
    ) if value is not None]
    if signals:
        frame_count = len(signals[0])
        if any(np.asarray(value).shape != (frame_count,) for value in signals):
            raise ValueError("Zone frame signals must all have the same length")
        def flags(value: np.ndarray | None) -> np.ndarray:
            return np.zeros(frame_count, dtype=bool) if value is None else np.asarray(value) > 0
        forward = flags(forward_missing_signal)
        reverse = flags(reverse_missing_signal)
        missing = flags(missing_signal) if missing_signal is not None else forward | reverse
        switches = flags(switch_frames)
        if missing_direction_known is not None:
            known = flags(missing_direction_known)
        elif forward_missing_signal is not None and reverse_missing_signal is not None:
            known = np.isfinite(forward_missing_signal) & np.isfinite(reverse_missing_signal)
        else:
            known = np.zeros(frame_count, dtype=bool)
        event_signals = {
            "Missing forward only": missing & known & forward & ~reverse,
            "Missing reverse only": missing & known & reverse & ~forward,
            "Missing both": missing & known & forward & reverse,
            "Missing direction unknown": missing & (~known | ~(forward | reverse)),
            "Switch frames": switches,
            "Missing mask + switch": missing & switches,
        }
        return {label: list(boolean_runs(values)) for label, values in event_signals.items()}
    return {}
