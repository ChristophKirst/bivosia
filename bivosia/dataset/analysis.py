"""Read frame-level quality and mask measures."""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path

import numpy as np

from bivosia.dataset.arrays import MASK_MISSING, MASK_QUALITY, MASK_SWITCH, read_array
from bivosia.dataset.jsonio import read_json
from bivosia.analysis.zone_detection import ZodEvidence
from bivosia.analysis.masks import permutation_change_measures


def load_mask_measures(
    frame_count: int, path: Path | None, metadata_path: Path | None
) -> dict[str, dict[str, np.ndarray]]:
    """Load derived maximum-badness measures for both mask passes."""
    if path is None or not path.is_file():
        return {}
    if metadata_path is None or not metadata_path.is_file():
        raise ValueError(f"Mask-measure metadata does not exist: {metadata_path}")
    metadata = read_json(metadata_path)
    names = list(metadata.get("measure_names", []))
    source = read_array(path)
    if source.ndim != 3 or source.shape[0] != 2 or source.shape[2] != len(names):
        raise ValueError(
            f"Expected mask-measure shape (2, frames, measures): {path}"
        )
    if source.shape[1] != frame_count:
        raise ValueError(
            f"Mask measures have {source.shape[1]:,} frames but the video has "
            f"{frame_count:,}: {path}"
        )
    output = {
        pass_name: {
            name: source[pass_index, :, measure_index]
            for measure_index, name in enumerate(names)
        }
        for pass_index, pass_name in enumerate(("FORWARD", "REVERSE"))
    }
    return output

def load_quality_evidence(resolve, iou_match_path: str | Path):
    """Load persisted arrays as a ``ZodEvidence`` without recomputing quality."""
    assignment = read_array(Path(iou_match_path).expanduser().resolve())
    quality = read_array(resolve(MASK_QUALITY))
    missing = read_array(resolve(MASK_MISSING))
    switches = read_array(resolve(MASK_SWITCH))
    if quality.shape != (assignment.shape[0],) or missing.shape != quality.shape or switches.shape != quality.shape:
        raise ValueError("Persisted quality arrays have incompatible frame counts")
    changes, expected_switches = permutation_change_measures(assignment, missing)
    if not np.array_equal(switches, expected_switches):
        raise ValueError("Persisted mask_switch array does not match mask_iou_match and mask_missing")
    return ZodEvidence(quality, missing, assignment, changes, switches)
