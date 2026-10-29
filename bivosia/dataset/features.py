"""Ragged component and hole records accompanying dense mask features."""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import json

import numpy as np

from bivosia.analysis.features import COMPONENT_DTYPE, HOLE_DTYPE


def create_geometry(geometry, frame_count, n_animals, frame_height, frame_width):
    geometry.create_dataset(
        "component_start", (2, frame_count, n_animals), dtype="i8", fillvalue=-1
    )
    geometry.create_dataset(
        "component_count", (2, frame_count, n_animals), dtype="i4", fillvalue=0
    )
    geometry.create_dataset(
        "hole_start", (2, frame_count, n_animals), dtype="i8", fillvalue=-1
    )
    geometry.create_dataset(
        "hole_count", (2, frame_count, n_animals), dtype="i4", fillvalue=0
    )
    geometry.create_dataset(
        "components", (0,), maxshape=(None,), chunks=(65536,), dtype=COMPONENT_DTYPE
    )
    geometry.create_dataset(
        "holes", (0,), maxshape=(None,), chunks=(65536,), dtype=HOLE_DTYPE
    )
    geometry.attrs.update(
        {
            "axes": "pass,frame,animal",
            "pass_names": json.dumps(["FORWARD", "REVERSE"]),
            "frame_height": frame_height,
            "frame_width": frame_width,
            "component_connectivity": 8,
            "all_component_sizes_retained": True,
        }
    )


def _append_records(dataset, records):
    start = len(dataset)
    if len(records):
        dataset.resize((start + len(records),))
        dataset[start:] = records
    return start


def write_geometry_block(geometry, pass_index, start, end, counts, records, hole_counts, hole_records):
    for name, block_counts, block_records in (
        ("component", counts, records), ("hole", hole_counts, hole_records)
    ):
        dataset_name = "components" if name == "component" else "holes"
        base = _append_records(geometry[dataset_name], block_records)
        offsets = base + np.cumsum(np.pad(block_counts.ravel()[:-1], (1, 0)))
        geometry[name + "_start"][pass_index, start:end] = offsets.reshape(block_counts.shape)
        geometry[name + "_count"][pass_index, start:end] = block_counts
