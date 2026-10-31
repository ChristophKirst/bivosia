"""Persist mask measures, component geometry, shape descriptors and UMAP embeddings.

Examples
--------
>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.dataset.mask_features import embedding_path
>>> embedding_path(Dataset('/tmp/example'), z_score=True).name
'mask_umap_zscore.h5'
>>> embedding_path(Dataset('/tmp/example'), z_score=False, n_components=3).name
'mask_umap_3d.h5'
"""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from dataclasses import dataclass
from pathlib import Path
import json
import uuid

import numpy as np

from bivosia.config import ANALYSIS_DEFAULTS, CONSENSUS_MASK_DEFAULTS, CONSENSUS_MASK_SOURCES, PROCESSING_DEFAULTS
from bivosia.analysis.embedding import EmbeddingResult, embedding_dimensions, validate_embedding
from bivosia.analysis.masks import preprocessing_settings, validate_downsample_factor
from bivosia.analysis.mask_features import (
    COMPONENT_DTYPE, HOLE_DTYPE, frame_sampling, sampled_frames,
    validate_sample_indices, validate_shape_metadata, validate_zernike_data, zernike_indices,
)
from bivosia.dataset.io.arrays import read_array
from bivosia.dataset.io.containers import container_output, open_container
from bivosia.dataset.masks import ConsensusMaskReader, consensus_metadata
from bivosia.dataset.io.json import read_json
from bivosia.dataset.video import RecordingStreams


# Framewise measures and component geometry

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


# Shape descriptors and embeddings

def sampling_settings(settings=None):
    """Read frame selection from saved or supplied processing settings."""
    return frame_sampling(**{key: (settings or {}).get(key, PROCESSING_DEFAULTS['analysis']['mask_shape'][key])
                             for key in ('start_frame', 'end_frame', 'frame_step')})


def _variant_path(dataset, kind, sampling, preprocess_masks, write, downsample_factor, **parameters):
    values = sampling_settings(sampling)
    factor = validate_downsample_factor(downsample_factor)
    if factor == 1 and values == sampling_settings():
        return None
    variant = '_downsampled' if factor > 1 else '_sampled'
    return dataset.file(kind + variant, write=write, downsample_factor=factor, **{**values,
                        'end_frame': 'last' if values['end_frame'] == -1 else values['end_frame']},
                        preprocessed='.preprocessed' if preprocess_masks else '', **parameters)


def zernike_path(dataset, *, preprocess_masks=False, sampling=None, write=False,
                 downsample_factor=ANALYSIS_DEFAULTS['mask_features']['zernike']['downsample_factor']):
    """Separate descriptors by preprocessing, resolution and selected frames."""
    sampled = _variant_path(dataset, 'mask_zernike', sampling, preprocess_masks, write, downsample_factor)
    return sampled or dataset.file('mask_zernike_preprocessed' if preprocess_masks else 'mask_zernike', write=write)


def embedding_path(dataset, *, z_score, n_components=ANALYSIS_DEFAULTS['embedding']['n_components'],
                   preprocess_masks=False, sampling=None, write=False,
                   downsample_factor=ANALYSIS_DEFAULTS['mask_features']['zernike']['downsample_factor']):
    dimensions = '_3d' if embedding_dimensions(n_components) == 3 else ''
    sampled = _variant_path(dataset, 'mask_umap', sampling, preprocess_masks, write, downsample_factor,
                           dimensions=dimensions, z_score='_zscore' if z_score else '')
    if sampled is not None:
        return sampled
    key = 'mask_umap' + dimensions
    key += '_zscore' if z_score else ''
    return dataset.file(key + ('_preprocessed' if preprocess_masks else ''), write=write)


def shape_settings(settings=None):
    """Combine configured shape-analysis defaults with supplied step settings."""
    return dict(ANALYSIS_DEFAULTS['mask_features']['zernike'], **ANALYSIS_DEFAULTS['embedding'],
                **ANALYSIS_DEFAULTS['masks']['preprocessing'], radius=0,
                mask_source=CONSENSUS_MASK_DEFAULTS['source'], **{
                    key: PROCESSING_DEFAULTS['analysis']['mask_shape'][key]
                    for key in ('start_frame', 'end_frame', 'frame_step')}) | (settings or {})


@dataclass(frozen=True)
class MaskShapeDescriptors:
    """Saved descriptor rows with source frame/identity indices and provenance."""

    values: np.ndarray
    frames: np.ndarray
    identities: np.ndarray
    metadata: dict


@dataclass(frozen=True)
class MaskShapeEmbedding:
    """A saved embedding with source frame/identity indices and provenance."""

    embedding: EmbeddingResult
    frames: np.ndarray
    identities: np.ndarray
    metadata: dict


class MaskShapeResults:
    """Locate and validate saved mask-shape descriptors and embeddings."""

    def __init__(self, dataset, settings, *, source=None):
        self.dataset = dataset
        self.settings = shape_settings(settings)
        self.sampling = sampling_settings(self.settings)
        self.factor = validate_downsample_factor(self.settings['downsample_factor'])
        self.dimensions = embedding_dimensions(self.settings['n_components'])
        self.preprocessing = preprocessing_settings(self.settings['preprocess_masks'], **{
            name: self.settings[name] for name in ANALYSIS_DEFAULTS['masks']['preprocessing']})
        self._source = source

    @property
    def source(self):
        if self._source is None:
            self._source = consensus_metadata(self.dataset, source=self.settings['mask_source'])
        return self._source

    def path(self, kind, *, write=False):
        options = dict(preprocess_masks=self.settings['preprocess_masks'], sampling=self.sampling,
                       downsample_factor=self.factor, write=write)
        if kind == 'mask_zernike':
            return zernike_path(self.dataset, **options)
        if kind == 'mask_umap':
            return embedding_path(self.dataset, z_score=self.settings['z_score'],
                                  n_components=self.dimensions, **options)
        raise ValueError(f'Unknown shape result: {kind}')

    @property
    def feature_expected(self):
        return dict(self.source, degree=self.settings['degree'], radius=self.settings['radius'],
                    preprocessing=self.preprocessing, sampling=self.sampling, downsample_factor=self.factor)

    def feature_metadata(self, *, infer_parameters=False):
        expected = self.feature_expected
        if infer_parameters and self.settings['radius'] <= 0:
            expected.pop('degree')
            expected.pop('radius')
        return shape_metadata(self.path('mask_zernike'), expected)

    def embedding_expected(self, features):
        """Bind an embedding to the validated descriptor file and its parameters."""
        return dict(self.feature_expected, degree=features['degree'], radius=features['radius'],
                    feature_id=features['feature_id'], z_score=self.settings['z_score'],
                    n_components=self.dimensions)

    def validate_saved(self, kind, *, infer_parameters=False):
        features = self.feature_metadata(infer_parameters=infer_parameters)
        if kind == 'mask_zernike':
            return features
        return shape_metadata(self.path(kind), self.embedding_expected(features))


def create_zernike(output, metadata):
    """Initialize appendable descriptor rows with their frame and identity indices."""
    indices = zernike_indices(metadata['degree'])
    output.create_dataset('moments', data=indices)
    output.create_dataset('features', shape=(0, len(indices)), maxshape=(None, len(indices)), dtype='f8')
    output.create_dataset('frames', shape=(0,), maxshape=(None,), dtype='i8')
    output.create_dataset('identities', shape=(0,), maxshape=(None,), dtype='i8')
    output.attrs['metadata'] = json.dumps(dict(metadata, feature_id=uuid.uuid4().hex))


def append_zernike(output, values, frames, identities, degree):
    """Validate and append aligned feature rows without modifying source masks."""
    validate_zernike_data(values, frames, identities, degree)
    for name, data in (('features', values), ('frames', frames), ('identities', identities)):
        array = output[name]
        start = len(array)
        array.resize(start + len(data), axis=0)
        array[start:] = data


def shape_metadata(path, expected):
    """Validate source metadata without reading descriptor or embedding arrays."""
    with open_container(path) as file:
        metadata = json.loads(file.attrs['metadata'])
    validate_shape_metadata(metadata, expected)
    return metadata


def read_zernike(path, expected) -> MaskShapeDescriptors:
    """Load validated mask-shape descriptor rows and their sample indices."""
    with open_container(path) as file:
        metadata = json.loads(file.attrs['metadata'])
        validate_shape_metadata(metadata, expected)
        values, frames, identities = (file[name][:] for name in ('features', 'frames', 'identities'))
    validate_zernike_data(values, frames, identities, metadata['degree'])
    return MaskShapeDescriptors(values, frames, identities, metadata)


def write_embedding(path, result, frames, identities, metadata):
    """Save a numerical embedding together with sample indices and provenance."""
    validate_embedding(result.coordinates, result.mean, result.scale, len(frames), metadata['feature_count'],
                       n_components=metadata['n_components'])
    validate_sample_indices(frames, identities)
    with container_output(path, overwrite=False) as output:
        for name, values in (('coordinates', result.coordinates), ('mean', result.mean), ('scale', result.scale),
                             ('frames', frames), ('identities', identities)):
            output.create_dataset(name, data=values)
        output.attrs['metadata'] = json.dumps(dict(metadata, umap_parameters=result.parameters))


def read_embedding(path, expected) -> MaskShapeEmbedding:
    """Load a validated embedding, scaling parameters and source sample indices."""
    with open_container(path) as file:
        metadata = json.loads(file.attrs['metadata'])
        validate_shape_metadata(metadata, expected)
        coordinates, mean, scale = (file[name][:] for name in ('coordinates', 'mean', 'scale'))
        validate_embedding(coordinates, mean, scale, len(file['frames']), metadata['feature_count'],
                           n_components=metadata['n_components'])
        frames, identities = file['frames'][:], file['identities'][:]
        validate_sample_indices(frames, identities)
    result = EmbeddingResult(coordinates, mean, scale, metadata.get('umap_parameters', {}))
    return MaskShapeEmbedding(result, frames, identities, metadata)


def shape_analysis_inputs(dataset, settings):
    """Read matching saved coordinates and the source for the shape-analysis window."""
    results = MaskShapeResults(dataset, settings)
    source = results.source
    degree, radius = results.settings['degree'], results.settings['radius']
    embedding = None
    with ConsensusMaskReader(dataset, source=results.settings['mask_source']) as masks:
        frame_count = masks.shape[0]
    sampled_frames(frame_count, **results.sampling)
    if results.path('mask_zernike').is_file():
        metadata = results.feature_metadata(infer_parameters=True)
        degree, radius = metadata['degree'], metadata['radius']
        result = results.path('mask_umap')
        if result.is_file():
            embedding = read_embedding(result, results.embedding_expected(metadata))
    return {'animals': None, 'degree': degree, 'radius': radius,
            'coordinates': embedding.embedding.coordinates if embedding is not None else None,
            'frames': embedding.frames if embedding is not None else None,
            'identities': embedding.identities if embedding is not None else None,
            'frame_count': frame_count, 'zones': read_json(dataset.annotation)['zones'], 'source': source,
            'message': 'Mask source: ' + CONSENSUS_MASK_SOURCES[source['mask_source']]['label']}


def open_feature_video(dataset, *, source=CONSENSUS_MASK_DEFAULTS['source']):
    """Open the recording with the selected globally labelled masks when available."""
    video = dataset.video
    try:
        masks = {'preferred': ConsensusMaskReader(dataset, source=source)}
    except FileNotFoundError:
        masks = {}
    return RecordingStreams({'original': video}, masks=masks)


def sample_masks(dataset, frames, identities, *, source, expected):
    """Yield requested animal masks in frame order, keeping their supplied row indices."""
    with ConsensusMaskReader(dataset, source=source) as masks:
        validate_shape_metadata(masks.metadata, expected)
        previous, labels = None, None
        for index in np.argsort(frames, kind='stable'):
            frame = int(frames[index])
            if frame != previous:
                if not masks.valid[frame]:
                    raise ValueError(f'Frame {frame} is no longer valid')
                labels, previous = masks[frame], frame
            yield int(index), labels == identities[index]
