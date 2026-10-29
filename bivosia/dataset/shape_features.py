"""Consensus-mask sources, Zernike feature files and saved UMAP embeddings.

Examples
--------
>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.dataset.shape_features import embedding_path
>>> embedding_path(Dataset('/tmp/example'), z_score=True).name
'mask_umap_zscore.h5'
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import json
import uuid

import numpy as np

from bivosia.config import ANALYSIS_DEFAULTS
from bivosia.analysis.embedding import validate_embedding
from bivosia.analysis.mask_preprocessing import preprocessing_settings
from bivosia.analysis.shape_features import (
    validate_consensus_export, validate_sample_indices, validate_shape_metadata,
    validate_zernike_data, zernike_indices,
)
from bivosia.dataset.containers import container_output, open_container
from bivosia.dataset.jsonio import read_json


def consensus_source(dataset):
    """Resolve and validate the selected annotation's exported consensus masks."""
    annotation = dataset.annotation
    if annotation is None:
        raise ValueError('Select a reviewed annotation and export its consensus masks first')
    record = read_json(annotation)
    if not record.get('export'):
        raise ValueError('Export consensus masks before running shape analysis')
    path = dataset.export_path(record['export']['path'])
    with open_container(path) as file:
        fingerprint = file.attrs.get('annotation_fingerprint')
        validate_consensus_export(record, fingerprint)
    stat = path.stat()
    metadata = {'source': dataset.relative(path), 'source_size': stat.st_size,
                'source_mtime_ns': stat.st_mtime_ns, 'annotation': dataset.relative(annotation),
                'annotation_fingerprint': fingerprint}
    return path, metadata


def zernike_path(dataset, *, preprocess_masks=False, write=False):
    """Keep descriptors of original and preprocessed masks in separate files."""
    return dataset.file('mask_zernike_preprocessed' if preprocess_masks else 'mask_zernike', write=write)


def embedding_path(dataset, *, z_score, preprocess_masks=False, write=False):
    key = 'mask_umap_zscore' if z_score else 'mask_umap'
    return dataset.file(key + ('_preprocessed' if preprocess_masks else ''), write=write)


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


def zernike_metadata(path, expected):
    with open_container(path) as file:
        metadata = json.loads(file.attrs['metadata'])
    validate_shape_metadata(metadata, expected)
    return metadata


def read_zernike(path, expected):
    with open_container(path) as file:
        metadata = json.loads(file.attrs['metadata'])
        validate_shape_metadata(metadata, expected)
        values, frames, identities = (file[name][:] for name in ('features', 'frames', 'identities'))
    validate_zernike_data(values, frames, identities, metadata['degree'])
    return values, frames, identities, metadata


def write_embedding(path, coordinates, mean, scale, frames, identities, metadata):
    validate_embedding(coordinates, mean, scale, len(frames), metadata['feature_count'])
    validate_sample_indices(frames, identities)
    with container_output(path, overwrite=False) as output:
        for name, values in (('coordinates', coordinates), ('mean', mean), ('scale', scale),
                             ('frames', frames), ('identities', identities)):
            output.create_dataset(name, data=values)
        output.attrs['metadata'] = json.dumps(metadata)


def read_embedding(path, expected):
    with open_container(path) as file:
        metadata = json.loads(file.attrs['metadata'])
        validate_shape_metadata(metadata, expected)
        coordinates, mean, scale = (file[name][:] for name in ('coordinates', 'mean', 'scale'))
        validate_embedding(coordinates, mean, scale, len(file['frames']), metadata['feature_count'])
        validate_sample_indices(file['frames'][:], file['identities'][:])
    return coordinates


def shape_analysis_inputs(dataset, settings):
    """Read matching saved coordinates and the source for the shape-analysis window."""
    path, source = consensus_source(dataset)
    preprocess = settings['preprocess_masks']
    preprocessing = preprocessing_settings(preprocess, **{
        name: settings[name] for name in ANALYSIS_DEFAULTS['mask_preprocessing']})
    expected = dict(source, preprocessing=preprocessing)
    features = zernike_path(dataset, preprocess_masks=preprocess)
    degree, radius = settings['degree'], settings['radius']
    coordinates = None
    if features.is_file():
        metadata = zernike_metadata(features, expected)
        if radius <= 0:
            degree, radius = metadata['degree'], metadata['radius']
        validate_shape_metadata(metadata, {'degree': degree, 'radius': radius})
        result = embedding_path(dataset, z_score=settings['z_score'], preprocess_masks=preprocess)
        if result.is_file():
            coordinates = read_embedding(result, {**expected, 'feature_id': metadata['feature_id'],
                'degree': degree, 'radius': radius, 'z_score': settings['z_score']})
    return {'animals': None, 'degree': degree, 'radius': radius, 'coordinates': coordinates,
            'message': f'Consensus masks: {path.name}'}
