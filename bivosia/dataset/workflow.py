"""Read the files and headers available to a dataset workflow.

Examples
--------
>>> from tempfile import TemporaryDirectory
>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.dataset.workflow import inspect_workflow
>>> with TemporaryDirectory() as directory:
...     info = inspect_workflow(Dataset(directory))
...     bool(info['masks']['forward']['exists'])
False
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from bivosia.dataset.formats import file_inspector
from bivosia.dataset.jsonio import read_json
from bivosia.analysis.revision import revision_fingerprint


def _inspect(resolve):
    try:
        path = resolve()
        exists = path.is_file()
        inspector = file_inspector(path) if exists else None
        metadata = inspector(path) if inspector is not None else None
        return {'path': path, 'exists': exists, 'metadata': metadata, 'error': ''}
    except Exception as error:
        return {'path': None, 'exists': False, 'metadata': None, 'error': str(error)}


def inspect_workflow(dataset):
    """Inspect headers without reading mask pixels or claiming review completion."""
    result = {'video': _inspect(lambda: dataset.video)}
    result['key_frames'] = {name: _inspect(lambda: dataset.file(name))
                            for name in ('key_frames', 'key_frame_masks')}
    result['masks'] = {name: _inspect(lambda: dataset.mask(name))
                       for name in ('forward', 'reverse')}
    result['analysis'] = {name: _inspect(lambda: dataset.array(name)) for name in (
        'mask_intersections', 'mask_areas', 'mask_iou_match', 'mask_quality',
        'mask_missing', 'mask_switch', 'mask_quality_metadata')}
    result['features'] = {'mask_zernike': _inspect(lambda: dataset.file('mask_zernike')),
                          'mask_umap': _inspect(lambda: dataset.file('mask_umap')
                              if dataset.file('mask_umap').is_file() else dataset.file('mask_umap_zscore'))}
    result['alignment'] = {'keypoint_alignment': _inspect(lambda: dataset.array('keypoint_alignment'))}
    result['keypoints'] = _inspect(dataset.keypoint_file)
    result['animals'] = None
    metadata = result['key_frames']['key_frame_masks']['metadata']
    result['key_frame_count'] = next((shape[0] for name, shape, _ in metadata.arrays
                                    if name == 'frames' and len(shape) == 1), None) if metadata else None
    if result['key_frames']['key_frames']['exists']:
        try:
            result['animals'] = read_json(dataset.file('key_frames'))['animals']
        except Exception as error:
            result['key_frames']['key_frames']['error'] = str(error)
    try:
        result['annotations'] = dataset.annotation_files
        result['annotation_error'] = ''
    except Exception as error:
        result['annotations'], result['annotation_error'] = [], str(error)
    result['export'] = None
    result['export_error'] = ''
    result['export_current'] = False
    try:
        annotation = dataset.annotation
        saved = read_json(annotation) if annotation else None
        exported = saved.get('export') if saved else None
        if exported:
            result['export'] = _inspect(lambda: dataset.export_path(exported['path']))
            result['export_current'] = exported['fingerprint'] == revision_fingerprint(
                saved['header'], saved['zones'], saved['links'], saved.get('link_frames', []))
    except Exception as error:
        result['export_error'] = str(error)
    return result
