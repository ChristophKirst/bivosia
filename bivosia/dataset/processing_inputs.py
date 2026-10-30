"""Locate required dataset products for individual and combined processing jobs."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from bivosia.dataset.jsonio import read_json


def require_products(dataset, products):
    paths = []
    if 'consensus_masks' in products:
        from bivosia.dataset.consensus_masks import consensus_metadata
        metadata = consensus_metadata(dataset)
        paths.extend(dataset.path(item['path']) for item in metadata['source_files'])
    if 'video' in products:
        try:
            paths.append(dataset.video)
        except ValueError as error:
            raise FileNotFoundError(str(error)) from error
    if 'key_frames' in products:
        paths.extend(dataset.file(key) for key in ('key_frames', 'key_frame_masks'))
    if 'masks' in products:
        paths.extend(dataset.mask(direction) for direction in ('forward', 'reverse'))
    if 'analysis' in products:
        paths.extend(dataset.array(key) for key in ('mask_intersections', 'mask_areas', 'mask_iou_match',
                     'mask_quality', 'mask_missing', 'mask_switch', 'mask_quality_metadata'))
    if 'keypoints' in products:
        paths.append(dataset.keypoint_file())
    if {'annotations', 'export', 'consensus_masks'} & set(products):
        annotation = dataset.annotation
        if annotation is None:
            raise FileNotFoundError('Select a ZoD annotation')
        paths.append(annotation)
        if 'export' in products:
            record = read_json(annotation)
            if not record.get('export'):
                raise FileNotFoundError('Export consensus masks before calculating features')
            paths.append(dataset.export_path(record['export']['path']))
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Required input is missing: {path}')
    return paths
