"""Save cleaned key-frame copies separately from their original annotations.

Examples
--------
>>> from tempfile import TemporaryDirectory
>>> import numpy as np
>>> from bivosia.config import ANALYSIS_DEFAULTS
>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.dataset.mask_cleanup import save_cleaned_key_frame
>>> with TemporaryDirectory() as directory:
...     path = save_cleaned_key_frame(Dataset(directory), 0, np.ones((4, 4), dtype=np.uint8),
...                                   frame_count=2, animals=1,
...                                   parameters=ANALYSIS_DEFAULTS['masks']['cleanup'])
...     path.name
'key_frames.cleaned.h5'
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import json
import logging

import numpy as np

from bivosia.analysis.masks import validate_cleaned_key_frames
from bivosia.dataset.containers import container_output, open_container
from bivosia.dataset.paths import auxiliary_path


def save_cleaned_key_frame(dataset, frame, mask, *, frame_count, animals, parameters):
    """Save one cleaned frame and its parameters, retaining other cleaned frames.

    Originals and the key-frame session are unchanged. Frame indices refer to
    the source video. The cleaned file is published only after writing succeeds.
    """
    mask = np.asarray(mask)
    validate_cleaned_key_frames([frame], mask[None], frame_count, animals)
    path = dataset.file('cleaned_key_frame_masks', write=True)
    if any(path.resolve() == dataset.file(key).resolve() for key in ('key_frames', 'key_frame_masks')):
        raise ValueError('Cleaned masks must be saved separately from original key frames')
    if auxiliary_path(path, 'partial').exists():
        raise FileExistsError(f'Incomplete cleaned masks already exist: {path}')
    saved, settings = {}, {}
    if path.exists():
        with open_container(path) as source:
            frames, masks = source['frames'][:], source['masks'][:]
            validate_cleaned_key_frames(frames, masks, frame_count, animals)
            if (source.attrs['frame_count'] != frame_count or source.attrs['animals'] != animals
                    or masks.shape[1:] != mask.shape):
                raise ValueError('Existing cleaned masks belong to different video dimensions or animal counts')
            saved = {int(index): value for index, value in zip(frames, masks)}
            settings = json.loads(source.attrs['parameters'])
    saved[int(frame)] = mask
    settings[str(frame)] = dict(parameters)
    frames = sorted(saved)
    with container_output(path) as output:
        output.attrs['frame_count'] = frame_count
        output.attrs['animals'] = animals
        output.attrs['parameters'] = json.dumps(settings, allow_nan=False)
        output.create_dataset('frames', data=np.asarray(frames, dtype=np.int64))
        masks = output.create_dataset('masks', shape=(len(frames), *mask.shape), dtype=np.uint8,
                                      chunks=(1, *mask.shape), compression='lzf')
        for index, frame in enumerate(frames):
            masks[index] = saved[frame]
    logging.getLogger(__name__).info('Saved cleaned key-frame masks: %s', path)
    return path
