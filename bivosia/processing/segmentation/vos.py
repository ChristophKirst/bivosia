"""Generate bidirectional masks from a dataset's saved key-frame labels.

Examples
--------
With an existing dataset, saved key frames and a local Cutie checkpoint:

>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.processing.segmentation.vos import segment
>>> dataset = Dataset("/path/to/dataset")
>>> outputs = segment(dataset, checkpoint='/path/to/cutie-base-mega.pth')  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import closing
import logging
import time

import numpy as np

from bivosia.analysis.keyframes import output_mask, output_shape, key_frame_intervals
from bivosia.config import VOS_BACKENDS
from bivosia.dataset.annotations.keyframes import KeyFrameSession
from bivosia.dataset.masks import check_output, output_files, output_paths, vos_outputs
from bivosia.dataset.video import open_video
from bivosia.segmentation.video import create_segmenter, model_availability, reference_frames
from bivosia.processing.log import Progress


def prepare(dataset, direction='both', output_scale=1.):
    """Load and validate the video, saved key frames and output locations."""
    if direction not in ('both', 'forward', 'reverse'):
        raise ValueError('Choose both, forward or reverse')
    directions = ('forward', 'reverse') if direction == 'both' else (direction,)
    with closing(open_video(dataset.video)) as video:
        key_frames = KeyFrameSession(dataset, video.count, video.shape)
        intervals = key_frame_intervals(key_frames.masks, key_frames.crop)
        shape = output_shape(video.shape, output_scale)
        for frame in sorted(set(f for interval in intervals for f in interval)):
            if not np.any(key_frames.masks[frame]):
                raise ValueError(f'Key frame at frame {frame} is empty')
            absent = sorted(set(range(1, key_frames.animals + 1)) - set(np.unique(key_frames.masks[frame])))
            if absent:
                logging.getLogger(__name__).warning('Key frame %s has no pixels for IDs %s', frame, absent)
    for path in output_paths(dataset, directions).values():
        check_output(path)
    return key_frames, intervals, directions, shape


def check(dataset, *, model=None, checkpoint=None, device='auto', parameters=None,
          direction='both', references='start', output_scale=1.):
    model = model or next(iter(VOS_BACKENDS))
    key_frames, intervals, directions, shape = prepare(dataset, direction, output_scale)
    reference_frames(sorted(key_frames.masks), *intervals[0], directions[0], references)
    selected, reason = model_availability(model, checkpoint, device, parameters)
    if reason:
        raise RuntimeError(reason)
    return {'device': selected, 'crop': key_frames.crop, 'animals': key_frames.animals,
            'intervals': len(intervals), 'shape': list(shape)}


def segment(dataset, *, model=None, checkpoint=None, device='auto', parameters=None,
            direction='both', references='start', output_scale=1., progress=None,
            cancelled=None, segmenter=None):
    model = model or next(iter(VOS_BACKENDS))
    key_frames, intervals, directions, shape = prepare(dataset, direction, output_scale)
    engine = segmenter or create_segmenter(model, checkpoint, device, parameters)
    if segmenter is None:
        device = engine.check()
    if references not in engine.reference_modes:
        raise ValueError(f'{model} does not support {references} references')
    total = len(directions) * (key_frames.crop[1] - key_frames.crop[0] + 1)
    current = {'direction': directions[0], 'interval': list(intervals[0]), 'frame': key_frames.crop[0]}
    started = time.monotonic()

    def report(done, count):
        if progress:
            progress({**current, 'completed': done, 'total': count, 'elapsed': time.monotonic() - started})

    tracker = Progress('VOS', total, callback=report, cancelled=cancelled)
    selected_key_frames = sorted(f for f in key_frames.masks if key_frames.crop[0] <= f <= key_frames.crop[1])
    try:
        with closing(open_video(dataset.video)) as video, vos_outputs(
                dataset, directions, key_frames.frame_count, shape, key_frames.crop) as outputs:
            for source in directions:
                masks, valid = outputs[source]
                for start, end in intervals:
                    tracker.check_cancelled()
                    current.update(direction=source, interval=[start, end])
                    indices = reference_frames(selected_key_frames, start, end, source, references)
                    refs = {frame: (video.read(frame), key_frames.masks[frame]) for frame in indices}
                    expected = iter(range(start, end + 1) if source == 'forward' else range(end, start - 1, -1))
                    frames = video.frames(start, end, source, cancelled)
                    for frame, mask in engine.propagate(frames, refs, direction=source,
                                                        animals=key_frames.animals, cancelled=cancelled):
                        tracker.check_cancelled()
                        if frame != next(expected, None):
                            raise ValueError('VOS returned an unexpected source frame')
                        mask = key_frames.masks.get(frame, mask)
                        masks[frame] = output_mask(mask, video.shape, shape, key_frames.animals)
                        current['frame'] = frame
                        if not valid[frame]:
                            valid[frame] = True
                            tracker.update(1)
                    if next(expected, None) is not None:
                        raise ValueError('VOS stopped before the interval ended')
            tracker.check_cancelled()
        tracker.cancelled = None  # Publication has completed; a late Stop cannot undo it.
        tracker.finish()
        return [dataset.relative(path) for output in output_paths(dataset, directions).values()
                for path in output_files(output)]
    finally:
        engine.close()
