"""Key-frame annotations, recording resources, mask history and cleaned copies.

Examples
--------
>>> import numpy as np
>>> from bivosia.dataset.annotations.keyframes import MaskDraft
>>> draft = MaskDraft(np.zeros((4, 4), dtype=np.uint8))
>>> draft.add_prompt(1, 2, 1)
>>> draft.dirty
True
>>> draft.undo()
>>> draft.dirty
False
"""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import ExitStack
from copy import deepcopy
import json
import logging

import numpy as np

from bivosia.analysis.keyframes import (
    add_frame, cropped_frames, fixed_frames, move_frame, paint_mask, propose_frames,
    validate_crop, validate_mask, validate_key_frames,
)
from bivosia.analysis.masks import validate_cleaned_key_frames
from bivosia.dataset.io.containers import open_container, container_output
from bivosia.dataset.io.json import atomic_json, read_json
from bivosia.dataset.paths import auxiliary_path, file_name
from bivosia.dataset.video import RecordingStreams


class KeyFrameSession:
    """One dataset's selected frames and saved labels, including cropped-out labels."""

    def __init__(self, dataset, frame_count, shape, animals=2):
        self.dataset, self.frame_count, self.shape = dataset, frame_count, tuple(shape)
        self.animals = animals
        self.crop = [0, frame_count - 1]
        self.frames = list(self.crop)
        self.masks, self.prompts = {}, {}
        self.model = None
        self.dirty = False
        json_path = dataset.file("key_frames")
        mask_path = dataset.file("key_frame_masks")
        if mask_path.is_file() and not json_path.is_file():
            raise ValueError(f"Key-frame masks exist without {file_name('key_frames')}; restore the annotation before editing")
        if json_path.is_file():
            if not mask_path.is_file():
                raise ValueError(f"Missing {file_name('key_frame_masks')}; restore the mask file before editing")
            data = read_json(json_path)
            self.crop, self.frames, self.animals = data["crop"], data["frames"], data["animals"]
            self.prompts = {int(k): v for k, v in data.get("prompts", {}).items()}
            self.model = data.get("model")
            with open_container(mask_path) as source:
                frames, masks = source["frames"], source["masks"]
                if frames.ndim != 1 or masks.shape != (len(frames), *self.shape):
                    raise ValueError("Key-frame mask dimensions differ from the video")
                if len(np.unique(frames)) != len(frames):
                    raise ValueError("Duplicate frames in key-frame masks")
                self.masks = {int(frame): mask for frame, mask in zip(frames, masks)}
        self.validate()

    @property
    def fixed_frames(self):
        return fixed_frames(self.crop, self.masks)

    def validate(self):
        validate_key_frames(self.crop, self.frames, self.masks, self.prompts,
                            self.shape, self.frame_count, self.animals)

    def propose(self, count):
        self.frames = propose_frames(*self.crop, count, self.masks)
        self.dirty = True

    def add(self, frame):
        frames = add_frame(self.frames, frame, self.crop)
        self.dirty |= frames != self.frames
        self.frames = frames

    def move(self, source, target):
        self.frames = move_frame(self.frames, source, target, self.crop, self.masks)
        self.dirty |= source != target

    def set_crop(self, start, end):
        validate_crop(start, end, self.frame_count)
        self.crop = [start, end]
        self.frames = cropped_frames(self.frames, self.crop, self.masks)
        self.dirty = True

    def set_animals(self, animals):
        validate_key_frames(self.crop, self.frames, self.masks, self.prompts,
                            self.shape, self.frame_count, animals)
        self.animals = animals
        self.dirty = True

    def draft(self, frame):
        return MaskDraft(self.masks.get(frame, np.zeros(self.shape, dtype=np.uint8)),
                         self.prompts.get(frame, []))

    def label(self, frame, draft):
        if frame not in self.frames:
            raise ValueError("Frame is not selected for labeling")
        self.masks[frame] = validate_mask(draft.mask, self.shape, self.animals)
        self.prompts[frame] = deepcopy(draft.prompts)
        self.dirty = True

    def save_frame(self, frame, draft, model=None):
        """Save an edited frame, restoring in-memory state if persistence fails."""
        old_mask, old_prompts = self.masks.get(frame), self.prompts.get(frame)
        old_model, old_dirty = self.model, self.dirty
        try:
            self.label(frame, draft)
            if model is not None:
                self.model = model
            self.save()
        except Exception:
            for collection, value in ((self.masks, old_mask), (self.prompts, old_prompts)):
                if value is None:
                    collection.pop(frame, None)
                else:
                    collection[frame] = value
            self.model, self.dirty = old_model, old_dirty
            raise

    def save(self):
        self.validate()
        json_path = self.dataset.file("key_frames", write=True)
        mask_path = self.dataset.file("key_frame_masks", write=True)
        payload = {"crop": self.crop, "animals": self.animals, "frames": self.frames,
                   "prompts": {str(k): v for k, v in sorted(self.prompts.items()) if v}}
        if self.model is not None:
            payload["model"] = self.model
        frames = sorted(self.masks)
        previous = read_json(json_path) if json_path.is_file() else None
        metadata_written = False
        try:
            with container_output(mask_path) as output:
                output.create_dataset("frames", data=np.asarray(frames, dtype=np.int64))
                masks = output.create_dataset("masks", shape=(len(frames), *self.shape),
                                              maxshape=(None, *self.shape), dtype=np.uint8,
                                              chunks=(1, *self.shape), compression="lzf")
                for index, frame in enumerate(frames):
                    masks[index] = self.masks[frame]
                output.flush()
                atomic_json(json_path, payload)
                metadata_written = True
        except Exception:
            if metadata_written:
                if previous is not None:
                    atomic_json(json_path, previous)
                else:
                    json_path.unlink()
            raise
        self.dirty = False
        logging.getLogger(__name__).info("Saved key frames: %s labeled frames", len(frames))


class MaskDraft:
    """An editable frame with bounded in-memory undo/redo, one step per stroke."""

    def __init__(self, mask, prompts=()):
        self.mask, self.prompts = mask.copy(), deepcopy(list(prompts))
        self.undo_stack, self.redo_stack = [], []
        self.mark_saved()

    def snapshot(self):
        return self.mask.copy(), deepcopy(self.prompts)

    def mark_saved(self):
        self.saved = self.snapshot()

    @property
    def dirty(self):
        return self.prompts != self.saved[1] or not np.array_equal(self.mask, self.saved[0])

    def checkpoint(self):
        self.undo_stack.append(self.snapshot())
        self.undo_stack = self.undo_stack[-20:]
        self.redo_stack.clear()

    def add_prompt(self, x, y, animal):
        self.checkpoint()
        self.prompts.append([float(x), float(y), int(animal)])

    def clear_prompts(self):
        self.checkpoint()
        self.prompts = []

    def replace_mask(self, mask):
        """Apply an approved label image as one undoable edit."""
        mask = validate_mask(mask, self.mask.shape, 255)
        self.checkpoint()
        self.mask = mask

    def paint(self, start, end, radius, animal):
        self.mask = paint_mask(self.mask, start, end, radius, animal)

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.snapshot())
            self.mask, self.prompts = self.undo_stack.pop()

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.snapshot())
            self.mask, self.prompts = self.redo_stack.pop()


class KeyFrameRecording:
    """Close all readers together, including those added by a mask viewer."""

    def __init__(self, dataset, *, animals=2):
        self.dataset = dataset
        self._resources = ExitStack()
        try:
            self.videos = self.manage(RecordingStreams({'original': dataset.video}))
            image = self.videos.read('original', 0)
            self.session = KeyFrameSession(dataset, self.videos.frame_count, image.shape[:2], animals)
        except BaseException:
            self.close()
            raise

    def manage(self, reader):
        """Register a reader for cleanup and return it to the caller."""
        self._resources.callback(reader.close)
        return reader

    def close(self):
        self._resources.close()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()


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
