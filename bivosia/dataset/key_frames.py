"""Key-frame annotations, HDF5 masks and editable mask history.

Examples
--------
>>> import numpy as np
>>> from bivosia.dataset.key_frames import MaskDraft
>>> draft = MaskDraft(np.zeros((4, 4), dtype=np.uint8))
>>> draft.add_prompt(1, 2, 1)
>>> draft.dirty
True
>>> draft.undo()
>>> draft.dirty
False
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from copy import deepcopy
import logging

import numpy as np

from bivosia.analysis.key_frames import (
    add_frame, cropped_frames, fixed_frames, move_frame, paint_mask, propose_frames,
    validate_crop, validate_mask, validate_key_frames,
)
from bivosia.dataset.jsonio import atomic_json, read_json
from bivosia.dataset.containers import open_container, container_output
from bivosia.dataset.paths import file_name


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
