"""Shared read-only video and integer-mask readers for datasets."""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from threading import Lock

import numpy as np

from bivosia.dataset.paths import folder_suffixes
from bivosia.dataset.masks import open_masks
from bivosia.dataset.formats import format_handler


def open_video(path, *, color='rgb', format=None):
    """Open a registered video reader with an explicit pixel color order."""
    return format_handler('video', path, format=format)(path, color=color)


class VideoBuffer:
    """One decoder thread with bounded prefetch and decoded-frame history."""

    capacity = 12
    cache_bytes = 128 * 1024**2
    reverse_batch_frames = 64

    def __init__(self, reader):
        self.reader = reader
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video")
        self.pending = {}
        self.cache = OrderedDict()
        self.cached_bytes = 0
        self.frame_bytes = 0
        self.cache_lock = Lock()
        self.last_read = None

    def _cached(self, frame):
        with self.cache_lock:
            image = self.cache.get(frame)
            if image is not None:
                self.cache.move_to_end(frame)
            return image

    def _remember(self, frame, image):
        with self.cache_lock:
            previous = self.cache.pop(frame, None)
            if previous is not None:
                self.cached_bytes -= previous.nbytes
            self.frame_bytes = image.nbytes
            if image.nbytes <= self.cache_bytes:
                self.cache[frame] = image
                self.cached_bytes += image.nbytes
            while self.cached_bytes > self.cache_bytes:
                _, removed = self.cache.popitem(last=False)
                self.cached_bytes -= removed.nbytes

    def _decode(self, frame, backward):
        image = self._cached(frame)
        if image is not None:
            return image
        count = min(self.reverse_batch_frames, max(1, self.cache_bytes // self.frame_bytes)) if backward and self.frame_bytes else 1
        # Compressed video decodes forward; reverse stepping reuses a batch.
        for index in range(max(0, frame - count + 1), frame + 1):
            image = self.reader.read(index)
            self._remember(index, image)
        return image

    def _request(self, frame, backward=False):
        return self.executor.submit(self._decode, frame, backward)

    def prefetch(self, frames):
        frames = list(dict.fromkeys(islice(frames, self.capacity)))
        for frame in tuple(self.pending):
            if frame not in frames:
                self.pending.pop(frame).cancel()
        for frame in frames:
            if frame not in self.pending:
                self.pending[frame] = self._request(frame)

    def read(self, frame):
        future = self.pending.pop(frame, None)
        image = self._cached(frame)
        if image is not None:
            if future is not None:
                future.cancel()
            if self.last_read is not None and frame != self.last_read + 1:
                self.prefetch(())
            self.last_read = frame
            return image
        if future is None:
            self.prefetch(())
            future = self._request(frame, backward=self.last_read is not None and frame == self.last_read - 1)
        image = future.result()
        self.last_read = frame
        return image

    def close(self):
        self.prefetch(())
        self.executor.shutdown(wait=True, cancel_futures=True)
        with self.cache_lock:
            self.cache.clear()
            self.cached_bytes = 0


class VideoSet:
    def __init__(self, paths: dict[str, Path], *, masks=None) -> None:
        self.masks = dict(masks or {})
        self.buffers = {}
        self.readers = {}
        try:
            for name, path in paths.items():
                if name in self.masks:
                    continue
                if path.suffix.lower() in folder_suffixes('masks'):
                    self.masks[name] = open_masks(path, require_valid=False, direction=name)
                else:
                    self.readers[name] = open_video(path, color='bgr')
        except BaseException:
            self.close()
            raise
        self.label_masks = set(self.masks)
        counts = {name: reader.count for name, reader in self.readers.items()}
        counts.update({name: int(masks.shape[0]) for name, masks in self.masks.items()})
        self.frame_count = min(counts.values())
        if self.frame_count < 1:
            self.close()
            raise ValueError("One or more videos report no frames.")
        self.counts = counts
        original = self.readers.get("original")
        fps = original.fps if original is not None else float("nan")
        self.source_fps = float(fps) if np.isfinite(fps) and fps > 0 else None

    def read(self, name: str, frame_number: int):
        if name in self.masks:
            return np.asarray(self.masks[name][frame_number])
        return self._buffer(name).read(frame_number)

    def _buffer(self, name):
        if name not in self.buffers:
            self.buffers[name] = VideoBuffer(self.readers[name])
        return self.buffers[name]

    def prefetch(self, name, frames):
        self._buffer(name).prefetch(frames)

    def stop_prefetch(self):
        for buffer in self.buffers.values():
            buffer.prefetch(())

    def close(self) -> None:
        for buffer in self.buffers.values():
            buffer.close()
        self.buffers.clear()
        for reader in self.readers.values():
            reader.close()
        self.readers.clear()
        for masks in self.masks.values():
            masks.close()
        self.masks.clear()
