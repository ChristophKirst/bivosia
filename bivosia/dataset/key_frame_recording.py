"""Open a recording and its key-frame annotations with shared resource ownership.

Examples
--------
>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.dataset.key_frame_recording import KeyFrameRecording
>>> with KeyFrameRecording(Dataset('/path/to/dataset')) as recording:  # doctest: +SKIP
...     print(recording.session.frame_count)
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from contextlib import ExitStack

from bivosia.dataset.key_frames import KeyFrameSession
from bivosia.dataset.video import VideoSet


class KeyFrameRecording:
    """Close all readers together, including those added by a mask viewer."""

    def __init__(self, dataset, *, animals=2):
        self.dataset = dataset
        self._resources = ExitStack()
        try:
            self.videos = self.manage(VideoSet({'original': dataset.video}))
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
