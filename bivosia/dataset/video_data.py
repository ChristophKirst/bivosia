"""Frame-indexed video interface with bounded reverse traversal."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from abc import ABC, abstractmethod


class VideoData(ABC):
    """Decoded frames with source frame counts, image dimensions and frame rate."""

    count: int
    shape: tuple[int, int]
    fps: float

    @abstractmethod
    def read(self, frame):
        """Read one frame in the color order selected when opening the reader."""

    @abstractmethod
    def close(self):
        """Release decoder resources."""

    def frames(self, start, end, direction, cancelled=None, buffer_bytes=128 * 1024**2):
        """Yield inclusive source-index/image pairs, without re-encoding the video."""
        if direction not in ('forward', 'reverse') or not 0 <= start <= end < self.count:
            raise ValueError('Invalid video interval or direction')
        step = max(1, buffer_bytes // (self.shape[0] * self.shape[1] * 3))
        if direction == 'forward':
            for frame in range(start, end + 1):
                if cancelled and cancelled():
                    raise InterruptedError('Video reading cancelled')
                yield frame, self.read(frame)
        else:
            last = end
            while last >= start:
                first = max(start, last - step + 1)
                block = []
                for frame in range(first, last + 1):
                    if cancelled and cancelled():
                        raise InterruptedError('Video reading cancelled')
                    block.append((frame, self.read(frame)))
                yield from reversed(block)
                last = first - 1

