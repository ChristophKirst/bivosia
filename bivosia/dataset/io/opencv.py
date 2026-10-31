"""OpenCV video decoding and source metadata."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from math import isfinite
from pathlib import Path

import cv2

from bivosia.dataset.io.interfaces import FileMetadata, VideoReader


def inspect_video(path):
    """Read reported frame count and estimate duration without decoding frames."""
    video = OpenCvVideoReader(path)
    try:
        duration = (video.count / video.fps
                    if isfinite(video.fps) and video.fps > 0 else None)
        return FileMetadata(frames=video.count, duration=duration)
    finally:
        video.close()


class OpenCvVideoReader(VideoReader):
    """Read OpenCV video containers in RGB (processing) or BGR (display) order."""

    def __init__(self, path, *, color='rgb'):
        if color not in ('rgb', 'bgr'):
            raise ValueError('Choose rgb or bgr video color order')
        self.path, self.color = Path(path), color
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            self.close()
            raise ValueError(f'Could not open video: {path}')
        self.count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.shape = (int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                      int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        if self.count < 1 or min(self.shape) < 1:
            self.close()
            raise ValueError('Video has no readable dimensions or frames')

    def read(self, frame):
        frame = int(frame)
        if int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) != frame:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = self.capture.read()
        if not ok:
            raise ValueError(f'Could not read frame {frame} from {self.path} video.')
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if self.color == 'rgb' else image

    def close(self):
        self.capture.release()
