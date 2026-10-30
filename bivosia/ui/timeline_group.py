"""Coordinate overview/detail navigation and shared timeline geometry."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from PySide6.QtCore import QObject


class TimelineGroup(QObject):
    """Keep a set of detail plots aligned with one overview's selected range."""

    def __init__(self, overview, details, *, aligned=None, content_heights=None):
        super().__init__(overview)
        self.overview = overview
        self.details = tuple(details)
        self.aligned = (overview, *(self.details if aligned is None else aligned))
        self.content_heights = content_heights or {}
        for plot in self.details:
            plot.viewChanged.connect(self.set_view)
        overview.windowMoved.connect(self.set_detail_view)

    def set_view(self, start, end):
        self.overview.set_view(start, end)
        self.set_detail_view(start, end)

    def set_detail_view(self, start, end):
        for plot in self.details:
            plot.set_view(start, end)

    def align_axes(self, *, current=False):
        width = max(plot.left_margin if current else plot.required_left_margin() for plot in self.aligned)
        for plot in self.aligned:
            plot.set_left_margin(width)

    def refresh_geometry(self):
        self.align_axes()
        for plot, minimum in self.content_heights.items():
            plot.setMinimumHeight(max(minimum, plot.minimum_content_height()))
        for plot in (self.overview, *self.details):
            plot.updateGeometry()
