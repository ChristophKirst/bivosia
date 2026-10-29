"""Plot descriptions for interval rows, series, legends and cursors."""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import numpy as np


from dataclasses import dataclass, field


@dataclass(frozen=True)
class IntervalBand:
    """Colored subrow with optional frame insets and vertical edge borders."""

    color: str
    offset: float = 0.0
    height: float = 1.0
    border_color: str | None = None
    frame_inset: float = 0.0  # removed from each end of the inclusive frame cells


@dataclass(frozen=True)
class IntervalSpec:
    key: str
    start: int
    end: int
    label: str = ""
    color: str | None = None
    bands: tuple[IntervalBand, ...] = ()
    payload: Any = None

    def __post_init__(self):
        if self.start < 0 or self.end < self.start:
            raise ValueError("Intervals must have nonnegative inclusive start <= end")
        if any(b.offset < 0 or b.height <= 0 or b.offset + b.height > 1.000001 for b in self.bands):
            raise ValueError("Interval bands must fit inside the row")
        if any(not 0 <= 2 * b.frame_inset < self.end - self.start + 1 for b in self.bands):
            raise ValueError("Interval band insets must leave a positive frame span")


@dataclass(frozen=True)
class IntervalTrack:
    key: str
    label: str
    intervals: tuple[IntervalSpec, ...] = ()
    color: str = "#aaaaaa"
    opacity: float = 1.0
    visible: bool = True


@dataclass(frozen=True)
class IntervalHit:
    row_key: str
    track_key: str
    interval: IntervalSpec
    frame: int

    @property
    def selection(self) -> tuple[str, str, str]:
        return self.row_key, self.track_key, self.interval.key


@dataclass(frozen=True)
class IntervalEdit:
    """A request only; the application/model validates and commits it."""

    hit: IntervalHit
    action: str  # start, end, or move
    frame: int  # new inclusive endpoint; for move, requested new start


@dataclass(frozen=True)
class LegendGroup:
    """One optional topic and its color keys; e.g. direction: [blue] F [red] R."""

    label: str = ""
    items: tuple[tuple[str, str], ...] = ()  # (label, color)


@dataclass(frozen=True)
class MarkerSpec:
    key: str
    frame: float
    color: str = "#eeeeee"
    payload: Any = None


@dataclass(frozen=True)
class SplitBoundary:
    """An internal split before frame, bounded by its neighbouring pieces."""

    key: str
    frame: int
    minimum: int
    maximum: int
    movable: bool = True

    def __post_init__(self):
        if not 0 <= self.minimum <= self.frame <= self.maximum:
            raise ValueError('Split boundary is outside its permitted range')


@dataclass(frozen=True)
class RowSpec:
    key: str
    label: str
    tracks: tuple[IntervalTrack, ...] = ()
    height: float = 1.0  # relative to other visible rows
    visible: bool = True
    selectable: bool = False
    draggable: bool = False
    editable: bool = False
    edit_requires_alt: bool = False
    double_click: bool = True
    show_interval_labels: bool = False
    legend: tuple[LegendGroup, ...] | None = None  # None uses tracks; () hides the legend
    legend_title: str | None = None
    markers: tuple[MarkerSpec, ...] = ()
    on_marker_double_click: Callable[[MarkerSpec], None] | None = field(default=None, compare=False)
    on_select: Callable[[IntervalHit], None] | None = field(default=None, compare=False)
    on_double_click: Callable[[IntervalHit], None] | None = field(default=None, compare=False)
    on_edit: Callable[[IntervalEdit], None] | None = field(default=None, compare=False)
    boundaries: tuple[SplitBoundary, ...] = ()
    on_split: Callable[[IntervalHit], None] | None = field(default=None, compare=False)
    on_boundary_move: Callable[[str, int], None] | None = field(default=None, compare=False)
    on_boundary_activate: Callable[[str], None] | None = field(default=None, compare=False)

    def __post_init__(self):
        if self.height <= 0:
            raise ValueError("Row height must be positive")


@dataclass(frozen=True)
class PlotInteractions:
    seek: bool = True
    pan: bool = True
    zoom: bool = True
    double_click: bool = True
    cursor_drag: bool = True
    y_pan: bool = True
    y_zoom: bool = True
    y_reset: bool = True


@dataclass(frozen=True)
class CursorSpec:
    key: str
    frame: int
    color: str = "#ff3b4f"
    label: str = ""
    visible: bool = True
    draggable: bool = True
    line: bool = True
    level: int = 0  # marker row above the plot
    dot: bool = True
    owner: str | None = None  # playback cursor owning a loop handle
    outline: str | None = None
    selectable: bool = False


@dataclass(frozen=True)
class SeriesSpec:
    key: str
    label: str
    values: np.ndarray
    color: str = "#b8bec7"
    frames: np.ndarray | None = None
    minimum: np.ndarray | None = None
    maximum: np.ndarray | None = None
    value_range: tuple[float, float] | None = None
    dashed: bool = False
    visible: bool = True
    show_envelope: bool = True
    temporal_extrema: bool = False
    show_infinity: bool = False
    width: float = 1.5


@dataclass(frozen=True)
class SeriesGroup:
    """Series controlled by one legend checkbox."""

    key: str
    label: str
    series_keys: tuple[str, ...]


@dataclass(frozen=True)
class ReferenceLine:
    label: str
    value: float
    color: str = "#aab2bd"
    series_key: str | None = None  # None uses the shared y-axis
    dashed: bool = True
    visible: bool = True
    label_side: str = "right"
