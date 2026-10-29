"""Interactive time-series plots with shared legends and display downsampling."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from dataclasses import dataclass
from html import escape
import math

from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QSizePolicy, QToolButton, QVBoxLayout, QWidget
import cv2
import numpy as np

from bivosia.ui.legend_panel import LegendPanel
from bivosia.ui.plot_models import SeriesGroup
from bivosia.ui.timeline_base import TimelineBase


@dataclass(frozen=True)
class CurveLevel:
    """One precomputed temporal resolution for a contiguous curve run."""

    factor: int
    mean: np.ndarray
    minimum: np.ndarray | None
    maximum: np.ndarray | None
    temporal_minimum: np.ndarray | None = None
    temporal_maximum: np.ndarray | None = None


@dataclass(frozen=True)
class CurveRun:
    """A contiguous frame run and all of its precomputed resolution levels."""

    first_frame: int
    frame_count: int
    levels: tuple[CurveLevel, ...]


class GraphTimeline(TimelineBase):
    """Named series on independent or shared scales, with optional interaction."""

    seriesVisibilityChanged = Signal(str, bool)
    seriesChanged = Signal()

    def __init__(self, frame_count, title="quality", *, series=(), interactions=None):
        super().__init__(frame_count, interactions=interactions)
        self.title = title
        self.series = {}
        self.visible_series = set()
        self.metric_ranges = {}
        self._data_ranges = {}
        self._curve_pyramids = {}
        self.reference_lines = ()
        self.reference_labels_visible = False
        self.envelope_visible = True
        self.shared_y_range = None
        self._default_y_range = None
        self._y_drag_start = None
        self.set_series(series)

    def set_series(self, series):
        specs = tuple(series)
        if len({s.key for s in specs}) != len(specs):
            raise ValueError("Series keys must be unique")
        ranges, pyramids, infinity_frames = {}, {}, {}
        for spec in specs:
            values = np.asarray(spec.values)
            frames = np.arange(len(values)) if spec.frames is None else np.asarray(spec.frames)
            if values.ndim != 1 or frames.shape != values.shape:
                raise ValueError(f"Series {spec.key}: frames and values must be 1-D and aligned")
            if spec.frames is None and len(values) != self.frame_count:
                raise ValueError(f"Series {spec.key}: expected {self.frame_count} frame samples")
            if not np.all(np.isfinite(frames)) or np.any(frames != np.floor(frames)) or np.any(np.diff(frames) <= 0):
                raise ValueError(f"Series {spec.key}: frames must be increasing integers")
            if frames.size and (frames[0] < 0 or frames[-1] >= self.frame_count):
                raise ValueError(f"Series {spec.key}: frame outside video")
            infinity_frames[spec.key] = frames[np.isposinf(values)] if spec.show_infinity else np.empty(0)
            if (spec.minimum is None) != (spec.maximum is None):
                raise ValueError("Both envelope bounds must be supplied together")
            bounds = [values]
            for bound in (spec.minimum, spec.maximum):
                if bound is not None:
                    if np.shape(bound) != values.shape:
                        raise ValueError(f"Series {spec.key}: envelope shape mismatch")
                    bounds.append(np.asarray(bound))
            finite = [a[np.isfinite(a)] for a in bounds]
            finite = [a for a in finite if a.size]
            ranges[spec.key] = spec.value_range or (
                (min(float(a.min()) for a in finite), max(float(a.max()) for a in finite)) if finite else (0., 1.)
            )
            low, high = ranges[spec.key]
            if not np.isfinite(low) or not np.isfinite(high) or high < low:
                raise ValueError(f"Series {spec.key}: invalid value range")
            # The caller may switch between cached feature configurations.
            if self.series.get(spec.key) is spec:
                pyramids[spec.key] = self._curve_pyramids[spec.key]
                continue
            valid = np.isfinite(values)
            if spec.minimum is not None:
                valid &= np.isfinite(spec.minimum) & np.isfinite(spec.maximum)
            indices = np.flatnonzero(valid)
            runs = []
            if indices.size:
                breaks = np.flatnonzero(np.diff(frames[indices]) > 1) + 1
                for run_indices in np.split(indices, breaks):
                    mean = np.asarray(values[run_indices], dtype=np.float32)
                    levels = [CurveLevel(
                        1, mean,
                        None if spec.minimum is None else np.asarray(spec.minimum)[run_indices].astype(np.float32),
                        None if spec.maximum is None else np.asarray(spec.maximum)[run_indices].astype(np.float32),
                        mean if spec.temporal_extrema else None,
                        mean if spec.temporal_extrema else None,
                    )]
                    while len(levels[-1].mean) > 2:
                        levels.append(self._reduce_curve_level(levels[-1]))
                    runs.append(CurveRun(int(frames[run_indices[0]]), len(run_indices), tuple(levels)))
            pyramids[spec.key] = tuple(runs)
        self.series = {s.key: s for s in specs}
        self.visible_series = {s.key for s in specs if s.visible}
        self._data_ranges = ranges
        self.metric_ranges = dict(ranges)
        self._curve_pyramids = pyramids
        self._infinity_frames = infinity_frames
        self.shared_y_range = self._default_y_range = None
        self.invalidate()
        self.seriesChanged.emit()

    def set_visible_series(self, keys):
        keys = set(keys)
        if not keys <= self.series.keys():
            raise KeyError(f"Unknown series: {keys - self.series.keys()}")
        changed = self.visible_series ^ keys
        self.visible_series = keys
        self.invalidate()
        for key in changed:
            self.seriesVisibilityChanged.emit(key, key in keys)

    def set_series_visible(self, key, visible):
        keys = self.visible_series | {key} if visible else self.visible_series - {key}
        self.set_visible_series(keys)

    def set_envelope_visible(self, visible):
        self.envelope_visible = bool(visible)
        self.invalidate()

    def set_reference_lines(self, lines):
        self.reference_lines = tuple(lines)
        self.invalidate()

    def fit_shared_y_range(self, *, include=(0.,)):
        ranges = [self._data_ranges[k] for k in self.visible_series]
        low = min([v[0] for v in ranges] + list(include) or [0.])
        high = max([v[1] for v in ranges] + list(include) or [1.])
        margin = .04 * max(1e-6, high - low)
        self._default_y_range = (low - margin, high + margin)
        self.reset_y_range()

    def set_shared_y_range(self, low, high):
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError("Y-axis bounds must be finite and increasing")
        self.shared_y_range = (float(low), float(high))
        self.metric_ranges = {key: self.shared_y_range for key in self.series}
        self.invalidate()

    def reset_y_range(self):
        if self._default_y_range is not None:
            self.set_shared_y_range(*self._default_y_range)

    def _quality_axis_tick_values(self):
        if self.shared_y_range is None:
            return []
        low, high = self.shared_y_range
        pixels_per_unit = max(1., self._metric_rect().height() - 10) / (high - low)
        separation = max(18, self.fontMetrics().height() + 4)
        anchors = [value for value in (-1., 0., 1.) if low <= value <= high]
        values = [
            endpoint for endpoint in (low, high)
            if all(abs(endpoint - anchor) * pixels_per_unit >= separation for anchor in anchors)
        ]
        values.extend(anchors)
        return sorted(set(values))

    @staticmethod
    def _format_axis_value(value):
        """Format compact axis values while retaining one decimal for integers."""
        text = f"{value:.3g}"
        return text if "." in text or "e" in text.lower() else f"{text}.0"

    def required_left_axis_width(self):
        return max((
            self.fontMetrics().horizontalAdvance(self._format_axis_value(value))
            for value in self._quality_axis_tick_values()
        ), default=0) + 18

    def _metric_rect(self):
        return self._plot_rect()

    @staticmethod
    def _reduce_curve_level(level: CurveLevel) -> CurveLevel:
        """Build the next 2x-coarser level using vectorized pair reductions."""
        count = len(level.mean)
        padded_count = count + count % 2

        def padded(values: np.ndarray, fill: float) -> np.ndarray:
            result = np.full(padded_count, fill, dtype=np.float32)
            result[:count] = values
            return result.reshape(-1, 2)

        means = padded(level.mean, np.nan)
        finite_means = np.isfinite(means)
        valid_count = np.sum(finite_means, axis=1)
        mean = np.divide(
            np.nansum(means, axis=1), valid_count,
            out=np.full(len(means), np.nan, dtype=np.float32),
            where=valid_count > 0,
        )
        minimum = maximum = None
        if level.minimum is not None and level.maximum is not None:
            lows = padded(level.minimum, np.inf)
            highs = padded(level.maximum, -np.inf)
            minimum = np.min(lows, axis=1)
            maximum = np.max(highs, axis=1)
            minimum[~np.isfinite(minimum)] = np.nan
            maximum[~np.isfinite(maximum)] = np.nan
        temporal_minimum = temporal_maximum = None
        if (
            level.temporal_minimum is not None
            and level.temporal_maximum is not None
        ):
            temporal_lows = padded(level.temporal_minimum, np.inf)
            temporal_highs = padded(level.temporal_maximum, -np.inf)
            temporal_minimum = np.min(temporal_lows, axis=1)
            temporal_maximum = np.max(temporal_highs, axis=1)
            temporal_minimum[~np.isfinite(temporal_minimum)] = np.nan
            temporal_maximum[~np.isfinite(temporal_maximum)] = np.nan
        return CurveLevel(
            level.factor * 2,
            mean,
            minimum,
            maximum,
            temporal_minimum,
            temporal_maximum,
        )

    def _visible_curve_groups(
        self, key: str, first: int, last: int, max_points: int
    ) -> list[
        tuple[
            int,
            np.ndarray,
            np.ndarray,
            np.ndarray | None,
            np.ndarray | None,
            np.ndarray | None,
            np.ndarray | None,
        ]
    ]:
        """Select and crop a precomputed level; no aggregation occurs here."""
        runs = self._curve_pyramids.get(key, ())
        visible_span = max(1, last - first + 1)
        factor_needed = max(1.0, visible_span / max(1, max_points))
        level_index = max(0, int(math.ceil(math.log2(factor_needed))))
        groups = []
        for run in runs:
            run_last = run.first_frame + run.frame_count - 1
            if run_last < first or run.first_frame > last:
                continue
            level = run.levels[min(level_index, len(run.levels) - 1)]
            factor = level.factor
            lo = max(0, int(math.floor((first - run.first_frame) / factor)) - 1)
            hi = min(
                len(level.mean),
                int(math.ceil((last - run.first_frame + 1) / factor)) + 1,
            )
            if hi <= lo:
                continue
            indices = np.arange(lo, hi, dtype=np.float32)
            frames = run.first_frame + (indices + 0.5) * factor - 0.5
            frames = np.minimum(frames, float(run_last))
            groups.append(
                (
                    factor,
                    frames,
                    level.mean[lo:hi],
                    None if level.minimum is None else level.minimum[lo:hi],
                    None if level.maximum is None else level.maximum[lo:hi],
                    (
                        None
                        if level.temporal_minimum is None
                        else level.temporal_minimum[lo:hi]
                    ),
                    (
                        None
                        if level.temporal_maximum is None
                        else level.temporal_maximum[lo:hi]
                    ),
                )
            )
        return groups

    def _paint_static(self, painter):
        super()._paint_static(painter)
        rect = self._metric_rect()
        painter.fillRect(rect, QColor("#1c2027"))
        painter.save()
        painter.setClipRect(rect)
        frame_span = max(1., self.view_end - self.view_start)

        def coordinates(frames, values, bounds):
            low, high = bounds
            values = np.asarray(values)
            if self.shared_y_range is None:
                values = np.clip(values, low, high)
            return (rect.left() + (frames - self.view_start) * rect.width() / frame_span,
                    rect.bottom() - 4 - (values - low) * (rect.height() - 10) / max(1e-12, high - low))

        def path_for(x, y):
            path = QPainterPath()
            if len(x):
                path.moveTo(float(x[0]), float(y[0]))
                for px, py in zip(x[1:], y[1:]):
                    path.lineTo(float(px), float(py))
            return path

        ratio = max(1.0, painter.device().devicePixelRatioF())
        raster_width = max(1, round(rect.width() * ratio))
        raster_height = max(1, round(rect.height() * ratio))
        extrema_raster = np.zeros(
            (raster_height, raster_width, 4), dtype=np.uint8
        )

        def raster_path(x, y, color, *, smooth=False):
            """Draw one extrema curve directly at physical display resolution."""
            finite = np.isfinite(x) & np.isfinite(y)
            if not np.any(finite):
                return
            points = np.column_stack((
                np.rint((np.asarray(x)[finite] - rect.left()) * ratio),
                np.rint((np.asarray(y)[finite] - rect.top()) * ratio),
            )).astype(np.int32)
            points[:, 0] = np.clip(points[:, 0], 0, raster_width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, raster_height - 1)
            rgba = (
                color.red(), color.green(), color.blue(),
                255 if smooth else 180,
            )
            if len(points) == 1:
                cv2.circle(extrema_raster, tuple(points[0]), max(1, round(ratio)), rgba, -1)
            else:
                cv2.polylines(
                    extrema_raster,
                    [points],
                    False,
                    rgba,
                    max(1, round(ratio)) if smooth else 1,
                    cv2.LINE_AA if smooth else cv2.LINE_8,
                )

        if self.shared_y_range is not None and self.shared_y_range[0] <= 0 <= self.shared_y_range[1]:
            _, y = coordinates(np.array([0]), np.array([0]), self.shared_y_range)
            painter.setPen(QPen(QColor("#68717d"), 1.0))
            painter.drawLine(QLineF(rect.left(), float(y[0]), rect.right(), float(y[0])))

        for key, spec in self.series.items():
            if key not in self.visible_series:
                continue
            bounds = self.metric_ranges[key]
            color = QColor(spec.color)
            style = Qt.PenStyle.DashLine if spec.dashed else Qt.PenStyle.SolidLine
            line_pen = QPen(color, spec.width, style)
            if spec.temporal_extrema:
                line_pen.setCosmetic(True)
            overview_envelope = (
                spec.temporal_extrema
                and frame_span / max(1., rect.width()) > .5
            )
            # The pyramid has power-of-two levels. Request the screen width so
            # connected extrema samples cover the visible horizontal pixels
            # without calculating during painting.
            target = max(
                2,
                round(rect.width() * ratio)
                if overview_envelope
                else round(rect.width() / 2),
            )
            if spec.temporal_extrema and not overview_envelope:
                target = max(2, round(frame_span + 1))
            groups = self._visible_curve_groups(key, int(self.view_start), int(self.view_end), target)
            for factor, frames, mean, minimum, maximum, temporal_min, temporal_max in groups:
                x, y = coordinates(frames, mean, bounds)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing, len(frames) <= 300)
                if overview_envelope and temporal_min is not None:
                    _, y0 = coordinates(frames, temporal_min, bounds)
                    _, y1 = coordinates(frames, temporal_max, bounds)
                    if factor == 1:
                        raster_path(x, y0, color, smooth=True)
                    else:
                        raster_path(x, y0, color)
                        raster_path(x, y1, color)
                    continue
                if self.envelope_visible and spec.show_envelope and minimum is not None and len(frames) >= 2:
                    _, y0 = coordinates(frames, minimum, bounds)
                    _, y1 = coordinates(frames, maximum, bounds)
                    fill = QColor(color)
                    fill.setAlpha(30 if spec.dashed else 42)
                    if len(frames) <= 300:
                        band = path_for(np.concatenate((x, x[::-1])), np.concatenate((y0, y1[::-1])))
                        band.closeSubpath()
                        painter.fillPath(band, fill)
                    else:
                        painter.setPen(QPen(fill, 1.2))
                        painter.drawLines([QLineF(float(px), float(a), float(px), float(b)) for px, a, b in zip(x, y0, y1)])
                    painter.setPen(QPen(color, 1.2, style))
                    painter.drawPath(path_for(x, y0))
                    painter.drawPath(path_for(x, y1))
                else:
                    if spec.temporal_extrema:
                        raster_path(x, y, color, smooth=True)
                        continue
                    painter.setPen(line_pen)
                    if len(x) == 1:
                        painter.drawEllipse(QRectF(float(x[0]) - 1.5, float(y[0]) - 1.5, 3, 3))
                    else:
                        painter.drawPath(path_for(x, y))

        if np.any(extrema_raster[:, :, 3]):
            raster_image = QImage(
                extrema_raster.data,
                raster_width,
                raster_height,
                extrema_raster.strides[0],
                QImage.Format.Format_RGBA8888,
            ).copy()
            raster_image.setDevicePixelRatio(ratio)
            painter.drawImage(QPointF(rect.left(), rect.top()), raster_image)

        for key in self.visible_series:
            frames = self._infinity_frames[key]
            first = np.searchsorted(frames, self.view_start, side='left')
            last = np.searchsorted(frames, self.view_end, side='right')
            if last <= first:
                continue
            pixels = np.unique(np.rint(
                rect.left() + (frames[first:last] - self.view_start) * rect.width() / frame_span))
            painter.setPen(QPen(QColor(self.series[key].color), 2))
            painter.drawLines([QLineF(float(x), rect.top() + 2, float(x), rect.top() + 7)
                               for x in pixels])
            painter.drawText(QRectF(rect.left() + 3, rect.top() + 9, 24, 20),
                             Qt.AlignmentFlag.AlignLeft, '∞')

        for line in self.reference_lines:
            if not line.visible or not np.isfinite(line.value):
                continue
            if line.series_key is not None:
                if line.series_key not in self.visible_series:
                    continue
                bounds = self.metric_ranges[line.series_key]
            else:
                bounds = self.shared_y_range
            if bounds is None:
                continue
            _, y = coordinates(np.array([0]), np.array([line.value]), bounds)
            y = float(y[0])
            if not bounds[0] <= line.value <= bounds[1] or not rect.top() <= y <= rect.bottom():
                continue
            painter.setPen(QPen(QColor(line.color), 1.2, Qt.PenStyle.DashLine if line.dashed else Qt.PenStyle.SolidLine))
            painter.drawLine(QLineF(rect.left(), y, rect.right(), y))
            if self.reference_labels_visible:
                width = self.fontMetrics().horizontalAdvance(line.label) + 8
                height = max(16, self.fontMetrics().height() + 2)
                left = rect.left() + 3 if line.label_side == "left" else rect.right() - width - 3
                label_rect = QRectF(
                    left,
                    max(rect.top(), min(y - height, rect.bottom() - height)),
                    width,
                    height,
                )
                painter.fillRect(label_rect, QColor(28, 32, 39, 215))
                painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, line.label)
        painter.restore()
        if self.shared_y_range is not None:
            painter.setPen(self.palette().text().color())
            label_height = max(16, self.fontMetrics().height() + 2)
            for value in self._quality_axis_tick_values():
                _, y = coordinates(np.array([0]), np.array([value]), self.shared_y_range)
                y = float(y[0])
                painter.drawLine(QLineF(rect.left() - 4, y, rect.left(), y))
                painter.drawText(QRectF(1, y - label_height / 2,
                                        max(1, rect.left() - 9), label_height),
                                 Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                                 self._format_axis_value(value))

    def wheelEvent(self, event):
        if not event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            return super().wheelEvent(event)
        if self.shared_y_range is None or not self.interactions.y_zoom:
            return
        rect = self._metric_rect()
        if not rect.contains(event.position()):
            return
        delta = event.angleDelta().y() / 120. if event.angleDelta().y() else event.pixelDelta().y() / 40.
        if not delta:
            return
        low, high = self.shared_y_range
        fraction = float(np.clip((rect.bottom() - 4 - event.position().y()) / max(1., rect.height() - 10), 0., 1.))
        anchor = low + fraction * (high - low)
        span = float(np.clip((high - low) * .75 ** np.clip(delta, -20., 20.), 1e-6, 1e6))
        self.set_shared_y_range(anchor - fraction * span, anchor + (1 - fraction) * span)
        event.accept()

    def mousePressEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if self.shared_y_range is not None and self._metric_rect().contains(event.position()):
                if event.button() == Qt.MouseButton.RightButton and self.interactions.y_reset:
                    self._y_drag_start = None
                    self.reset_y_range()
                elif event.button() == Qt.MouseButton.LeftButton and self.interactions.y_pan:
                    self._drag_mode = None
                    self._y_drag_start = (event.position().y(), *self.shared_y_range)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._y_drag_start is not None and event.buttons() & Qt.MouseButton.LeftButton:
            y, low, high = self._y_drag_start
            shift = (event.position().y() - y) * (high - low) / max(1., self._metric_rect().height() - 10)
            self.set_shared_y_range(low + shift, high + shift)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._y_drag_start is not None and event.button() == Qt.MouseButton.LeftButton:
            self._y_drag_start = None
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if not event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            super().mouseDoubleClickEvent(event)


class GraphPanel(QWidget):
    """Graph with a generated or supplied legend and optional collapse control."""

    def __init__(self, graph, *, checkable=False, groups=None, title="", legend=None):
        super().__init__()
        self.graph = graph
        self.checkboxes = {}
        self.checkable = checkable
        self.groups = groups
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self.collapse_button = QToolButton()
        self.collapse_button.setText(title)
        self.collapse_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.collapse_button.setCheckable(True)
        self.collapse_button.setChecked(True)
        self.collapse_button.setArrowType(Qt.ArrowType.DownArrow)
        self.collapse_button.setVisible(bool(title))
        self.collapse_button.setToolTip("Collapse / expand plot")
        self.collapse_button.toggled.connect(self.set_expanded)
        header.addWidget(self.collapse_button, 0, Qt.AlignmentFlag.AlignTop)
        if legend is None:
            self.legend_layout = QHBoxLayout()
            self.legend = LegendPanel(self.legend_layout)
            self._rebuild_legend()
            graph.seriesVisibilityChanged.connect(self._sync_checkbox)
            graph.seriesChanged.connect(self._rebuild_legend)
        else:
            self.legend = legend
        header.addWidget(self.legend, 1)
        layout.addLayout(header)
        layout.addWidget(graph, 1)

    def set_expanded(self, expanded):
        self.collapse_button.blockSignals(True)
        self.collapse_button.setChecked(expanded)
        self.collapse_button.blockSignals(False)
        self.collapse_button.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.legend.setVisible(expanded)
        self.graph.setVisible(expanded)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Preferred if expanded else QSizePolicy.Policy.Fixed)
        self.setMaximumHeight(16777215 if expanded else self.layout().sizeHint().height())
        self.updateGeometry()

    def _rebuild_legend(self):
        previous_text = self.reference_label.text() if hasattr(self, "reference_label") else ""
        while self.legend_layout.count():
            item = self.legend_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.checkboxes.clear()
        self._legend_groups = tuple(self.groups) if self.groups is not None else tuple(
            SeriesGroup(key, spec.label, (key,)) for key, spec in self.graph.series.items())
        for group in self._legend_groups:
            spec = self.graph.series[group.series_keys[0]]
            if self.checkable:
                item = QCheckBox(group.label)
                item.setStyleSheet(f"QCheckBox {{ color: {spec.color}; }}")
                item.setChecked(set(group.series_keys) <= self.graph.visible_series)
                item.toggled.connect(lambda checked, g=group: self._toggle_group(g, checked))
                self.checkboxes[group.key] = item
            else:
                item = QLabel(f'<span style="color:{escape(spec.color)}">■</span> {escape(group.label)}')
            self.legend_layout.addWidget(item)
        self.reference_label = QLabel(previous_text)
        self.reference_label.setMinimumWidth(0)
        self.legend_layout.addWidget(self.reference_label, 1)

    def _toggle_group(self, group, checked):
        keys = set(group.series_keys)
        self.graph.set_visible_series(self.graph.visible_series | keys if checked else self.graph.visible_series - keys)

    def _sync_checkbox(self, key, visible):
        for group in self._legend_groups:
            checkbox = self.checkboxes.get(group.key)
            if checkbox is not None and key in group.series_keys:
                checkbox.blockSignals(True)
                checkbox.setChecked(set(group.series_keys) <= self.graph.visible_series)
                checkbox.blockSignals(False)

    def set_reference_lines(self, lines, *, suffix=""):
        text = " &nbsp; ".join(f'<span style="color:{escape(color)}">━━</span> {escape(label)}' for label, color in lines)
        self.reference_label.setText(text + escape(suffix))
