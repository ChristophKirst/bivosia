"""Shared frame selection, playback and loop controls."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from time import perf_counter
from weakref import WeakSet

from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton,
    QRadioButton, QSpinBox, QStyle, QWidget,
)

from bivosia.ui.plot_models import CursorSpec
from bivosia.ui.playback import next_frame


class FrameKeyNavigation(QObject):
    """Route arrow keys to zone or independent left/right frame players."""

    def __init__(self, owner, step, linking=lambda: False):
        super().__init__(owner)
        self.owner, self.step, self.linking = owner, step, linking
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() != QEvent.Type.KeyPress or event.modifiers() != Qt.KeyboardModifier.NoModifier:
            return False
        app = QApplication.instance()
        active = app.activeWindow()
        if app.activeModalWidget() is not None or active is None or not (
                active is self.owner.window() or self.owner.isAncestorOf(active)):
            return False
        key = event.key()
        if key not in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
            return False
        side = ('left' if key in (Qt.Key.Key_Left, Qt.Key.Key_Right) else 'right') if self.linking() else 'zone'
        offset = -1 if key in (Qt.Key.Key_Left, Qt.Key.Key_Down) else 1
        self.step(side, offset)
        return True

    def close(self):
        QApplication.instance().removeEventFilter(self)


class FramePlayer(QWidget):
    selected = Signal()
    frameRequested = Signal(int, bool)  # frame, playback (rather than a manual seek)
    frameChanged = Signal(int)
    playingChanged = Signal(bool)
    rangeChanged = Signal()
    sourceChanged = Signal(str)

    def __init__(self, name, color, directions=False, *, selectable=True):
        super().__init__()
        self.name, self.color = name, color
        self.frame, self.count = 0, 1
        self.loop_start = self.loop_end = 0
        self._loop_from_cursor = True
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.select = QRadioButton(name.title())
        self.select.setAutoExclusive(False)
        self.select.clicked.connect(self.selected)
        self.select.setVisible(selectable)
        self.spin = QSpinBox()
        self.spin.valueChanged.connect(self.seek)
        self.source = QComboBox()
        self.source.addItems(["forward", "reverse"])
        self.source.setVisible(directions)
        self.source.currentTextChanged.connect(self.sourceChanged)
        self.play = QPushButton()
        self.play.setCheckable(True)
        self._set_play_icon(False)
        self.play.toggled.connect(self._toggle)
        self.loop = QCheckBox("Loop")
        self.loop.setChecked(True)
        self.loop_bounds = QLabel("0–0")
        self.loop.toggled.connect(self.loop_bounds.setEnabled)
        self.loop.toggled.connect(self.rangeChanged)
        self.interval = QDoubleSpinBox()
        self.interval.setRange(1, 10000)
        self.interval.setValue(20)
        self.interval.setDecimals(1)
        for widget in (self.select, self.spin, self.source, self.play, self.loop, self.loop_bounds,
                       self.interval, QLabel("ms/frame")):
            layout.addWidget(widget)
        layout.addStretch()
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._step)
        self.setEnabled(False)

    def configure(self, count, fps):
        self.stop()
        self.count = count
        self.spin.setRange(0, count - 1)
        self.interval.setValue(1000 / (fps or 50))
        self.set_frame(0)
        self.set_loop(0, count - 1)
        self.setEnabled(True)

    def set_frame(self, frame):
        frame = max(0, min(self.count - 1, int(frame)))
        changed = frame != self.frame
        self.frame = frame
        if not self.loop_start <= frame <= self.loop_end:
            self._loop_from_cursor = False
        self.spin.blockSignals(True)
        self.spin.setValue(self.frame)
        self.spin.blockSignals(False)
        if changed:
            self.frameChanged.emit(self.frame)

    def seek(self, frame):
        self.stop()
        self.selected.emit()
        self.set_frame(frame)
        self.frameRequested.emit(self.frame, False)

    def set_loop(self, start, end):
        self.loop_start = max(0, min(self.count - 1, int(start)))
        self.loop_end = max(self.loop_start, min(self.count - 1, int(end)))
        self.loop_bounds.setText(f"{self.loop_start:,}–{self.loop_end:,}")
        self.rangeChanged.emit()

    @property
    def bounds(self):
        return (self.loop_start, self.loop_end) if self.looping else (0, self.count - 1)

    @property
    def looping(self):
        """Apply the selected loop only to playback started inside its bounds."""
        return self.loop.isChecked() and (not self.play.isChecked() or self._loop_from_cursor)

    def cursors(self):
        cursors = [CursorSpec(self.name, self.frame, self.color)]
        if self.loop.isChecked():
            for end, frame, text in (("start", self.loop_start, "["), ("end", self.loop_end, "]")):
                cursors.append(CursorSpec(f"{self.name}:{end}", frame, self.color,
                                          text, line=False, level=0, dot=False, owner=self.name))
        return cursors

    def move_cursor(self, key, frame):
        """Apply one of this player's playhead or loop-handle cursor moves."""
        owner, _, endpoint = key.partition(":")
        if owner != self.name:
            return False
        if endpoint == "start":
            self.set_loop(min(frame, self.loop_end), self.loop_end)
        elif endpoint == "end":
            self.set_loop(self.loop_start, max(frame, self.loop_start))
        elif endpoint:
            return False
        else:
            self.seek(frame)
        return True

    def stop(self):
        self.play.setChecked(False)

    def _set_play_icon(self, playing):
        label = "Pause" if playing else "Play"
        icon = (
            QStyle.StandardPixmap.SP_MediaPause
            if playing else QStyle.StandardPixmap.SP_MediaPlay
        )
        self.play.setIcon(self.style().standardIcon(icon))
        self.play.setToolTip(label)
        self.play.setAccessibleName(label)

    def _toggle(self, playing):
        self.timer.stop()
        self._set_play_icon(playing)
        if playing:
            self._loop_from_cursor = self.loop_start <= self.frame <= self.loop_end
        self.playingChanged.emit(playing)
        if playing:
            self.timer.start(max(1, round(self.interval.value())))

    def _step(self):
        started = perf_counter()
        frame = next_frame(self.frame, *self.bounds, self.looping)
        if frame is None:
            self.stop()
            return
        self.set_frame(frame)
        self.frameRequested.emit(frame, True)
        if self.play.isChecked():
            elapsed = (perf_counter() - started) * 1000
            self.timer.start(max(1, round(self.interval.value() - elapsed)))


class PlayerCursorController:
    """Synchronize shared player cursors with timeline plots."""

    def __init__(self, players, on_cursor_moved=None):
        self.players = {player.name: player for player in players}
        self.on_cursor_moved = on_cursor_moved
        self.line_plots = ()
        self.bracket_plots = ()
        self._connected_plots = WeakSet()
        for player in self.players.values():
            player.frameChanged.connect(self.refresh)
            player.rangeChanged.connect(self.refresh)

    def set_plots(self, *, line=(), brackets=()):
        """Set plots showing playheads only and plots also showing brackets."""
        self.line_plots = tuple(line)
        self.bracket_plots = tuple(brackets)
        for plot in (*self.line_plots, *self.bracket_plots):
            if plot not in self._connected_plots:
                plot.cursorMoved.connect(self._cursor_moved)
                self._connected_plots.add(plot)
        self.refresh()

    def refresh(self, *_args):
        cursors = tuple(
            cursor
            for player in self.players.values()
            for cursor in player.cursors()
        )
        lines = tuple(cursor for cursor in cursors if cursor.line)
        for plot in self.line_plots:
            plot.set_cursors(lines)
        for plot in self.bracket_plots:
            plot.set_cursors(cursors)

    def _cursor_moved(self, key, frame):
        if self.on_cursor_moved is not None:
            self.on_cursor_moved(key, frame)
            return
        owner = key.partition(":")[0]
        player = self.players.get(owner)
        if player is not None:
            player.move_cursor(key, frame)
