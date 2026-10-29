"""Shared review viewer and independent frame-player controls."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from PySide6.QtCore import QEvent, Signal
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from bivosia.ui.link_editor import MaskVideoGrid
from bivosia.ui.colors import identity_colors
from bivosia.ui.player import FramePlayer
from bivosia.ui.video_panels import MaskDisplayControl
from bivosia.ui.link_colors import current_link_labels, aligned_labels, global_label_colors


class ReviewVideoViewer(QWidget):
    activated = Signal(str)
    displayChanged = Signal()

    def __init__(self, columns):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 6, 6, 6)
        self.grid = MaskVideoGrid(tuple(name for name, _ in columns))
        self.players = {}
        self.active_side = columns[0][0]
        for name, color in columns:
            player = FramePlayer(name, color, directions=len(columns) == 2)
            self.players[name] = player
            self.grid.headers[name].addWidget(player)
            player.selected.connect(lambda key=name: self.activated.emit(key))
        controls = QHBoxLayout()
        controls.addStretch()
        self.mask_display = MaskDisplayControl()
        self.mask_display.valueChanged.connect(self.displayChanged)
        controls.addWidget(self.mask_display)
        controls.addWidget(QLabel("Colors"))
        self.color_mode = QComboBox()
        self.color_mode.addItem("Labels", "labels")
        if len(columns) == 2:
            self.color_mode.addItem("L/R", "sides")
            self.color_mode.addItem("Current link", "link")
        self.color_mode.addItem("Global", "global")
        if len(columns) == 2:
            self.color_mode.setCurrentIndex(self.color_mode.findData("link"))
        self.color_mode.currentIndexChanged.connect(self.displayChanged)
        controls.addWidget(self.color_mode)
        reset = QPushButton("Reset zoom")
        reset.clicked.connect(self.grid.navigation.reset)
        controls.addWidget(reset)
        layout.addLayout(controls)
        layout.addWidget(self.grid, 1)
        for widget in self.findChildren(QWidget):
            widget.installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Type.FocusIn, QEvent.Type.MouseButtonPress):
            self.active_side = getattr(watched, "side", None) or next(
                (name for name, player in self.players.items() if player is watched or player.isAncestorOf(watched)),
                self.active_side)
            self.activated.emit(self.active_side)
        return super().eventFilter(watched, event)

    def stop(self):
        for player in self.players.values():
            player.stop()

    def render(self, data, sources, seed, preview=None, mapping=None, linking=False):
        self.grid.sources = sources
        self.grid.linking = linking
        mode = self.color_mode.currentData()
        local = current_link_labels(mapping, data.n) if mode == "link" and mapping is not None else {}
        for side, player in self.players.items():
            frame = player.frame
            player.source.blockSignals(True)
            player.source.setCurrentText(sources[side])
            player.source.blockSignals(False)
            player.source.setEnabled(linking)
            palette = identity_colors(data.n, .04 if side == "right" and mode == "sides" else .27)
            original = data.image(frame, side)
            for direction in ("forward", "reverse"):
                labels = tuple(range(data.n))
                colors = palette
                if local:
                    labels = aligned_labels(data.linking, direction, sources[side], frame, local[side], seed)
                elif mode == "global" and preview is not None:
                    colors = global_label_colors(data.linking, preview, direction, frame, seed)
                colors = [colors[i] for i in labels]
                self.grid.panes[side, direction].set_frame(
                    original, data.mask(direction, frame), data.centers(direction, frame), colors,
                    sources[side] == direction, self.grid.linking,
                    self.mask_display.opacity, self.mask_display.video_visible,
                    self.mask_display.outlines_visible,
                )
