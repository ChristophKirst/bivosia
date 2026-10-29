"""Exclusive mode buttons for mask selection, prompts and brush editing."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QToolBar


def mode_icon(mode, color):
    """Draw small vector symbols using the current palette's text color."""
    pixmap = QPixmap(20, 20)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(color, 1.5))
    if mode == "Prompt":
        painter.drawEllipse(QPointF(10, 10), 3, 3)
        for first, last in (((10, 1), (10, 5)), ((10, 15), (10, 19)),
                            ((1, 10), (5, 10)), ((15, 10), (19, 10))):
            painter.drawLine(QPointF(*first), QPointF(*last))
    elif mode == "Select":
        painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in
                                       ((3, 2), (16, 11), (10, 12), (7, 18))]))
    elif mode == "Paint":
        painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in
                                       ((3, 17), (5, 11), (14, 2), (18, 6), (9, 15))]))
        painter.drawLine(12, 4, 16, 8)
    elif mode == "Erase":
        painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in
                                       ((2, 12), (11, 3), (18, 10), (11, 17), (7, 17))]))
        painter.drawLine(6, 8, 14, 16)
    else:
        painter.drawLine(2, 10, 18, 10)
        painter.drawLine(10, 2, 10, 18)
        for x, y, dx, dy in ((2, 10, 3, 3), (18, 10, -3, 3), (10, 2, 3, 3), (10, 18, 3, -3)):
            if y == 10:
                painter.drawLine(x, y, x + dx, y - dy)
                painter.drawLine(x, y, x + dx, y + dy)
            else:
                painter.drawLine(x, y, x - dx, y + dy)
                painter.drawLine(x, y, x + dx, y + dy)
    painter.end()
    return QIcon(pixmap)


class MaskTools(QToolBar):
    changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.setMovable(False)
        self.setFloatable(False)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.group = QActionGroup(self)
        self.group.setExclusive(True)
        self.modes = {}
        for name, tip in (("Select", "Click a mask to select its animal"),
                          ("Prompt", "Click to add a point for the selected animal"),
                          ("Paint", "Left-drag to paint the selected animal"),
                          ("Erase", "Left-drag to erase mask pixels"),
                          ("Pan", "Left-drag to move the image")):
            action = QAction(mode_icon(name, self.palette().windowText().color()), name, self)
            action.setCheckable(True)
            action.setToolTip(tip)
            action.toggled.connect(lambda checked, name=name: self.changed.emit(name) if checked else None)
            self.group.addAction(action)
            self.addAction(action)
            self.modes[name] = action
        self.set_mode("Prompt")

    @property
    def mode(self):
        return self.group.checkedAction().text()

    def set_mode(self, mode):
        self.modes[mode].setChecked(True)
