"""Application-wide keyboard controls for BiVOSIA interface text.

Examples
--------
>>> from PySide6.QtWidgets import QApplication
>>> from bivosia.ui.font_zoom import install_font_zoom
>>> app = QApplication.instance() or QApplication([])  # doctest: +SKIP
>>> controller = install_font_zoom(app)  # doctest: +SKIP
>>> controller.reset()  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QAction, QFont, QFontInfo
from PySide6.QtWidgets import QApplication


FONT_ZOOM_PROPERTY = "bivosia_font_zoom_controller"


class FontZoomController(QObject):
    """Change the application font with Ctrl+plus, Ctrl+minus and Ctrl+zero."""

    def __init__(self, application: QApplication):
        super().__init__(application)
        self.application = application
        self.default_font = QFont(application.font())
        point_size = self.default_font.pointSizeF()
        self.default_point_size = (
            point_size if point_size > 0
            else QFontInfo(self.default_font).pointSizeF()
        )
        self.point_size = self.default_point_size
        application.installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() != QEvent.Type.KeyPress:
            return super().eventFilter(watched, event)
        modifiers = event.modifiers()
        if not modifiers & Qt.KeyboardModifier.ControlModifier:
            return super().eventFilter(watched, event)
        if modifiers & (Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.MetaModifier):
            return super().eventFilter(watched, event)
        key = event.key()
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.increase()
        elif key == Qt.Key.Key_Minus:
            self.decrease()
        elif key == Qt.Key.Key_0:
            self.reset()
        else:
            return super().eventFilter(watched, event)
        event.accept()
        return True

    def increase(self):
        self.set_point_size(self.point_size + 1.0)

    def decrease(self):
        self.set_point_size(self.point_size - 1.0)

    def reset(self):
        self.point_size = self.default_point_size
        self._apply_font(QFont(self.default_font))

    def set_point_size(self, point_size):
        point_size = max(7.0, min(24.0, float(point_size)))
        if point_size == self.point_size:
            return
        font = QFont(self.default_font)
        font.setPointSizeF(point_size)
        self.point_size = point_size
        self._apply_font(font)

    def _apply_font(self, font):
        """Apply one font to existing and future interface text."""
        self.application.setFont(font)
        for widget in self.application.allWidgets():
            widget.setFont(font)
            widget.updateGeometry()
            widget.update()
        for window in self.application.topLevelWidgets():
            for action in window.findChildren(QAction):
                action.setFont(font)
            layout = window.layout()
            if layout is not None:
                layout.invalidate()
                layout.activate()
        QTimer.singleShot(0, self._refresh_windows)

    def _refresh_windows(self):
        for window in self.application.topLevelWidgets():
            refresh = getattr(window, "refresh_font_geometry", None)
            if callable(refresh):
                refresh()


def install_font_zoom(application=None):
    """Install one font zoom controller on the current Qt application."""
    application = application or QApplication.instance()
    if application is None:
        raise RuntimeError("A QApplication is required before installing font zoom")
    controller = application.property(FONT_ZOOM_PROPERTY)
    if controller is None:
        controller = FontZoomController(application)
        application.setProperty(FONT_ZOOM_PROPERTY, controller)
    return controller
