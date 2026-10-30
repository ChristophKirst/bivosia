"""Shared panel dividers and detachable panels."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QDockWidget, QMainWindow, QProxyStyle, QSplitter, QStyle, QWidget


SPLITTER_HANDLE_WIDTH = 5
DOCK_SEPARATOR_WIDTH = 5


class _PanelStyle(QProxyStyle):
    """Small dot grips for native splitters and dock separators."""

    def pixelMetric(self, metric, option=None, widget=None):
        if metric == QStyle.PixelMetric.PM_SplitterWidth:
            return SPLITTER_HANDLE_WIDTH
        if metric == QStyle.PixelMetric.PM_DockWidgetSeparatorExtent:
            return DOCK_SEPARATOR_WIDTH
        return super().pixelMetric(metric, option, widget)

    @staticmethod
    def _grip(option, painter):
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        painter.setBrush(option.palette.highlight() if hovered else option.palette.text())
        painter.setOpacity(1. if hovered else .45)
        center = QPointF(option.rect.center())
        horizontal = option.rect.width() > option.rect.height()
        for offset in (-5, 0, 5):
            painter.drawEllipse(center + (QPointF(offset, 0) if horizontal else QPointF(0, offset)), 1.2, 1.2)
        painter.restore()

    def drawControl(self, element, option, painter, widget=None):
        if element == QStyle.ControlElement.CE_Splitter:
            self._grip(option, painter)
        elif element == QStyle.ControlElement.CE_DockWidgetTitle:
            painter.save()
            font = painter.font()
            font.setBold(True)
            painter.setFont(font)
            super().drawControl(element, option, painter, widget)
            painter.restore()
        else:
            super().drawControl(element, option, painter, widget)

    def drawPrimitive(self, element, option, painter, widget=None):
        if element == QStyle.PrimitiveElement.PE_IndicatorDockWidgetResizeHandle:
            self._grip(option, painter)
        else:
            super().drawPrimitive(element, option, painter, widget)


def _style_panels(widget):
    style = _PanelStyle()
    style.setParent(widget)
    widget.setStyle(style)


class PanelSplitter(QSplitter):
    """Project-wide resizable panel divider with a visible drag grip."""

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        _style_panels(self)

    def createHandle(self):
        handle = super().createHandle()
        handle.setStyle(self.style())
        return handle


class PanelHost(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        _style_panels(self)
        self.setDockNestingEnabled(True)


class DockPanel(QDockWidget):
    """A detachable panel, or a fixed untitled area beside detachable panels."""

    def __init__(self, title, widget, parent, *, fixed=False, closable=False):
        super().__init__(title, parent)
        self.setStyle(parent.style())
        if fixed:
            self.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
            self.setTitleBarWidget(QWidget(self))
            self.toggleViewAction().setVisible(False)
        else:
            self.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable |
                             QDockWidget.DockWidgetFeature.DockWidgetFloatable |
                             (QDockWidget.DockWidgetFeature.DockWidgetClosable if closable
                              else QDockWidget.DockWidgetFeature.NoDockWidgetFeatures))
        self.setWidget(widget)

    def event(self, event):
        title_double_click = event.type() == QEvent.Type.NonClientAreaMouseButtonDblClick or (
            event.type() == QEvent.Type.MouseButtonDblClick and self.widget() is not None
            and event.position().y() < self.widget().geometry().top())
        if self.isFloating() and title_double_click and event.button() == Qt.MouseButton.LeftButton:
            self.showNormal() if self.isMaximized() else self.showMaximized()
            event.accept()
            return True
        return super().event(event)


def add_view_menu(window, panels):
    """Add show/hide actions for the supplied detachable panels."""
    menu = window.menuBar().addMenu('View')
    for title, panel in panels.items():
        action = panel.toggleViewAction()
        action.setText(panel.windowTitle() or title)
        menu.addAction(action)
    return menu
