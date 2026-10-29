"""Shared legend container that inherits the active interface palette."""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PySide6.QtWidgets import QLayout

from html import escape

from PySide6.QtWidgets import QHBoxLayout, QWidget


def legend_groups_html(groups):
    """Format labeled color keys for shared legends."""
    topics = []
    for group in groups:
        if not group.items:
            continue
        keys = "&nbsp; ".join(
            f'<span style="color:{escape(color)}">■</span>' + (f'&nbsp;{escape(label)}' if label else '')
            for label, color in group.items)
        topics.append((f"{escape(group.label)}:&nbsp;" if group.label else "") + keys)
    return " &nbsp;|&nbsp; ".join(topics)


class LegendPanel(QWidget):
    """Compact legend panel shared by all timeline plot types."""

    def __init__(self, layout: QLayout, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(10)
        self.setLayout(layout)

    @classmethod
    def containing(cls, widget: QWidget) -> "LegendPanel":
        """Wrap one legend widget with the standard legend layout."""
        layout = QHBoxLayout()
        layout.addWidget(widget, 1)
        return cls(layout)
