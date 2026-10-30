"""Severity colors for processing logs and multiline tracebacks."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import re

from PySide6.QtCore import QEvent
from PySide6.QtGui import QColor, QPalette, QSyntaxHighlighter


class LogHighlighter(QSyntaxHighlighter):
    """Color warnings and errors while keeping progress text in the normal palette."""

    header = re.compile(r'^(?:\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:[,.]\d+)?\s+)?'
                        r'(DEBUG|INFO|WARNING|ERROR|CRITICAL)\b')

    def __init__(self, editor):
        self.editor = editor
        super().__init__(editor.document())
        editor.installEventFilter(self)

    def highlightBlock(self, text):
        match = self.header.match(text)
        if match:
            level = match.group(1)
            severity = 2 if level in ('ERROR', 'CRITICAL') else 1 if level == 'WARNING' else 0
        else:
            severity = max(0, self.previousBlockState())
        self.setCurrentBlockState(severity)
        if severity:
            dark = self.editor.palette().color(QPalette.ColorRole.Base).lightness() < 128
            colors = ('#e6b450', '#ff7979') if dark else ('#996300', '#b42332')
            self.setFormat(0, len(text), QColor(colors[severity - 1]))

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.PaletteChange:
            self.rehighlight()
        return False
