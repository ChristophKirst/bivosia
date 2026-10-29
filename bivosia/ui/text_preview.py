"""Scrollable read-only text, highlighted JSON and rendered Markdown.

Examples
--------
>>> from PySide6.QtWidgets import QApplication
>>> from bivosia.ui.text_preview import TextPreview
>>> app = QApplication.instance() or QApplication([])  # doctest: +SKIP
>>> window = TextPreview('settings.json', '{"animals": 6}', 'json')  # doctest: +SKIP
>>> window.show()  # doctest: +SKIP
>>> app.exec()  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path

from PySide6.QtCore import QRegularExpression, Qt
from PySide6.QtGui import QColor, QFontDatabase, QPalette, QSyntaxHighlighter, QTextCharFormat
from PySide6.QtWidgets import QMainWindow, QPlainTextEdit

from bivosia.ui.markdown_view import MarkdownView


class JsonHighlighter(QSyntaxHighlighter):
    """Color JSON keys, strings, numbers and literals in the original text."""

    tokens = QRegularExpression(
        r'(?<key>"(?:\\.|[^"\\])*")(?=\s*:)|'
        r'(?<string>"(?:\\.|[^"\\])*")|'
        r'(?<number>-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)|'
        r'(?<literal>\b(?:true|false|null)\b)')

    def __init__(self, document, palette):
        super().__init__(document)
        dark = palette.color(QPalette.ColorRole.Base).lightness() < 128
        self.formats = {}
        for name, light, dim in (
                ('key', '#005cc5', '#79b8ff'), ('string', '#22863a', '#9ecbff'),
                ('number', '#b04b00', '#ffab70'), ('literal', '#6f42c1', '#b392f0')):
            style = QTextCharFormat()
            style.setForeground(QColor(dim if dark else light))
            self.formats[name] = style

    def highlightBlock(self, text):
        matches = self.tokens.globalMatch(text)
        while matches.hasNext():
            match = matches.next()
            for name, style in self.formats.items():
                start = match.capturedStart(name)
                if start >= 0:
                    self.setFormat(start, match.capturedLength(name), style)
                    break


def create_text_view(text, syntax='plain', *, images=None):
    """Create a read-only editor with JSON highlighting or rendered Markdown."""
    if syntax == 'markdown':
        view = MarkdownView(images)
        view.setMarkdown(text)
    else:
        view = QPlainTextEdit()
        view.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        view.highlighter = (JsonHighlighter(view.document(), view.palette())
                            if syntax == 'json' else None)
        view.setPlainText(text)
    view.setReadOnly(True)
    return view


class TextPreview(QMainWindow):
    """Display supplied text in a separate window with optional documentation images."""

    def __init__(self, filename, text, syntax='plain', parent=None, *, images=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle(Path(filename).name)
        self.resize(800, 600)
        self.text = create_text_view(text, syntax, images=images)
        self.setCentralWidget(self.text)
