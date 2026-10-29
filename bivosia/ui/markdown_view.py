"""Markdown documentation with section links, styled code and bundled media.

Examples
--------
>>> from PySide6.QtWidgets import QApplication
>>> from bivosia.ui.markdown_view import MarkdownView
>>> app = QApplication.instance() or QApplication([])  # doctest: +SKIP
>>> view = MarkdownView()  # doctest: +SKIP
>>> view.setMarkdown('# Guide\\n\\nUse `python -m bivosia.run.run_app`.')  # doctest: +SKIP
>>> view.show()  # doctest: +SKIP
>>> app.exec()  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import re

from PySide6.QtCore import QByteArray, QEvent, Qt, QUrl
from PySide6.QtGui import (QColor, QDesktopServices, QImage, QMovie, QPalette,
                          QTextBlockFormat, QTextCharFormat, QTextCursor,
                          QTextDocument, QTextFormat, QTextFrameFormat, QTextLength,
                          QTextTable)
from PySide6.QtWidgets import QTextBrowser


class MarkdownView(QTextBrowser):
    """Display supplied Markdown and explicitly supplied local image paths."""

    def __init__(self, images=None):
        super().__init__()
        self._image_paths = dict(images or {})
        self._images = {}
        self._movies = {}
        self._image_cursors = []
        self._code_frames = []
        self._ready = False
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)
        self.anchorClicked.connect(self._open_link)

    def setMarkdown(self, text):
        self._ready = False
        for movie in self._movies.values():
            movie.stop()
            movie.deleteLater()
        self._movies.clear()
        self._images.clear()
        self._image_cursors.clear()
        self._code_frames.clear()
        super().setMarkdown(text)
        self._prepare_document()
        self._ready = True
        self._apply_style()
        self._resize_images()
        if self.isVisible():
            for movie in self._movies.values():
                movie.start()

    def _prepare_document(self):
        document = self.document()
        code_formats, block_formats, headings, code_groups = [], [], [], []
        anchors = set()
        block = document.begin()
        group = None
        while block.isValid():
            code = block.blockFormat().hasProperty(QTextFormat.Property.BlockCodeLanguage)
            if code:
                if group is None:
                    group = [block.position(), 0, block.blockFormat()]
                    code_groups.append(group)
                group[1] = block.position() + block.length() - 1
            else:
                group = None
            style = block.charFormat()
            if code and style.hasProperty(QTextFormat.Property.FontPointSize):
                style.clearProperty(QTextFormat.Property.FontPointSize)
                block_formats.append((block.position(), style))
            fragments = block.begin()
            while not fragments.atEnd():
                fragment = fragments.fragment()
                style = fragment.charFormat()
                if ((code or style.fontFixedPitch())
                        and style.hasProperty(QTextFormat.Property.FontPointSize)):
                    style.clearProperty(QTextFormat.Property.FontPointSize)
                    code_formats.append((fragment.position(), fragment.length(), style))
                fragments += 1
            if block.blockFormat().headingLevel():
                name = re.sub(r'[^\w\- ]', '', block.text().lower()).replace(' ', '-')
                anchor, suffix = name, 0
                while anchor in anchors:
                    suffix += 1
                    anchor = f'{name}-{suffix}'
                anchors.add(anchor)
                headings.append((block.position(), anchor))
            block = block.next()

        # Code keeps its monospace family and inherits the document's font size.
        cursor = QTextCursor(document)
        for position, style in block_formats:
            cursor.setPosition(position)
            cursor.setBlockCharFormat(style)
        for position, length, style in code_formats:
            cursor.setPosition(position)
            cursor.setPosition(position + length, QTextCursor.MoveMode.KeepAnchor)
            cursor.setCharFormat(style)
        for position, anchor in headings:
            cursor.setPosition(position)
            cursor.movePosition(QTextCursor.MoveOperation.NextCharacter,
                                QTextCursor.MoveMode.KeepAnchor)
            style = QTextCharFormat()
            style.setAnchor(True)
            style.setAnchorNames([anchor])
            cursor.mergeCharFormat(style)

        # Work backward so inserting a frame leaves earlier source positions intact.
        for start, end, block_format in reversed(code_groups):
            cursor.setPosition(start)
            cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
            frame = cursor.insertFrame(QTextFrameFormat())
            frame.firstCursorPosition().setBlockFormat(block_format)
            before = QTextCursor(frame.firstCursorPosition().block().previous())
            before.setBlockFormat(QTextBlockFormat())
            self._code_frames.append(frame)

        block = document.begin()
        while block.isValid():
            fragments = block.begin()
            while not fragments.atEnd():
                fragment = fragments.fragment()
                if fragment.charFormat().isImageFormat():
                    style = fragment.charFormat().toImageFormat()
                    resource = self.loadResource(QTextDocument.ResourceType.ImageResource,
                                                 QUrl(style.name()))
                    if isinstance(resource, QImage) and not resource.isNull():
                        cursor = QTextCursor(document)
                        cursor.setPosition(fragment.position())
                        cursor.setPosition(fragment.position() + fragment.length(),
                                           QTextCursor.MoveMode.KeepAnchor)
                        width = style.width() or min(640, resource.width())
                        self._image_cursors.append((cursor, width, resource.size()))
                fragments += 1
            block = block.next()

    def _apply_style(self):
        document = self.document()
        em = self.fontMetrics().height()
        dark = self.palette().color(QPalette.ColorRole.Base).lightness() < 128
        code_background = QColor('#21262d' if dark else '#f6f8fa')
        document.setDocumentMargin(em * 1.5)
        block = document.begin()
        while block.isValid():
            style = block.blockFormat()
            code = style.hasProperty(QTextFormat.Property.BlockCodeLanguage)
            level = style.headingLevel()
            style.setTopMargin(em * (2 if level <= 2 else 1.5) if level else 0)
            style.setBottomMargin(em * (0.7 if level else 0.5))
            style.setLineHeight(120, QTextBlockFormat.LineHeightTypes.ProportionalHeight.value)
            if block == document.begin():
                style.setTopMargin(0)
            if block.textList():
                style.setBottomMargin(em * 0.25)
            if code:
                style.setTopMargin(0)
                style.setBottomMargin(0)
                style.setNonBreakableLines(False)
            if block.text().strip() == '\ufffc':
                style.setAlignment(Qt.AlignmentFlag.AlignHCenter)
                style.setLineHeight(100, QTextBlockFormat.LineHeightTypes.ProportionalHeight.value)
            QTextCursor(block).setBlockFormat(style)
            fragments = block.begin()
            inline = []
            while not fragments.atEnd():
                fragment = fragments.fragment()
                if not code and fragment.charFormat().fontFixedPitch():
                    inline.append((fragment.position(), fragment.length()))
                fragments += 1
            for position, length in inline:
                cursor = QTextCursor(document)
                cursor.setPosition(position)
                cursor.setPosition(position + length, QTextCursor.MoveMode.KeepAnchor)
                color = QTextCharFormat()
                color.setBackground(code_background)
                cursor.mergeCharFormat(color)
            block = block.next()
        for frame in self._code_frames:
            style = QTextFrameFormat()
            style.setBackground(code_background)
            style.setPadding(em)
            style.setBorder(0)
            style.setTopMargin(em * 0.3)
            style.setBottomMargin(em * 0.3)
            style.setWidth(QTextLength(QTextLength.Type.PercentageLength, 100))
            frame.setFrameFormat(style)
        border = QColor('#3d444d' if dark else '#d0d7de')
        for frame in document.rootFrame().childFrames():
            if isinstance(frame, QTextTable):
                style = frame.format()
                style.setBorderCollapse(True)
                style.setBorder(0.5)
                style.setBorderBrush(border)
                style.setCellPadding(em * 0.35)
                style.setCellSpacing(0)
                frame.setFormat(style)
                for row in range(frame.rows()):
                    for column in range(frame.columns()):
                        cell = frame.cellAt(row, column)
                        style = cell.format().toTableCellFormat()
                        style.setBorder(0.5)
                        style.setBorderBrush(border)
                        style.setBackground(code_background if row % 2 == 0
                                            else self.palette().color(QPalette.ColorRole.Base))
                        cell.setFormat(style)
    def _resize_images(self):
        width = max(1, self.viewport().width() - 2 * self.document().documentMargin())
        for cursor, requested_width, size in self._image_cursors:
            style = cursor.charFormat().toImageFormat()
            target = min(width, requested_width)
            style.setWidth(target)
            style.setHeight(target * size.height() / size.width())
            cursor.setCharFormat(style)
        # Reflow all blocks after resizing media and styling frames or tables.
        document = self.document()
        document.markContentsDirty(0, document.characterCount())

    def loadResource(self, kind, url):
        """Load only image paths explicitly supplied by the caller."""
        name = url.toString()
        if kind != QTextDocument.ResourceType.ImageResource or name not in self._image_paths:
            return QByteArray()
        if name not in self._images:
            path = self._image_paths[name]
            if str(path).lower().endswith('.gif'):
                movie = QMovie(str(path), parent=self)
                movie.jumpToFrame(0)
                self._images[name] = movie.currentImage()
                self._movies[name] = movie
                movie.frameChanged.connect(lambda frame, name=name: self._update_movie(name))
            else:
                self._images[name] = QImage(str(path))
        return self._images[name]

    def _update_movie(self, name):
        image = self._movies[name].currentImage()
        self._images[name] = image
        document = self.document()
        document.addResource(QTextDocument.ResourceType.ImageResource, QUrl(name), image)
        for cursor, _, _ in self._image_cursors:
            if cursor.charFormat().toImageFormat().name() == name:
                document.markContentsDirty(cursor.selectionStart(), 1)
        self.viewport().update()

    def _open_link(self, url):
        if url.isRelative() and not url.path():
            self.scrollToAnchor(url.fragment())
        elif url.scheme() in ('http', 'https', 'mailto'):
            QDesktopServices.openUrl(url)

    def changeEvent(self, event):
        super().changeEvent(event)
        if (getattr(self, '_ready', False)
                and event.type() in (QEvent.Type.FontChange, QEvent.Type.PaletteChange)):
            self._apply_style()
            self._resize_images()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._ready:
            self._resize_images()

    def showEvent(self, event):
        super().showEvent(event)
        for movie in self._movies.values():
            if movie.state() == QMovie.MovieState.NotRunning:
                movie.start()
            else:
                movie.setPaused(False)

    def hideEvent(self, event):
        for movie in self._movies.values():
            movie.setPaused(True)
        super().hideEvent(event)
