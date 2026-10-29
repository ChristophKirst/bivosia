"""BiVOSIA information window with the application graphic and a masked video preview.

Examples
--------
>>> from PySide6.QtWidgets import QApplication, QWidget
>>> from bivosia.ui.about_dialog import show_about
>>> app = QApplication.instance() or QApplication([])  # doctest: +SKIP
>>> viewer = QWidget()  # doctest: +SKIP
>>> show_about(viewer)  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QMovie, QPixmap
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFrame, QLabel,
                               QScrollArea, QVBoxLayout, QWidget)

from bivosia import __version__


def show_about(parent):
    """Show project information and play the bundled preview while open."""
    dialog = QDialog(parent)
    dialog.setWindowTitle('About BiVOSIA')
    layout = QVBoxLayout(dialog)
    content = QWidget()
    details = QVBoxLayout(content)
    details.addWidget(QLabel(f'<b>BiVOSIA</b> · Version {__version__}'))
    description = QLabel(
        'Bidirectional video-object segmentation, zone analysis, '
        'and manual revision and identity linking for animal recordings.')
    description.setWordWrap(True)
    details.addWidget(description)
    assets = Path(__file__).with_name('assets')
    graphic = QLabel()
    graphic.setAlignment(Qt.AlignmentFlag.AlignCenter)
    graphic.setPixmap(QPixmap(str(assets / 'bivosia-graphic.png')).scaledToWidth(
        480, Qt.TransformationMode.SmoothTransformation))
    details.addWidget(graphic)
    preview = QLabel()
    preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
    movie = QMovie(str(assets / 'about.gif'), parent=dialog)
    preview.setMovie(movie)
    details.addWidget(preview)
    details.addWidget(QLabel(f'{__author__}\n{__contact__}'))
    scroll = QScrollArea()
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setWidgetResizable(True)
    scroll.setWidget(content)
    layout.addWidget(scroll)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    movie.start()
    available = dialog.screen().availableGeometry()
    dialog.resize(content.sizeHint().width() + 40,
                  min(content.sizeHint().height() + 60, available.height() - 80))
    try:
        dialog.exec()
    finally:
        movie.stop()
        dialog.deleteLater()
