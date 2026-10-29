"""BiVOSIA application icon and shared system action icons with Qt fallbacks."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QStyle


_ACTIONS = {
    'new': ('document-new', QStyle.StandardPixmap.SP_FileIcon),
    'open': ('document-open', QStyle.StandardPixmap.SP_DialogOpenButton),
    'recent': ('document-open-recent', QStyle.StandardPixmap.SP_FileDialogBack),
    'workflow': ('view-list-tree', QStyle.StandardPixmap.SP_FileDialogDetailedView),
    'files': ('folder', QStyle.StandardPixmap.SP_DirIcon),
    'edit': ('document-edit', QStyle.StandardPixmap.SP_FileDialogContentsView),
    'run': ('media-playback-start', QStyle.StandardPixmap.SP_MediaPlay),
    'view': ('document-preview', QStyle.StandardPixmap.SP_FileDialogContentsView),
    'export': ('document-save-as', QStyle.StandardPixmap.SP_DialogSaveButton),
    'console': ('utilities-terminal', QStyle.StandardPixmap.SP_ComputerIcon),
    'refresh': ('view-refresh', QStyle.StandardPixmap.SP_BrowserReload),
    'cancel': ('process-stop', QStyle.StandardPixmap.SP_BrowserStop),
    'window': ('window-new', QStyle.StandardPixmap.SP_TitleBarMaxButton),
    'help': ('help-contents', QStyle.StandardPixmap.SP_DialogHelpButton),
    'about': ('help-about', QStyle.StandardPixmap.SP_MessageBoxInformation),
    'quit': ('application-exit', QStyle.StandardPixmap.SP_DialogCloseButton),
}


def application_icon():
    """Load the scalable sage-and-sand BiVOSIA application icon."""
    return QIcon(str(Path(__file__).with_name('assets') / 'bivosia-icon.svg'))


def action_icon(name):
    """Use the desktop theme where available and a standard Qt icon otherwise."""
    theme, fallback = _ACTIONS[name]
    return QIcon.fromTheme(theme, QApplication.style().standardIcon(fallback))
