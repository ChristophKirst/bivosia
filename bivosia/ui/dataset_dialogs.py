"""Choose a dataset folder and its experimental video.

Examples
--------
>>> from PySide6.QtWidgets import QApplication
>>> from bivosia.ui.dataset_dialogs import create_dataset_dialog
>>> app = QApplication.instance() or QApplication([])  # doctest: +SKIP
>>> dataset = create_dataset_dialog(None)  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from PySide6.QtWidgets import QFileDialog

from bivosia.dataset.formats import format_suffixes
from bivosia.dataset.setup import create_dataset, select_video


def create_dataset_dialog(parent):
    """Create a dataset in a folder selected or created in the file dialog."""
    path = QFileDialog.getExistingDirectory(parent, 'New dataset — choose an empty folder')
    return create_dataset(path) if path else None


def select_video_dialog(parent, dataset):
    """Choose a recording that remains in its current location."""
    suffixes = ' '.join('*' + suffix for suffix in format_suffixes('video'))
    path, _ = QFileDialog.getOpenFileName(parent, 'Select experimental video',
                                         str(dataset.directory), f'Video ({suffixes})')
    if not path:
        return False
    select_video(dataset, path)
    return True
