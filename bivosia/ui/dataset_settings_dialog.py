"""Configure a dataset's experimental video, keypoints and storage locations.

Examples
--------
>>> from PySide6.QtWidgets import QApplication
>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.ui.dataset_settings_dialog import DatasetSettingsDialog
>>> app = QApplication.instance() or QApplication([])  # doctest: +SKIP
>>> dialog = DatasetSettingsDialog(Dataset('/path/to/dataset'))  # doctest: +SKIP
>>> dialog.show()  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton, QVBoxLayout)

from bivosia.ui.dataset_dialogs import select_keypoints_dialog, select_video_dialog
from bivosia.ui.dataset_locations_dialog import DatasetLocationsDialog
from bivosia.ui.icons import action_icon


class DatasetSettingsDialog(QDialog):
    """Show the current dataset and save its input and location selections."""

    def __init__(self, dataset, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle(f'Dataset settings · {dataset.directory.name}')
        self.resize(700, 200)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.directory = QLineEdit(str(dataset.directory))
        self.directory.setReadOnly(True)
        form.addRow('Dataset folder', self.directory)
        self.video = QLineEdit()
        self.video.setReadOnly(True)
        self.video.textChanged.connect(self.video.setToolTip)
        row = QHBoxLayout()
        row.addWidget(self.video, 1)
        self.select_video_button = QPushButton('Select video…')
        self.select_video_button.setIcon(action_icon('open'))
        self.select_video_button.clicked.connect(self.select_video)
        row.addWidget(self.select_video_button)
        form.addRow('Video', row)
        self.keypoints = QLineEdit()
        self.keypoints.setReadOnly(True)
        self.keypoints.textChanged.connect(self.keypoints.setToolTip)
        row = QHBoxLayout()
        row.addWidget(self.keypoints, 1)
        self.select_keypoints_button = QPushButton('Select keypoints…')
        self.select_keypoints_button.setIcon(action_icon('open'))
        self.select_keypoints_button.clicked.connect(self.select_keypoints)
        row.addWidget(self.select_keypoints_button)
        form.addRow('Keypoints', row)
        self.locations_button = QPushButton('Locations…')
        self.locations_button.setIcon(action_icon('files'))
        self.locations_button.clicked.connect(self.edit_locations)
        form.addRow('', self.locations_button)
        layout.addLayout(form)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)
        self.refresh()

    def refresh(self):
        self.dataset.reload()
        try:
            self.video.setText(str(self.dataset.video))
            self.status.clear()
        except ValueError as error:
            self.video.clear()
            self.status.setText(str(error))
        try:
            path = self.dataset.keypoint_file()
            self.keypoints.setText(str(path) if path.is_file() else '')
        except ValueError as error:
            self.keypoints.clear()
            self.status.setText(str(error))

    def select_video(self):
        try:
            self.dataset.reload()
            if select_video_dialog(self, self.dataset):
                self.refresh()
        except Exception as error:
            self.status.setText(str(error))

    def edit_locations(self):
        DatasetLocationsDialog(self.dataset, self).exec()
        self.refresh()

    def select_keypoints(self):
        try:
            self.dataset.reload()
            if select_keypoints_dialog(self, self.dataset):
                self.refresh()
        except Exception as error:
            self.status.setText(str(error))
