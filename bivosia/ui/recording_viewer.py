"""Construct the standard viewer from a dataset.

Examples
--------
With an existing dataset containing mask analysis results:

>>> from PySide6.QtWidgets import QApplication
>>> from bivosia.dataset.layout import Dataset
>>> from bivosia.ui.recording_viewer import create_recording_viewer
>>> app = QApplication.instance() or QApplication([])  # doctest: +SKIP
>>> viewer = create_recording_viewer(Dataset("/path/to/dataset"))  # doctest: +SKIP
>>> viewer.showMaximized()  # doctest: +SKIP
>>> app.exec()  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QMenu, QMessageBox

from bivosia.dataset.layout import Dataset
from bivosia.ui.about_dialog import show_about
from bivosia.ui.zone_video_viewer import ZoneVideoViewer
from bivosia.ui.plot_styles import measure_labels


# Keep windows opened from the File menu alive until Qt deletes them.
_viewers = set()


def _open_recording(viewer):
    """Replace the viewer only after the selected recording loads successfully."""
    path = QFileDialog.getExistingDirectory(
        viewer, 'Open recording folder', str(viewer.dataset.directory.parent))
    if not path:
        return
    try:
        replacement = create_recording_viewer(Dataset(path))
    except Exception as error:
        QMessageBox.warning(viewer, 'Open recording folder', str(error))
        return
    replacement.restoreGeometry(viewer.saveGeometry())
    replacement.show()
    viewer.close()


def create_recording_viewer(dataset, *, zone_rows=()):
    """Open the selected annotation with optional additional timeline rows."""
    from bivosia.processing.revision import RevisionData
    from bivosia.dataset.session import RevisionSession
    from bivosia.ui.annotations import revision_zone
    from bivosia.ui.link_colors import LinkColors

    data = RevisionData(dataset)
    try:
        annotation = data.dataset.annotation
        session = (RevisionSession.load(annotation, data.missing_forward, data.missing_reverse, data.n)
                   if annotation else None)
        zones = [] if session is None else [revision_zone(zone) for zone in session.zones]
        seed = session.header['seed'] if session else .6
        global_links = LinkColors(data).preview(session, None, None, seed) if session else None
        viewer = ZoneVideoViewer(
            data.paths, zones, expected_masks=data.n,
            evidence=data.evidence, missing_signal=data.missing,
            forward_missing_signal=data.missing_forward, reverse_missing_signal=data.missing_reverse,
            missing_direction_known=data.missing_forward | data.missing_reverse | ~data.missing,
            quality_arrays={"zod_local_quality": data.quality, "mean_raw_assigned_iou": data.assigned_iou},
            feature_data=data.measures, feature_labels=measure_labels(data.measures),
            keypoints=data.keypoints, global_links=global_links, link_seed=seed,
            zone_rows=zone_rows)
        viewer.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        viewer.dataset = dataset
        viewer.file_menu = QMenu('File', viewer)
        viewer.menuBar().insertMenu(viewer.view_menu.menuAction(), viewer.file_menu)
        viewer.file_menu.addAction('Open...', lambda: _open_recording(viewer))
        viewer.help_menu = viewer.menuBar().addMenu('Help')
        viewer.help_menu.addAction('About BiVOSIA', lambda: show_about(viewer))
        _viewers.add(viewer)
        viewer.destroyed.connect(lambda: _viewers.discard(viewer))
        return viewer
    finally:
        data.close()
