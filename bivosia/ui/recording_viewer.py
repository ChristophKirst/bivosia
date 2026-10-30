"""Construct the standard viewer from a dataset.

Examples
--------
With a dataset containing a video and any available masks or analysis results:

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

from bivosia.dataset.layout import Dataset
from bivosia.dataset.jsonio import read_json
from bivosia.ui.application_menus import add_dataset_window_menus
from bivosia.ui.window_manager import dataset_windows
from bivosia.ui.zone_video_viewer import ZoneVideoViewer
from bivosia.ui.plot_styles import measure_labels


def _open_recording(viewer, path):
    """Replace the viewer only after the selected recording loads successfully."""
    replacement = create_recording_viewer(Dataset(path))
    replacement.file_menu.remember(replacement.dataset)
    replacement.restoreGeometry(viewer.saveGeometry())
    replacement.show()
    viewer.close()


def create_recording_viewer(dataset, *, zone_rows=(), parent=None, open_dataset=None, controls=None):
    """Open the selected annotation as a window or an embedded viewer.

    An embedded viewer uses its editor's dataset-opening callback and is not
    registered as a separate application window. Its owner closes it to release
    the recording readers. Optional controls supply a (title, widget) pair for
    the viewer's left column.
    """
    from bivosia.processing.revision import RevisionData
    from bivosia.config import ANALYSIS_DEFAULTS
    from bivosia.dataset.session import RevisionSession, load_revision_zones
    from bivosia.ui.annotations import revision_zone
    from bivosia.analysis.linking import prepare_links

    from bivosia.ui.dataset_access import open_remote_reader
    data = RevisionData(dataset, remote_reader=open_remote_reader(dataset, optional=True), require_analysis=False)
    try:
        annotation = data.annotation
        session = (RevisionSession.load(annotation, data.missing_forward, data.missing_reverse, data.n,
                                        temporal_quality=data.temporal_quality)
                   if annotation and data.linking is not None and data.missing_forward is not None else None)
        saved_zones = session.zones if session else load_revision_zones(annotation, data.count) if annotation else ()
        zones = [revision_zone(zone) for zone in saved_zones]
        seed = session.header['seed'] if session else ANALYSIS_DEFAULTS['zones_of_disagreement']['theta_quality']
        global_links = prepare_links(session).resolve() if session else None
        annotation_mode = read_json(annotation)['header'].get('mode') if annotation else None
        viewer = ZoneVideoViewer(
            data.paths, zones, videos=data.videos, expected_masks=data.n,
            evidence=data.evidence, missing_signal=data.missing,
            forward_missing_signal=data.missing_forward, reverse_missing_signal=data.missing_reverse,
            quality_arrays={key: value for key, value in
                            (("zod_local_quality", data.quality), ("mean_raw_assigned_iou", data.assigned_iou))
                            if value is not None},
            feature_data=data.measures, feature_labels=measure_labels(data.measures),
            keypoints=data.keypoints, global_links=global_links, link_seed=seed,
            zone_rows=zone_rows, annotation_label='Annotations' if annotation_mode == 'automatic' else 'Hand',
            controls=controls)
        viewer.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        if parent is not None:
            viewer.setParent(parent)
        viewer.dataset = dataset
        viewer.recording = data
        windows = dataset_windows()
        viewer.file_menu, viewer.help_menu = add_dataset_window_menus(
            viewer, windows, lambda: viewer.dataset,
            open_dataset or (lambda path: _open_recording(viewer, path)))
        if parent is None:
            viewer.setProperty('dataset_window_kind', 'viewer')
            windows.add_dataset(dataset)
            windows.track(viewer)
        return viewer
    except BaseException:
        data.close()
        raise
