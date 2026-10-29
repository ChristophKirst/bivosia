#!/usr/bin/env python3
"""Open the standard video viewer for a dataset folder.

Examples
--------
>>> from bivosia.run.run_viewer import main
>>> import sys
>>> sys.argv = ['run_viewer', '/path/to/dataset']  # doctest: +SKIP
>>> main()  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path
import argparse
import logging
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtWidgets import QApplication

from bivosia.config import DEFAULT_DATASET
from bivosia.dataset.layout import Dataset
from bivosia.processing.log import configure_logging
from bivosia.ui.font_zoom import install_font_zoom
from bivosia.ui.icons import application_icon
from bivosia.ui.recording_viewer import create_recording_viewer


def main():
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", type=Path,
                        default=DEFAULT_DATASET)
    parser.add_argument("--storage-folder", type=Path)
    parser.add_argument("--video", type=Path)
    args = parser.parse_args()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setWindowIcon(application_icon())
    install_font_zoom(app)
    dataset = Dataset(args.dataset, args.storage_folder, args.video)
    viewer = create_recording_viewer(dataset)
    viewer.showMaximized()
    logging.getLogger(__name__).info("Opened standard viewer: %s", args.dataset.name)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
