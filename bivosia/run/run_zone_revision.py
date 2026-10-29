#!/usr/bin/env python3
"""Open the zone revision/linking editor; optionally specify any dataset folder.

Examples
--------
>>> from bivosia.run.run_zone_revision import main
>>> import sys
>>> sys.argv = ['run_zone_revision', '/path/to/dataset']  # doctest: +SKIP
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
from bivosia.ui.revision_linking_editor import RevisionLinkingWindow


def main():
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", default=str(DEFAULT_DATASET),
                        help="Dataset folder (default: six-vole data)")
    parser.add_argument("--storage-folder", type=Path, help="Storage root containing recording/, masks/ and keypoints/, relative to the dataset or absolute")
    parser.add_argument("--video", type=Path, help="Video filename within the recording folder")
    parser.add_argument("--annotations", type=Path, help="Resume this independent annotation file")
    args = parser.parse_args()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setWindowIcon(application_icon())
    install_font_zoom(app)
    dataset = Dataset(args.dataset, args.storage_folder, args.video)
    window = RevisionLinkingWindow(dataset, annotation=args.annotations)
    window.showMaximized()
    logging.getLogger(__name__).info("Opened revision/linking editor: %s", Path(args.dataset).name)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
