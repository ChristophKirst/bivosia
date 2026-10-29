"""Open a read-only dataset file window.

Examples
--------
>>> from bivosia.run.run_dataset import main
>>> main(['/path/to/dataset'])  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path
import argparse
import os
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault('QT_API', 'pyside6')

from PySide6.QtWidgets import QApplication

from bivosia.dataset.layout import Dataset
from bivosia.processing.log import configure_logging
from bivosia.ui.dataset_viewer import DatasetWidget
from bivosia.ui.font_zoom import install_font_zoom
from bivosia.ui.icons import application_icon


def main(argv=None):
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('dataset', type=Path, help='Dataset folder')
    parser.add_argument('--storage-folder', type=Path,
                        help='Additional storage root containing recording/, masks/ and keypoints/')
    args = parser.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setWindowIcon(application_icon())
    install_font_zoom(app)
    widget = DatasetWidget(Dataset(args.dataset, args.storage_folder))
    widget.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
