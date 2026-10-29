"""Open the video-object segmentation interface.

Examples
--------
>>> from bivosia.run.run_vos import main
>>> import sys
>>> sys.argv = ['run_vos', '/path/to/dataset']  # doctest: +SKIP
>>> main()  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path
import argparse
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault('QT_API', 'pyside6')

from PySide6.QtWidgets import QApplication

from bivosia.dataset.layout import Dataset
from bivosia.processing.log import configure_logging
from bivosia.ui.font_zoom import install_font_zoom
from bivosia.ui.icons import application_icon
from bivosia.ui.vos_window import VosWindow


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('dataset', nargs='?')
    args = parser.parse_args()
    configure_logging()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setWindowIcon(application_icon())
    install_font_zoom(app)
    window = VosWindow(Dataset(args.dataset) if args.dataset else None)
    window.showMaximized()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
