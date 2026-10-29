"""Open BiVOSIA's dataset and window manager.

Examples
--------
>>> from bivosia.run.run_app import main
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

from bivosia.processing.log import configure_logging
from bivosia.ui.font_zoom import install_font_zoom
from bivosia.ui.icons import application_icon
from bivosia.ui.main_window import MainWindow


def main(argv=None):
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('datasets', nargs='*', type=Path, help='Dataset folders to open')
    args = parser.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setWindowIcon(application_icon())
    install_font_zoom(app)
    window = MainWindow()
    for path in args.datasets:
        window.open_dataset(path)
    window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
