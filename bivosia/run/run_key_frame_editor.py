"""Open the key-frame and mask-labeling editor.

Examples
--------
>>> from bivosia.run.run_key_frame_editor import main
>>> import sys
>>> sys.argv = ['run_key_frame_editor', '/path/to/dataset']  # doctest: +SKIP
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
os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtWidgets import QApplication

from bivosia.config import KEY_FRAME_BACKENDS
from bivosia.dataset.layout import Dataset
from bivosia.processing.log import configure_logging
from bivosia.ui.font_zoom import install_font_zoom
from bivosia.ui.icons import application_icon
from bivosia.ui.key_frame_editor import KeyFrameWindow


def main():
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", type=Path, help="Dataset folder")
    parser.add_argument("--storage-folder", type=Path, help="Additional storage root containing recording/, masks/ and keypoints/")
    parser.add_argument("--video", type=Path, help="Video within the dataset's recording folder")
    parser.add_argument("--animals", type=int, default=2)
    parser.add_argument("--model", choices=tuple(KEY_FRAME_BACKENDS), default=next(iter(KEY_FRAME_BACKENDS)))
    parser.add_argument("--checkpoint", type=Path)
    for name, backend in KEY_FRAME_BACKENDS.items():
        for parameter, option in backend.get('arguments', {}).items():
            parser.add_argument(option, dest=f'{name}_{parameter}')
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()
    parameters = {parameter: getattr(args, f'{args.model}_{parameter}')
                  for parameter in KEY_FRAME_BACKENDS[args.model].get('arguments', {})
                  if getattr(args, f'{args.model}_{parameter}') is not None}
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setWindowIcon(application_icon())
    install_font_zoom(app)
    dataset = Dataset(args.dataset, args.storage_folder, args.video)
    window = KeyFrameWindow(dataset, model=args.model, checkpoint=args.checkpoint,
                            device=args.device, parameters=parameters, animals=args.animals)
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
