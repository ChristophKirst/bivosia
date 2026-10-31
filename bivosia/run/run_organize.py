"""Organize a folder of videos or recording subfolders into datasets.

Examples
--------
>>> from bivosia.run.run_organize import main
>>> main(["incoming"])  # doctest: +SKIP
>>> main(["incoming", "--apply"])  # doctest: +SKIP
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from collections import Counter
from pathlib import Path
import argparse
import logging
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bivosia.dataset.paths import file_path
from bivosia.dataset.storage.organize import apply_plan, plan_collection
from bivosia.processing.log import configure_logging


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--apply", action="store_true", help="Apply the plan; default is inspection only")
    parser.add_argument("--save-reorganization", action="store_true",
                        help="Save a file reorganization record when applying the plan")
    parser.add_argument("--data-folder", action="append", default=[], metavar="NAME",
                        help="Preserve this subfolder as experiment data, not a dataset (repeatable)")
    args = parser.parse_args(argv)
    configure_logging()
    logger = logging.getLogger(__name__)
    try:
        root = args.folder.expanduser().resolve(strict=True)
        plan = plan_collection(root, data_folders=args.data_folder)
        logger.info("%s datasets; %s entries to move", len(plan.datasets), len(plan.moves))
        counts = Counter(path.name for path in plan.existing_analysis)
        logger.info("Existing analysis files (presence only): %s", dict(sorted(counts.items())))
        if plan.moves:
            first = plan.datasets[0]
            for move in plan.moves:
                if move.source.parent in (root, first):
                    logger.info("%s -> %s", move.source.relative_to(root),
                                move.destination.relative_to(root) if move.destination.is_relative_to(root)
                                else move.destination)
        if args.apply:
            record = root / file_path("reorganization") if args.save_reorganization else None
            apply_plan(plan, reorganization_file=record)
        else:
            logger.info("Inspection only; use --apply to move files")
    except (KeyboardInterrupt, InterruptedError):
        logger.warning("Organizing interrupted; inspect source and destination folders before continuing")
        return 1
    except Exception:
        logger.exception("Organizing failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
