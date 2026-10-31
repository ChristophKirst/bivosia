"""Run mask quantification for recordings or feature analysis for one dataset.

Examples
--------
>>> from bivosia.run.run_analysis import main
>>> import sys
>>> sys.argv = ['run_analysis', 'quantify', '/path/to/dataset', '--n-animals', '2']  # doctest: +SKIP
>>> main()  # doctest: +SKIP
"""

from __future__ import annotations

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path
import argparse
import logging
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bivosia.dataset.layout import Dataset, datasets_in
from bivosia.processing.log import configure_logging
from bivosia.processing.analysis.mask_quality import quantify_many


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("quantify", "features", "align"))
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--collection", action="store_true",
                        help="Quantify the dataset folders immediately inside the given folder")
    parser.add_argument("--n-animals", type=int, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--temporal-intersections", action="store_true",
                        help="Also count consecutive-frame intersections in both directions (quantify only)")
    parser.add_argument("--annotations", type=Path, help="Restrict feature extraction to these annotated zones")
    parser.add_argument("--no-convex-hull", action="store_true",
                        help="Skip convex-hull perimeter excess (features only)")
    parser.add_argument("--fft-features", action="store_true",
                        help="Include FFT energy, multiscale perimeter and bending roughness (features only)")
    parser.add_argument("--hu-features", action="store_true",
                        help="Calculate Hu moments and Hu-based shape change (features only)")
    parser.add_argument("--fallback-previous", action="store_true",
                        help="Reuse previous keypoint assignments on tied frames (align only)")
    parser.add_argument("--exclude-keypoints", nargs='+', default=[], metavar="NAME",
                        help="Keypoint names omitted from alignment scoring (align only)")
    args = parser.parse_args()
    worker_options = {} if args.workers is None else {'workers': args.workers}
    configure_logging()
    logger = logging.getLogger(__name__)
    try:
        if args.hu_features and args.stage != 'features':
            parser.error('Hu features apply only to features')
        if args.fft_features and args.stage != 'features':
            parser.error('FFT features apply only to features')
        if args.fallback_previous and args.stage != 'align':
            parser.error('Previous-assignment fallback applies only to align')
        if args.exclude_keypoints and args.stage != 'align':
            parser.error('Keypoint exclusions apply only to align')
        if args.stage == 'align':
            if args.collection or args.temporal_intersections or args.annotations or args.no_convex_hull:
                parser.error('Alignment uses one dataset and its selected mask annotation')
            from bivosia.processing.analysis.alignment import calculate_keypoint_alignment

            calculate_keypoint_alignment(Dataset(args.dataset), args.n_animals,
                exclude_keypoints=args.exclude_keypoints, **worker_options,
                **({'fallback_previous': True} if args.fallback_previous else {}))
        elif args.stage == "quantify":
            if args.annotations or args.no_convex_hull:
                parser.error("Annotation selection and convex-hull features apply only to features")
            datasets = datasets_in(args.dataset) if args.collection else [Dataset(args.dataset)]
            results = quantify_many(datasets, args.n_animals, **worker_options,
                                    **({'temporal_intersections': True} if args.temporal_intersections else {}))
            for dataset, paths in zip(datasets, results):
                result = dataset.relative(paths["metadata"])
                logger.info("Saved %s: %s", dataset.directory.name, result,
                            extra={"result_path": result})
        else:
            if args.temporal_intersections or args.collection:
                parser.error("Temporal intersections and collection processing apply only to quantify")
            from bivosia.processing.analysis.mask_features import calculate_dataset_features

            dataset = Dataset(args.dataset)
            outputs = calculate_dataset_features(
                dataset, args.n_animals, **worker_options, annotations=args.annotations,
                **({'include_convex_hull': False} if args.no_convex_hull else {}),
                **({'include_fft_features': True} if args.fft_features else {}),
                **({'include_hu_features': True} if args.hu_features else {}))
            result = outputs['mask_measures']
            logger.info("Saved %s", dataset.relative(result), extra={"result_path": dataset.relative(result)})
    except (InterruptedError, KeyboardInterrupt):
        logger.warning("Analysis cancelled; any unfinished outputs remain as partial files")
        return 1
    except Exception:
        logger.exception("Analysis failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
