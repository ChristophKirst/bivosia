"""Dataset hierarchy, file naming, storage preferences and application defaults."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from pathlib import Path
import sys

PACKAGE_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = PACKAGE_DIRECTORY.parent
SOURCE_CHECKOUT = (PROJECT_DIRECTORY / 'pyproject.toml').is_file()
RESOURCE_DIRECTORY = PROJECT_DIRECTORY if SOURCE_CHECKOUT else PACKAGE_DIRECTORY / '_resources'
EXTERNAL_DIRECTORY = (PROJECT_DIRECTORY.parent / 'external' if SOURCE_CHECKOUT
                      else Path(sys.prefix) / 'share/bivosia/external')
DEFAULT_DATASET = (PROJECT_DIRECTORY.parent if SOURCE_CHECKOUT else Path.cwd()) / 'data/posthoc_analysis'
APPLICATION_SETTINGS = ('BiVOSIA', 'Application')
REMOTE_SETTINGS = {'installations': 'remote_installations', 'datasets': 'remote_datasets'}
REMOTE_PROTOCOL_VERSION = 2
KEY_FRAME_BACKENDS = {
    'sam3': {
        'class': ('bivosia.segmentation.backends.sam', 'Sam3Segmenter'),
        'label': 'SAM3',
        'checkpoint': EXTERNAL_DIRECTORY / 'sam/checkpoints/sam3.pt',
        'checkpoint_suffixes': ('.pt', '.pth'),
        'defaults': {},
    },
    'sam2': {
        'class': ('bivosia.segmentation.backends.sam', 'Sam2Segmenter'),
        'label': 'SAM2',
        'checkpoint': EXTERNAL_DIRECTORY / 'sam/checkpoints/sam2.1_hiera_tiny.pt',
        'checkpoint_suffixes': ('.pt', '.pth'),
        'defaults': {'config': 'configs/sam2.1/sam2.1_hiera_t.yaml'},
        'arguments': {'config': '--sam2-config'},
    },
}
VOS_BACKENDS = {
    'cutie': {
        'class': ('bivosia.segmentation.backends.cutie', 'CutieSegmenter'),
        'label': 'Cutie',
        'checkpoint': EXTERNAL_DIRECTORY / 'cutie/checkpoints/cutie-base-mega.pth',
        'checkpoint_suffixes': ('.pth', '.pt'),
        'backbones': ('resnet18-5c106cde.pth', 'resnet50-19c8e357.pth'),
        'reference_modes': ('start', 'ends', 'all'),
        'defaults': {
            'internal_size': 480, 'memory_every': 10, 'long_term': True,
            'memory_min': 5, 'memory_max': 10, 'long_term_tokens': 10000,
            'mixed_precision': False,
        },
        'options': (
            ('internal_size', 'Internal size', -1, 8192),
            ('memory_every', 'Memory every', 1, 1000),
            ('long_term', 'Long-term memory', None, None),
            ('memory_min', 'Memory min', 2, 1000),
            ('memory_max', 'Memory max', 3, 1000),
            ('long_term_tokens', 'Long-term tokens', 4096, 1000000),
            ('mixed_precision', 'Mixed precision', None, None),
        ),
        'help': {
            'internal_size': 'Maximum internal short edge; -1 keeps native size',
            'memory_every': 'Store working memory every N frames; no frames are skipped',
        },
    },
}

MODEL_INSTALL_LOCK = EXTERNAL_DIRECTORY / 'model_install.lock'
MODEL_INSTALLATIONS = {
    'cutie': {
        'repository': 'https://github.com/hkchengrex/Cutie.git',
        'revision': 'ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a',
        'source': EXTERNAL_DIRECTORY / 'cutie/Cutie',
        'requirements': RESOURCE_DIRECTORY / 'requirements/requirements-vos.txt',
        'build': ('hatchling', 'editables'),
        'python': (3, 10),
        'weights': (
            ('cutie-base-mega.pth', 'https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth',
             'md5', 'a6071de6136982e396851903ab4c083a'),
            ('resnet18-5c106cde.pth', 'https://download.pytorch.org/models/resnet18-5c106cde.pth', 'sha256', '5c106cde'),
            ('resnet50-19c8e357.pth', 'https://download.pytorch.org/models/resnet50-19c8e357.pth', 'sha256', '19c8e357'),
        ),
    },
    'sam2': {
        'repository': 'https://github.com/facebookresearch/sam2.git',
        'revision': '2b90b9f5ceec907a1c18123530e92e794ad901a4',
        'source': EXTERNAL_DIRECTORY / 'sam/sam2',
        'requirements': RESOURCE_DIRECTORY / 'requirements/requirements-sam2.txt',
        'build': ('setuptools>=64', 'wheel'),
        'python': (3, 10),
        'weights': (
            ('sam2.1_hiera_tiny.pt',
             'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt',
             'sha256', '7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69'),
        ),
    },
}

LINKED_MASK_STREAMS = ('forward', 'reverse', 'preferred')

DATA_FORMATS = {
    'masks': {
        'hdf5': {
            'suffixes': ('.h5', '.hdf5', '.hdf'),
            'reader': ('bivosia.dataset.hdf5_masks', 'Hdf5Masks'),
            'writer': ('bivosia.dataset.hdf5_masks', 'stage_masks'),
            'inspector': ('bivosia.dataset.hdf5_io', 'inspect_arrays'),
        },
        'npy': {
            'suffixes': ('.npy',),
            'reader': ('bivosia.dataset.numpy_masks', 'NpyMasks'),
            'writer': ('bivosia.dataset.numpy_masks', 'stage_npy_masks'),
            'inspector': ('bivosia.dataset.numpy_io', 'inspect_npy'),
            'companions': {'validity': '{stem}.valid.npy'},
        },
        'npz': {
            'suffixes': ('.npz',),
            'reader': ('bivosia.dataset.numpy_masks', 'NpzMasks'),
            'writer': ('bivosia.dataset.numpy_masks', 'stage_npz_masks'),
            'inspector': ('bivosia.dataset.numpy_io', 'inspect_npz'),
            'scratch': {'buffer': '{name}.partial.npy'},
        },
    },
    'linked_masks': {
        'hdf5': {
            'suffixes': ('.h5', '.hdf5', '.hdf'),
            'reader': ('bivosia.dataset.hdf5_masks', 'linked_mask_streams'),
            'inspector': ('bivosia.dataset.hdf5_io', 'inspect_arrays'),
        },
    },
    'keypoints': {
        'sleap': {
            'suffixes': ('.h5', '.hdf5', '.hdf'),
            'reader': ('bivosia.dataset.sleap', 'SleapPoints'),
            'inspector': ('bivosia.dataset.hdf5_io', 'inspect_arrays'),
        },
    },
    'video': {
        'opencv': {
            'suffixes': ('.mp4', '.avi', '.mov', '.mkv'),
            'reader': ('bivosia.dataset.opencv_video', 'OpenCvVideo'),
            'inspector': ('bivosia.dataset.opencv_video', 'inspect_video'),
        },
    },
    'arrays': {
        'npy': {
            'suffixes': ('.npy',),
            'reader': ('bivosia.dataset.numpy_io', 'read_array'),
            'writer': ('bivosia.dataset.numpy_io', 'array_output'),
            'inspector': ('bivosia.dataset.numpy_io', 'inspect_npy'),
        },
    },
    'documents': {
        'json': {
            'suffixes': ('.json',),
            'reader': ('bivosia.dataset.jsonio', '_read_json'),
            'writer': ('bivosia.dataset.jsonio', '_atomic_json'),
            'temporary': {'prefix': '.{name}.', 'suffix': '.tmp'},
        },
    },
    'text': {
        'plain': {
            'suffixes': ('.txt',),
            'reader': ('bivosia.dataset.text', 'read_text'),
            'syntax': 'plain',
        },
        'json': {
            'suffixes': ('.json',),
            'reader': ('bivosia.dataset.text', 'read_text'),
            'syntax': 'json',
        },
        'markdown': {
            'suffixes': ('.md', '.markdown'),
            'reader': ('bivosia.dataset.text', 'read_text'),
            'syntax': 'markdown',
        },
    },
    'containers': {
        'hdf5': {
            'suffixes': ('.h5', '.hdf5', '.hdf'),
            'reader': ('bivosia.dataset.hdf5_io', 'open_hdf5'),
            'writer': ('bivosia.dataset.hdf5_io', 'hdf5_output'),
            'inspector': ('bivosia.dataset.hdf5_io', 'inspect_arrays'),
        },
    },
}
JOB_TERMINAL_STATES = ("complete", "failed", "cancelled")
DATASET_VIEWER_DEFAULTS = {'max_text_bytes': 1024 * 1024}

# Groups follow module names in analysis/.
ANALYSIS_DEFAULTS = {
    'mask_cleanup': {
        # Contour areas in full-resolution mask pixels squared; square kernel side in pixels.
        'contour_min_area': 1000,
        'contour_max_area': 100000,
        'overlap_kernel': 5,
    },
    'mask_preprocessing': {
        # Input mask pixels: full-resolution equivalents of 1100 px², 300 px, 3 px at 4× downsampling.
        'filter_by_area': True,
        'main_body_area': 17600,
        'filter_by_distance': True,
        'fragment_distance': 1200,
        'apply_opening': True,
        'opening_radius': 12,
    },
    'zone_detection': {
        'theta_quality': 0.60,
        'theta_recovery': 0.90,
        'theta_gap': 0,
        'expand_to_baseline': True,
    },
    'alignment': {
        'fallback_previous': False,
        'tie_relative_tolerance': 1e-12,
        'tie_absolute_tolerance': 1e-12,
    },
    'linking': {
        'tie_absolute_tolerance': 1e-7,
    },
    'point_masks': {
        'dilation_kernel': 5,
    },
    'shape_features': {
        'degree': 8,
        'z_score': False,
        'preprocess_masks': False,
    },
    'masks': {
        'count_dtype': 'uint32',
        'temporal_intersections': False,
    },
    'features': {
        'include_convex_hull': True,
        'include_hu_features': False,
        'include_fft_features': False,
        'fft_features': {
            'sample_count': 256,
            'low_frequency_cutoff': 8,
            'perimeter_cutoffs': (8, 16, 32),
            'bending_cutoff': 32,
        },
    },
}

# Groups follow module names in processing/.
PROCESSING_DEFAULTS = {
    'remote_images': {
        'timeout_seconds': 300,
    },
    'quantify': {
        'workers': None,
        'overlap_batch_size': 512,
        'quality_batch_size': 16384,
    },
    'features': {
        'workers': None,
        'batch_size': 256
    },
    'alignment': {
        'workers': 1,
        'block_size': 512
    },
    'keypoints': {
        'block_size': 16384
    },
    'shape_features': {
        'block_size': 32,
    },
    'jobs': {
        'heartbeat_seconds': 5,
        'stale_seconds': 30
    },
}

DATASET_LAYOUT = {
    "storage": "dataset",
    "files": {
        "description": {"name": "dataset.json"},
    },
    "folders": {
        "recording": {
            "name": "recording",
            "storage": "external",
            "files": {
                "video": {"name": "{filename}", "data": "video"},
            },
        },
        "masks": {
            "name": "masks",
            "storage": "external",
            "data": "masks",
            "files": {
                "forward": {"name": "masks_forward.h5"},
                "reverse": {"name": "masks_reverse.h5"},
                "linked_masks": {"name": "{annotation}.linked.h5"},
            },
            "folders": {
                "key_frame": {
                    "name": "key_frame",
                    "files": {
                        "key_frames": {"name": "key_frames.json"},
                        "key_frame_masks": {"name": "key_frames.h5"},
                        "cleaned_key_frame_masks": {"name": "key_frames.cleaned.h5"},
                    },
                },
            },
        },
        "keypoints": {
            "name": "keypoints",
            "storage": "external",
            "files": {"keypoints": {"name": "keypoints.h5"}},
        },
        "annotations": {
            "name": "annotations",
            "files": {
                "annotation": {"name": "{prefix}seed_{seed:.2f}_recover_{recover:.2f}.json"},
            },
        },
        "experiment": {
            "name": "experiment",
            "files": {"reorganization": {"name": "file_reorganization.json"}},
        },
        "analysis": {
            "name": "analysis",
            "files": {
                "keypoint_alignment": {"name": "keypoint_alignment.h5"},
                "keypoint_displacement": {"name": "keypoint_displacement.h5"},
                "mask_intersections": {"name": "mask_intersections.npy"},
                "mask_temporal_intersections": {"name": "mask_temporal_intersections.npy"},
                "mask_areas": {"name": "mask_areas.npy"},
                "mask_iou_match": {"name": "mask_iou_match.npy"},
                "mask_quality": {"name": "mask_quality.npy"},
                "mask_missing": {"name": "mask_missing.npy"},
                "mask_switch": {"name": "mask_switch.npy"},
                "mask_quality_metadata": {"name": "mask_quality_metadata.json"},
                "mask_features": {"name": "mask_features.npy", "companions": {"metadata": "{stem}.json"}},
                "mask_measures": {"name": "mask_measures.npy", "companions": {"metadata": "{stem}.json"}},
                "mask_geometry": {"name": "mask_geometry.h5"},
                "mask_zernike": {"name": "mask_zernike.h5"},
                "mask_zernike_preprocessed": {"name": "mask_zernike.preprocessed.h5"},
                "mask_umap": {"name": "mask_umap.h5"},
                "mask_umap_preprocessed": {"name": "mask_umap.preprocessed.h5"},
                "mask_umap_zscore": {"name": "mask_umap_zscore.h5"},
                "mask_umap_zscore_preprocessed": {"name": "mask_umap_zscore.preprocessed.h5"},
            },
            "folders": {
                "cache": {"name": "cache"},
                "jobs": {
                    "name": "jobs",
                    "folders": {
                        "job": {
                            "name": "{job_id}",
                            "files": {
                                "job_settings": {"name": "job.json"},
                                "job_status": {"name": "status.json"},
                                "job_log": {"name": "run.log"},
                                "job_cancel": {"name": "cancel"},
                                "job_claim": {"name": "worker.lock"},
                            },
                        },
                    },
                },
            },
        },
    },
}

FILE_PATTERNS = {
    "partial": "{name}.partial",
    "lock": "{name}.lock",
    "download": "{name}.download",
}
