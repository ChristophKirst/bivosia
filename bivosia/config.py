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

EXTERNAL_DIRECTORY = (PROJECT_DIRECTORY / 'external' if SOURCE_CHECKOUT
                      else Path(sys.prefix) / 'share/bivosia/external')

DEFAULT_DATASET = (PROJECT_DIRECTORY.parent if SOURCE_CHECKOUT else Path.cwd()) / 'data/analysis'

APPLICATION_SETTINGS = ('BiVOSIA', 'Application')

FONT_SIZE_SETTING = 'font_point_size'

REMOTE_SETTINGS = {'installations': 'remote_installations', 'datasets': 'remote_datasets'}

REMOTE_PROTOCOL_VERSION = 1

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

CONSENSUS_MASK_DEFAULTS = {'source': 'auto'}
CONSENSUS_MASK_SOURCES = {
    'auto': {'label': 'Automatic', 'inputs': ('consensus_masks', 'annotations')},
    'export': {'label': 'Exported masks', 'inputs': ('export', 'annotations')},
    'annotation': {'label': 'Masks + annotation', 'inputs': ('masks', 'analysis', 'annotations')},
}

DATA_FORMATS = {
    'masks': {
        'hdf5': {
            'suffixes': ('.h5', '.hdf5', '.hdf'),
            'reader': ('bivosia.dataset.io.hdf5', 'Hdf5MaskReader'),
            'writer': ('bivosia.dataset.io.hdf5', 'stage_masks'),
            'inspector': ('bivosia.dataset.io.hdf5', 'inspect_arrays'),
        },
        'npy': {
            'suffixes': ('.npy',),
            'reader': ('bivosia.dataset.io.numpy', 'NpyMaskReader'),
            'writer': ('bivosia.dataset.io.numpy', 'stage_npy_masks'),
            'inspector': ('bivosia.dataset.io.numpy', 'inspect_npy'),
            'companions': {'validity': '{stem}.valid.npy'},
        },
        'npz': {
            'suffixes': ('.npz',),
            'reader': ('bivosia.dataset.io.numpy', 'NpzMaskReader'),
            'writer': ('bivosia.dataset.io.numpy', 'stage_npz_masks'),
            'inspector': ('bivosia.dataset.io.numpy', 'inspect_npz'),
            'scratch': {'buffer': '{name}.partial.npy'},
        },
    },
    'linked_masks': {
        'hdf5': {
            'suffixes': ('.h5', '.hdf5', '.hdf'),
            'reader': ('bivosia.dataset.io.hdf5', 'linked_mask_streams'),
            'inspector': ('bivosia.dataset.io.hdf5', 'inspect_arrays'),
        },
    },
    'keypoints': {
        'sleap': {
            'suffixes': ('.h5', '.hdf5', '.hdf'),
            'reader': ('bivosia.dataset.io.sleap', 'SleapKeypointReader'),
            'inspector': ('bivosia.dataset.io.hdf5', 'inspect_arrays'),
        },
    },
    'video': {
        'opencv': {
            'suffixes': ('.mp4', '.avi', '.mov', '.mkv'),
            'reader': ('bivosia.dataset.io.opencv', 'OpenCvVideoReader'),
            'inspector': ('bivosia.dataset.io.opencv', 'inspect_video'),
        },
    },
    'arrays': {
        'npy': {
            'suffixes': ('.npy',),
            'reader': ('bivosia.dataset.io.numpy', 'read_array'),
            'writer': ('bivosia.dataset.io.numpy', 'array_output'),
            'inspector': ('bivosia.dataset.io.numpy', 'inspect_npy'),
        },
    },
    'documents': {
        'json': {
            'suffixes': ('.json',),
            'reader': ('bivosia.dataset.io.json', '_read_json'),
            'writer': ('bivosia.dataset.io.json', '_atomic_json'),
            'temporary': {'prefix': '.{name}.', 'suffix': '.tmp'},
        },
    },
    'text': {
        'plain': {
            'suffixes': ('.txt',),
            'reader': ('bivosia.dataset.io.text', 'read_text'),
            'syntax': 'plain',
        },
        'json': {
            'suffixes': ('.json',),
            'reader': ('bivosia.dataset.io.text', 'read_text'),
            'syntax': 'json',
        },
        'markdown': {
            'suffixes': ('.md', '.markdown'),
            'reader': ('bivosia.dataset.io.text', 'read_text'),
            'syntax': 'markdown',
        },
    },
    'containers': {
        'hdf5': {
            'suffixes': ('.h5', '.hdf5', '.hdf'),
            'reader': ('bivosia.dataset.io.hdf5', 'open_hdf5'),
            'writer': ('bivosia.dataset.io.hdf5', 'hdf5_output'),
            'inspector': ('bivosia.dataset.io.hdf5', 'inspect_arrays'),
        },
    },
}

JOB_TERMINAL_STATES = ("complete", "failed", "cancelled")

AUTOMATIC_ANNOTATION_PREFIX = 'automatic_{run_id}_'

WORKFLOW_STEPS = {
    'vos': {'label': 'VOS', 'inputs': ('video', 'key_frames'), 'output': 'masks'},
    'quantify': {'label': 'Mask analysis', 'inputs': ('masks',), 'output': 'analysis'},
    'detect_link': {'label': 'ZoD detection and linking', 'inputs': ('masks', 'analysis'), 'output': 'annotations'},
    'export': {'label': 'Export', 'inputs': ('masks', 'analysis', 'annotations'), 'output': 'export'},
    'shape_features': {'label': 'Mask features',
                       'inputs': CONSENSUS_MASK_SOURCES[CONSENSUS_MASK_DEFAULTS['source']]['inputs'],
                       'output': 'features'},
    'align': {'label': 'Keypoint alignment', 'inputs': ('masks', 'annotations', 'keypoints'), 'output': 'alignment'},
}

DATASET_VIEWER_DEFAULTS = {'max_text_bytes': 1024 * 1024}

AUTO_DIRECTION_METHODS = {
    'missing_masks': 'Missing masks',
    'missing_mask_counts': 'Missing-mask counts',
    'temporal_quality': 'Missing-mask counts + temporal quality',
}


# Groups follow module names in analysis/.
ANALYSIS_DEFAULTS = {
    'zones_of_disagreement': {
        'theta_quality': 0.60,
        'theta_recovery': 0.90,
        'theta_gap': 0,
        'expand_to_baseline': True,
        'direction_method': 'temporal_quality',
    },
    'alignment': {
        'dilation_kernel': 5,
        'fallback_previous': False,
        'tie_relative_tolerance': 1e-12,
        'tie_absolute_tolerance': 1e-12,
    },
    'mask_linking': {
        'tie_absolute_tolerance': 1e-7,
    },
    'embedding': {
        'n_components': 2,
        'z_score': False,
    },
    'mask_quality': {
        'count_dtype': 'uint32',
        'temporal_intersections': False,
    },
    'masks': {
        'cleanup': {
            # Contour areas in full-resolution mask pixels squared; square kernel side in pixels.
            'contour_min_area': 1000,
            'contour_max_area': 100000,
            'overlap_kernel': 5,
        },
        'preprocessing': {
            # Input mask pixels: full-resolution equivalents of 1100 px², 300 px, 3 px at 4× downsampling.
            'filter_by_area': True,
            'main_body_area': 17600,
            'filter_by_distance': True,
            'fragment_distance': 1200,
            'apply_opening': True,
            'opening_radius': 12,
        },
    },
    'mask_features': {
        'include_convex_hull': True,
        'include_hu_features': False,
        'include_fft_features': False,
        'fft_features': {
            'sample_count': 256,
            'low_frequency_cutoff': 8,
            'perimeter_cutoffs': (8, 16, 32),
            'bending_cutoff': 32,
        },
        'zernike': {
            'degree': 8,
            'downsample_factor': 1,
            'preprocess_masks': False,
        },
    },
}

# Groups follow packages and module names in processing/.
PROCESSING_DEFAULTS = {
    'remote': {
        'session': {
            'timeout_seconds': 300,
        },
    },
    'analysis': {
        'mask_quality': {
            'workers': None,
            'overlap_batch_size': 512,
            'quality_batch_size': 16384,
        },
        'mask_features': {
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
        'mask_shape': {
            'block_size': 32,
            'start_frame': 0,
            'end_frame': -1,
            'frame_step': 1,
        },
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
                "mask_temporal_quality": {"name": "mask_temporal_quality.npy"},
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
                "mask_zernike_sampled": {"name": "mask_zernike{preprocessed}.frames_{start_frame}-{end_frame}_step_{frame_step}.h5"},
                "mask_umap_sampled": {"name": "mask_umap{dimensions}{z_score}{preprocessed}.frames_{start_frame}-{end_frame}_step_{frame_step}.h5"},
                "mask_zernike_downsampled": {"name": "mask_zernike{preprocessed}.downsample_{downsample_factor}.frames_{start_frame}-{end_frame}_step_{frame_step}.h5"},
                "mask_umap_downsampled": {"name": "mask_umap{dimensions}{z_score}{preprocessed}.downsample_{downsample_factor}.frames_{start_frame}-{end_frame}_step_{frame_step}.h5"},
                "mask_umap": {"name": "mask_umap.h5"},
                "mask_umap_preprocessed": {"name": "mask_umap.preprocessed.h5"},
                "mask_umap_zscore": {"name": "mask_umap_zscore.h5"},
                "mask_umap_zscore_preprocessed": {"name": "mask_umap_zscore.preprocessed.h5"},
                "mask_umap_3d": {"name": "mask_umap_3d.h5"},
                "mask_umap_3d_preprocessed": {"name": "mask_umap_3d.preprocessed.h5"},
                "mask_umap_3d_zscore": {"name": "mask_umap_3d_zscore.h5"},
                "mask_umap_3d_zscore_preprocessed": {"name": "mask_umap_3d_zscore.preprocessed.h5"},
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
}
