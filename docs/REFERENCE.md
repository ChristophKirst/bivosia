<img src="../bivosia/ui/assets/bivosia-icon.svg" width="72" alt="BiVOSIA icon" />

# BiVOSIA reference

[Quick guide](DOCUMENTATION.md) · [Project overview](../README.md) · [Remote setup](REMOTE_SETUP.md) · [Implementation notes](IMPLEMENTATION.md)

**From video to reviewed identities, shape features and aligned keypoints.**

![Forward and reverse animal-mask streams compared to find disagreements](../bivosia/ui/assets/bivosia-graphic.png)

## Contents

- [Installation](#installation)
- [Datasets and the application workflow](#datasets-and-the-application-workflow)
- [Key-frame annotation](#key-frame-annotation)
- [Video-object segmentation](#video-object-segmentation)
- [Remote installations](#remote-installations)
- [Zone review, identity linking and export](#zone-review-identity-linking-and-export)
- [Mask features and UMAP](#mask-features-and-umap)
- [Keypoint alignment](#keypoint-alignment)
- [Data formats and storage](#data-formats-and-storage)
- [Command-line examples and batch processing](#command-line-examples-and-batch-processing)
- [Code and development](#code-and-development)
- [License](#license)

## Installation

Use **Python 3.10+**. Run installation from the project folder containing `pyproject.toml`:

```bash
python -m pip install -e .
python -m bivosia.run.run_app
```

- `pip install -e .` installs this checkout for development; `pip install .` makes a regular installation.
- After installation, the `bivosia` command and `python -m bivosia.run.run_app` work from any directory.
- Dependencies and Python-version requirements are declared in `pyproject.toml`.
- Replace example paths and animal counts with your recording's values.
- Viewing, manual annotation and analysis work without segmentation models.
- Install BiVOSIA, its Python environment and GPU drivers yourself.

### Install Cutie and SAM

Use **Install** beside **Model** in the key-frame or VOS window. Installation
runs on the selected **Local** or **SSH** target.

| Model | Purpose | Installation |
|---|---|---|
| SAM2 | Key-frame masks from point prompts | Automatic; defaults to SAM2.1 Tiny. |
| SAM3 | Key-frame masks from point prompts | Manual package and authorized checkpoint; no automatic download. |
| Cutie | Propagate masks through video | Automatic, including its encoder checkpoints. |

The same installers are available from the command line:

```bash
python -m bivosia.run.run_install sam2
python -m bivosia.run.run_install cutie
```

- In a source/editable installation, sources and weights go into `external/sam/`
  and `external/cutie/` inside the outer project folder, beside `pyproject.toml`.
  The project's `.gitignore` excludes `/external/`.
- In a regular wheel installation, model downloads use `share/bivosia/external/`
  under the selected Python environment. Use a writable virtual environment; datasets stay separate.
- Packages install into the selected Python environment. Existing core package
  versions and modified source checkouts are preserved; conflicts are reported.
- **Restart the local app after installation.** Setup checks imports, checkpoints
  and devices; full inference is a separate step.
- Remote installation requires BiVOSIA to exist on that machine first.
  See [Remote setup](REMOTE_SETUP.md).

## Datasets and the application workflow

1. **New dataset** creates a description in an empty folder; **File → Open dataset**
   opens an existing one.
2. Select the dataset and open **Workflow**.
3. The **Dataset and video** gear opens dataset settings. **Select video…** records its path without moving the recording; **Locations…** edits local/mounted and remote storage folders. **Open Video** plays the selected recording.
4. Annotate key frames → run VOS → analyze masks → review/link zones → export.
5. Open **Mask features** or **Keypoint alignment** for further analysis.

![Six animals with reviewed identity masks](../bivosia/ui/assets/about.gif)

*Example recording: six animals with reviewed identity masks, shown at 3× speed.*

- **Green:** outputs are available. **Amber:** work is missing. **Purple:** saved
  annotations need review or validation.
- Changed annotations require a new export.
- Click a window entry to bring it forward; reopening an editor reuses its window.
- **File → Recent datasets** remembers datasets between launches.

### Browsing dataset files

Open **View files** to browse the dataset, mounted storage and remote locations.

- **Info** shows array names, shapes and dtypes, or video frame count and estimated
  duration. Header inspection does not load full arrays or decode video frames.
- Select a video for paused playback and frame selection.
- Select JSON, TXT or Markdown for a text preview. **Open in window** separates it;
  drag the divider to resize the embedded preview.
- Text previews accept UTF-8 files up to 1 MiB. **Refresh** reloads the listing and headers.
- **Locations…** adds or edits mounted folders and remote host/folder entries in `dataset.json`.
  Removing a location leaves its files in place.
- Remote listings show filenames, sizes, modification times and location colors.
  Content previews and array/video metadata are available for local and mounted files.
- Right-click a file, category or job folder for **Move**, **Copy** or **Delete…**.
  Folder actions include subfolders across the displayed locations. Files keep their relative paths.
- Jobs offer **Delete…** only. Execution records stay where they ran and are excluded
  from folder transfers and consolidation. Deleting a job leaves analysis results intact.
- Right-click **jobs → Clean up completed jobs…** to review and remove completed job
  records and logs. Failed and canceled jobs remain for inspection; nothing is deleted automatically.
- Move and Copy list registered locations; individual files also offer **Choose folder…**.
  Transfers work locally and over SSH and verify copied bytes.
- Folder conflicts ask which version to use. Move removes source copies matching the
  chosen version after copying succeeds; differing, unchosen versions remain.
  Individual file transfers refuse existing destinations.
- Delete confirms affected paths and file count before removal. Folder deletion removes
  listed contents and empty folders; active jobs block changes. The listing refreshes afterward.

### Organizing existing recordings

Inspect the proposed moves before applying them:

```bash
python -m bivosia.run.run_organize /path/to/collection
python -m bivosia.run.run_organize /path/to/collection --apply
```

- Loose videos create dataset folders; existing recording folders keep their names.
- Videos move to `recording/`, recognized masks to `masks/`, other data to `experiment/`.
- Existing standard folders stay in place. Results are not recalculated or converted.
- Moves stay on the same filesystem, verify file sizes/timestamps and never overwrite.
- Use `--data-folder NAME` for auxiliary clip folders; ambiguous video/mask pairs
  need manual assignment.
- Add `--save-reorganization` with `--apply` to save the move plan in
  `experiment/file_reorganization.json`.

### Analysis jobs and cancellation

- **Run** starts a background job; its detachable Processing panel shows logs, progress and cancellation.
- Keyframes, VOS, Mask analysis and Features have processing controls. Configure alignment in its settings and run it from Workflow.
- Workflow gear buttons open the same parameter panels without loading the editors. Dataset overrides are stored in `dataset.json` under `processing`.
- **Run** executes selected steps in order at one Local/SSH location. Saved inputs can satisfy dependencies without rerunning their producing steps.
- Closing an editor or Workflow leaves its submitted job running, including the remaining steps of an automatic run.
- UMAP checks cancellation before fitting and before saving. Cancellation during
  fitting waits for the library call to finish.

## Key-frame annotation

1. Open the key-frame editor and select **Add**, or right-click the timeline.
2. Label animals manually, or select **SAM2/SAM3** and place point prompts.
3. Inspect and accept the proposed masks, then **Save**.
4. Label **both crop endpoints** before running VOS.

- Manual labeling needs only the video.
- Unavailable models are grayed out; their tooltips explain missing requirements.
- CUDA is used when available, otherwise CPU. `--device` selects one explicitly.

### Remote SAM inference

- Choose **Processing → SSH** and select a [registered installation](#remote-installations).
- The model, checkpoint and device are checked on that host; local SAM is unnecessary.
- BiVOSIA sends the full-resolution image and prompts losslessly over SSH, then
  returns masks and scores to the local editor. **No remote dataset is required.**
- The remote session reuses the model and image embedding as prompts change.
  Changing the model/target or closing the editor releases it.
- A failed request reports an error; the next prediction starts a fresh session.
  No hosted endpoint or permanent service is needed.

### Prediction, correction and saving

| Action | Result |
|---|---|
| SAM / Predict selected | Predict all prompted animals / only the selected animal. |
| Accept selected / Reject selected | Keep or discard the selected proposal. |
| Approve all | Accept the remaining proposals. |
| Prompt | Refine points, then predict again. |
| Paint / Erase | Edit with the left mouse button; Radius is in image pixels. |
| Undo / Redo | Undo or redo a brush stroke. |
| Left / Right | Cycle animal IDs, skipping background. |
| Save | Write accepted masks; pending proposals are excluded. |

### Optional mask cleanup

- **Clean up** creates a separate preview. Paint, Erase and Undo/Redo edit that preview;
  **Discard** returns to the original masks or proposals.
- **Save cleaned** writes `masks/key_frame/key_frames.cleaned.h5`, preserving other
  cleaned frames and recording the parameters.
- Original annotations stay unchanged. Cleanup is never automatic; VOS uses the originals.
- Default contour area: **1,000–100,000 pixels²**, inclusive. Accepted contours are
  filled, including holes. A **5-pixel square dilation** identifies overlaps to clear
  from all cleaned animal masks. Values use the supplied resolution.

### Files and model requirements

- References: `masks/key_frame/key_frames.json` and `key_frames.h5`; no extracted images.
- ZoD/revision annotations belong in `annotations/`.
- SAM needs its upstream package, PyTorch/TorchVision and a checkpoint on the selected host.
- `--checkpoint` overrides the configured path. Use `--sam2-config` with a different SAM2 checkpoint.
- Weights download only through **Install**. SAM3 is always installed manually;
  see [model details and dependency constraints](IMPLEMENTATION.md#key-frame-editor).

## Video-object segmentation

1. Save the crop's endpoint key frames with consistent animal IDs.
2. Choose **Cutie**, the execution target and the forward/backward directions.
3. Set **References**: directional start, both interval endpoints or all labeled frames.
4. **Run**, inspect the results, then continue to mask analysis and zone review.

- Intermediate key frames divide the crop into intervals with fresh tracking state.
- Both directions retain the reference masks. Native output resolution is the default;
  optional output resizing uses nearest-neighbor labels.
- **Internal size** controls inference resolution. **Memory every** controls memory
  updates, not frame skipping.
- **Install** supplies Cutie and all three required checkpoints. **Check** reports
  missing files/imports; prediction itself does not download weights.
- Cutie is unnecessary for viewing results or exporting scripts.

### Jobs, scripts and results

- **Jobs** reconnects to saved job IDs; settings, status and logs live in `analysis/jobs/<id>/`.
- The processing console shows warnings in amber and errors, including tracebacks, in red.
- **Stop** requests cooperative cancellation. A lost SSH connection is not completion.
- **Export script** writes an editable `run_vos.py`; run it normally or with `--background`.
- Jobs depend on host logout policies and do not survive reboot. On a cluster, use
  an allocated compute node or your scheduler with the exported script.
- Outputs use uint8 `(frames, height, width)` masks in HDF5, NPY or NPZ, with validity
  tied to source frame indices. Existing script and mask files are protected.
- Failed or cancelled writes remove their incomplete outputs by default; job logs and
  completed results remain. Abrupt process termination can still leave temporary files.
- View remote masks directly. Copy or consolidate files through the dataset viewer when a local copy is required.
- A selected interval retains the full frame count and marks other frames invalid.
  **Full-recording quantification requires complete coverage.**

## Remote installations

Start with **Help → Remote setup** or the [remote setup guide](REMOTE_SETUP.md).

1. Install BiVOSIA and prepare Python on your workstation/server.
2. Open **Settings → Remote installations…** or **Manage** beside an installation selector.
3. Enter **Host**, **Python** and **Code**, then **Save** to check the connection and imports.
4. Select **Run on → SSH** in the key-frame or VOS window.

| Setting | Use |
|---|---|
| Host | SSH alias or `user@host`; configure ports and keys through OpenSSH. |
| Python | Absolute path to the remote environment's Python executable. |
| Code | Remote `bivosia/` package folder. Keep local and remote code in sync. |
| Default data folder | Optional server folder; a dataset subfolder is derived from its name. |
| Remote dataset | Resolved from `dataset.json`; manage it using **Locations…**. |

- Registration does not need model weights or a dataset. **Remove** only forgets the entry.
- Checkpoints are paths on the selected machine. **Install** can set up SAM2/Cutie there.
- Select **Processing / Run on → Local or SSH** for each job. Run prepares missing inputs in the dataset viewer, then continues automatically.
- **View files** owns dataset transfers: file **Copy / Move**, or **Consolidate…** for a complete copy.
  Consolidation presents conflicting filenames and lets you choose a source for each one before copying.
  Its destination description uses relative paths; sources remain in place.
- VOS checks inputs where it will run. SAM transfers its current image and prompts.
- Workflow status includes registered remote masks and analysis. The console reads local or SSH job logs.
- VOS, recording and ZoD viewers share the remote mask reader. It opens the existing format adapters
  on the server and transfers requested frames; plot arrays use a private temporary snapshot.
- SSH uses existing key/agent authentication and host-key checking.

## Zone review, identity linking and export

First run **Workflow → Open Masks → Run**, then choose **Open ZoDs**.
A Zone of Disagreement (ZoD) is an interval where the directional masks disagree.

1. **Preview** zone detection; **Automatic** creates and saves automatic directions and links.
2. Review each zone against the video. Choose **forward**, **reverse** or **invalid**
   in **Masks**, adjust Start/End with **Update**, then select **Approve →**. **Next →** advances without approving.
3. In **Linking annotation**, use **Next →** to inspect identity assignments at boundaries; **Approve →**
   records a verified assignment.
4. **Validate**, resolve remaining issues, then select **Export** in Workflow. Choose the processing location there and **Run**.

**ZoD detection**, **Zone annotation**, and **Linking annotation** are detachable panels.
Zone and linking annotations share tabs by default. Clicking a plotted annotation selects its zone in the list and updates Start, End and Masks. The lock button enables/disables zone edits in the plot; unlocked plots use Alt+drag. Field edits remain available while the plot is locked.

- Manual export requires reviewed zones and automatically resolved or approved boundaries.
- **Automatic** and automatic workflow runs use the same calculation. Adjacent boundaries use optimal mask overlap; across invalid intervals, incoming-direction labels propagate to the next valid frame before matching directions there. Link quality and warnings are retained; automatic proposals remain distinct from manual approvals.
- With an annotation open, **Automatic** asks whether to create a new annotation using the displayed parameters or fill unresolved links in the current annotation. Filling preserves zones, drafts, approvals and valid saved automatic links; it can be undone. New annotations use separate files.
- **Processing** shows progress and logs and allows cancellation. Results are saved locally to the editor's dataset only after calculation completes; cancellation leaves the annotation unchanged.
- Mask analysis computes framewise evidence only. Opening the editor or viewer loads saved links without generating proposals or changing review status.
- Automatic direction choices allow one temporarily missing animal as specified by the Methods. Original masks and existing manual annotations remain unchanged.
- The HDF5 bundle contains preferred consensus masks, both directions, validity and linking metadata.
- Source masks stay unchanged. Use the export for [mask features](#mask-features-and-umap).

### Automatic direction method

Select **Direction method** before starting an annotation set. The selection is
saved with its detector settings.

| Method | Proposal |
|---|---|
| Missing masks | Choose the direction with no missing masks. Both incomplete: invalid; both complete: forward. |
| Missing-mask counts | Invalid if either direction loses more than one mask in any frame. Otherwise choose fewer missing masks; equal nonzero totals: invalid; no missing masks: forward. |
| Missing-mask counts + temporal quality (default) | Apply the count rule; when no masks are missing, choose the direction with higher minimum temporal assignment quality. Ties: invalid. |

- The default follows the Methods manuscript. Temporal quality compares masks at
  `t` and `t+1`, using transitions starting within the zone; the final recording
  frame has no successor. No usable quality in either direction means invalid.
- Missing-mask and switch flags remain separate from quality; no −∞ zone scores.
- Existing saved choices and manual directions are preserved. New automatic
  proposals and interval edits use the selected rule.
- If temporal quality is unavailable, rerun **Workflow → Open Masks → Run**.
  Loading existing annotations does not require recalculating their proposals.

### Quantification outputs

| File | Shape / content |
|---|---|
| `mask_areas.npy` | `(2, frames, animals)`; forward then reverse, raw IDs 1…N. |
| `mask_intersections.npy` | `(frames, animals, animals)`; forward rows, reverse columns. |
| `mask_temporal_intersections.npy` | Optional `(2, frames-1, animals, animals)`; consecutive frames in chronological order. |
| `mask_temporal_quality.npy` | `(2, frames-1)`; minimum signed Hungarian assignment margin for each within-pass transition. |
| Match, quality, missing and switch arrays | Derived from the saved counts. |

- Counts are exact uint32; zero denotes a missing mask. Calculations promote counts to
  int64; frames above 4,294,967,295 pixels are rejected.
- IoU is derived as `I / (A + B - I)`, with NaN when either mask is absent.
- Temporal quality is calculated during mask analysis. Add `--temporal-intersections`
  to retain the full temporal count matrices as well. Zone detection thresholds are unchanged.
- Viewers read the count files; previous normalized IoU files are not a substitute.

## Mask features and UMAP

1. Save and validate the zone/link annotation.
2. Open **Workflow → Open features** and choose **Mask source** below.
3. Set a **Zernike radius** large enough to contain each animal around its centroid.
   The same radius applies to all samples; degree defaults to **8**.
4. Select **Start frame**, **End frame** and **Every Nth frame**. The range is inclusive; **Last** means the recording's final frame. Defaults use every frame.
5. Select **UMAP dimensions**: 2 (default) or 3. Optionally enable preprocessing or feature z-scoring, then **Run**.

Parameters and the detachable **Processing** panel are on the left. UMAP and video are side by side on the right, with the annotation timeline below.

- Click a UMAP point or timeline dot to show its video frame and outline that animal's mask.
- Click a video mask to select its UMAP point and timeline dot. Masks excluded from analysis can be outlined, but have no corresponding sample.
- Seek or play the video to highlight all samples at the current frame.
- The timeline's **UMAP** row shows sampled frames, with separate vertical positions for animals.
- **Color by: Sample** uses the same UMAP-2 gradient in the scatter plot and timeline; nearby values can share similar colors. **Animal** uses global identity colors.
- In 3D, right-drag rotates and the mouse wheel zooms. Left-click selects a sample.
- In 2D, drag to pan and use the mouse wheel to zoom. **Show masks** opens a separate plot of the current extent: each occupied grid cell shows its nearest-to-center sample, while empty cells stay blank. Thumbnails retain mask orientation and aspect ratio, using the selected preprocessing and colors. Click a mask to select its frame and animal.
- The video displays the selected consensus masks when available locally. Saved embeddings remain viewable when the local video is unavailable.

| Mask source | Behavior |
|---|---|
| Automatic (default) | Use a matching export if available; otherwise use forward/reverse masks and the saved annotation. |
| Exported masks | Require an export matching the selected annotation. |
| Masks + annotation | Apply saved direction choices and global identity links while reading the original masks. |

- The source setting is shared with the Workflow settings gear. Export is optional.
- Reading masks uses saved links; it does not generate or save new annotations.
- [Mahotas](https://mahotas.readthedocs.io/en/latest/api.html#mahotas.features.zernike_moments)
  calculates absolute Zernike moments, normalizing foreground weights by their sum.
- **Downsample factor** defaults to 1. A factor of 2 halves width and height with nearest-neighbor sampling before preprocessing. Source files stay unchanged.
- Radius, distances, areas and opening use the downsampled pixel grid; their numeric settings are not converted automatically.
- Masks are centered without individually normalizing their size. Invalid frames and absent animals are skipped.
- If an animal mask exceeds the Zernike radius, only that sample is excluded. The console warns with its frame, animal ID and required radius; other masks continue.
- Each feature row keeps its source frame and global animal ID.

### Optional preprocessing

**Preprocess masks** is off by default. When enabled, its independently selectable
steps run in this order for each animal:

| Step | Behavior | Default (at factor 1) |
|---|---|---|
| Filter by area | Keep only the largest 8-connected component if it exceeds Main body area; otherwise retain all. | 17,600 pixels² |
| Filter by distance | Keep the two largest components if their centroids are close enough; otherwise only the largest. Remove further components. | 1,200 pixels |
| Morphological opening | Erode then dilate with a disk to remove thin protrusions, such as tails. Zero skips opening. | Radius 12 pixels |

- All three steps start selected. Disable any step independently; disabling all leaves masks unchanged.
- Preprocessing runs before Zernike extraction and never modifies source masks or key frames.
- Completely removed animals are omitted from feature extraction.
- For masks downsampled fourfold per dimension, the methods' values are **1,100 / 300 / 3**.
  Set them explicitly when selecting **Downsample factor: 4**.

### UMAP and feature scaling

- **Z-score features before UMAP** is optional and off by default.
- Z-scoring uses each column's population standard deviation across all samples;
  constant columns become zero. Saved Zernike vectors stay unchanged.
- [Standard UMAP](https://umap-learn.readthedocs.io/en/latest/api.html) calculates the selected 2D or 3D embedding;
  other settings use library defaults: **15 neighbors, Euclidean distance, `min_dist=0.1`**.
- Sampling is applied before feature extraction; invalid frames and absent masks remain excluded. Fitting needs at least four samples for 2D or five for 3D, and variation
  between vectors; no random seed is imposed.
- These are standard descriptors and embedding settings, not the manuscript-specific settings.

### Saved results

| Result | File in `analysis/` |
|---|---|
| Zernike vectors | `mask_zernike.h5` |
| 2D UMAP | `mask_umap.h5` |
| 2D UMAP with z-scoring | `mask_umap_zscore.h5` |
| 3D UMAP | `mask_umap_3d.h5` |
| 3D UMAP with z-scoring | `mask_umap_3d_zscore.h5` |

- Preprocessed results add `.preprocessed` before `.h5`.
- Subsets add `.frames_START-END_step_N` before `.h5`; an open-ended range uses `last` for END. Full-recording files are preserved.
- Downsampled results add `.downsample_FACTOR` and the frame-selection suffix. Their metadata records both source and analysis image dimensions.
- Matching results are reused. Changing dimensions or a checkbox selects its corresponding saved
  result; **Run** calculates missing embeddings from existing descriptors when possible.
- Files record the source, settings and package versions. Completed results with different
  settings are not overwritten. Failed writes clean up their own incomplete outputs.

## Keypoint alignment

1. Open **Dataset settings → Select keypoints…**, choose a SLEAP analysis HDF5 file,
   enter its original image height and width, and save. The file is validated and
   copied into the configured `keypoints/` storage folder if it is not already
   in a registered keypoints folder. The source and existing files are preserved.
2. Open **Keypoint alignment**, choose the annotation and select keypoints.
3. Use the saved animal count, or enter it explicitly if unavailable.
4. Optionally enable previous-assignment fallback for ties, then run alignment.

Example SLEAP settings:

```json
{
  "keypoints": {
    "format": "sleap",
    "file": "keypoints.h5",
    "image_shape": [1536, 2048]
  }
}
```

- SLEAP files retain their native layout in `keypoints/`. Optional `invalid_frames`
  lists excluded source frame indices.
- Select the same file again to edit its source dimensions. Existing alignment
  or displacement results must be moved out of the dataset before changing their
  source file or dimensions.
- Points are scaled from their source resolution into mask pixels, then rounded
  to the nearest pixel. Stored coordinates stay unchanged.
- Alignment uses the selected annotation's mask direction and skips invalid frames.
  Zero overlap stays unmatched; ties are recorded.
- `--fallback-previous` reuses compatible assignments for ties, including initial ties
  propagated backward. It requires consistent mask identities across frames.
- **Workers** means parallel CPU processes; **Automatic** uses available CPUs up to
  the number of processing blocks. Existing results are protected.
- In the viewer, enable **Keypoints** and choose **Original label**, **Assigned label**
  or **Both**. Both shows the original color inside the assigned color.
- Alignment algorithms use generic `KeypointReader`; SLEAP is the current file-reader specialization.

## Data formats and storage

### Dataset layout

```text
dataset/
  dataset.json
  recording/          # video
  masks/              # forward, reverse and reviewed exports
    key_frame/        # key_frames.json and key_frames.h5
  keypoints/          # optional source keypoints
  annotations/        # zones, links and review history
  experiment/         # other imported data
  analysis/           # counts, alignment and shape results
    jobs/<job-id>/    # settings, status, logs and cancellation
```

- `config.py:DATASET_LAYOUT` defines exact names and storage placement.
- `Dataset.file('key_frames')` resolves the key-frame JSON; companions are selected
  with `companion='metadata'` where declared.
- Frame/prefetch caches stay in memory; `analysis/cache/` currently has no persistent output.

### Mask formats

All masks use `(frames, height, width)` labels: **0 = background**, positive IDs = animals.

| Format | Storage and access |
|---|---|
| HDF5: `.h5`, `.hdf5`, `.hdf` | `masks` and boolean `valid` datasets; block access. |
| NPY: `.npy` | Memory-mapped labels; optional `<stem>.valid.npy`. No sidecar means all frames valid. |
| NPZ: `.npz` | Compressed `masks` and `valid`; reading loads arrays into memory. |

Choose explicit forward/reverse filenames in `dataset.json` when needed:

```json
{"masks": {"forward": "masks_forward.npy", "reverse": "masks_reverse.npz"}}
```

- Paths are relative to `masks/`; a unique standard filename is found automatically.
- Writers refuse existing results and retain incomplete output separately.
- The reviewed linked-mask export is a separate HDF5 bundle with preferred/directional
  streams, validity and linking metadata. `open_linked_masks` opens its named streams.

### External storage and dataset settings

```json
{
  "storage_folders": ["/external/disk/experiment"],
  "remote_locations": [{"host": "workstation", "path": "/data/experiment"}],
  "video": "recording.mp4",
  "annotation": "annotations/hand_seed_0.60_recover_0.90.json"
}
```

- Storage roots may be absolute or dataset-relative; each uses the same logical layout.
- Files are searched across all roots. Duplicate logical filenames are an error;
  directories may span roots.
- Video filenames are relative to `recording/`; annotation paths are dataset-relative.
  A single video or annotation is selected automatically.
- `recording/`, `masks/` and `keypoints/` write to the first external root when configured;
  otherwise they use the dataset. `annotations/` and `analysis/` stay in the dataset.
- Use **View files → Locations…** to manage mounted storage roots and SSH locations.
  Remote paths are absolute filesystem paths, not `sftp://` URLs. One dataset folder is
  selected per SSH host; the registered installation supplies Python and code locations.
- Unavailable write roots are errors. Move files into the configured write location
  before rewriting; BiVOSIA does not silently fall back or create duplicate copies.
- Linking exports preserve source masks. Readers validate frame counts, dimensions,
  label IDs and annotation bounds.

## Command-line examples and batch processing

After installation, run from any directory. Use `--help` for each launcher's options.

```bash
# Open datasets in the app
python -m bivosia.run.run_app /path/to/dataset /path/to/another/dataset

# Annotate and track
python -m bivosia.run.run_key_frame_editor /path/to/dataset --animals 6 --model sam2
python -m bivosia.run.run_vos /path/to/dataset

# Quantify, review and view
python -m bivosia.run.run_analysis quantify /path/to/dataset --n-animals 6
python -m bivosia.run.run_zone_revision /path/to/dataset
python -m bivosia.run.run_viewer /path/to/dataset

# Additional mask measurements and keypoint alignment
python -m bivosia.run.run_analysis features /path/to/dataset --n-animals 6
python -m bivosia.run.run_analysis align /path/to/dataset --n-animals 2 --workers 4
```

### Processing multiple recordings

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m bivosia.run.run_analysis quantify /path/to/collection \
  --collection --n-animals 2 --workers 64 --temporal-intersections
```

- `--collection` selects immediate subfolders containing `dataset.json` or `masks/`.
- All mask pairs are checked before processing. `--workers` is the **total** shared
  worker count across recordings.
- Results stay in each dataset's `analysis/`. **Existing quantification outputs are
  replaced**; source masks and annotations stay unchanged.
- Omit `--temporal-intersections` to skip temporal overlaps.
- Quantification needs NumPy, SciPy and h5py. Collections must use the current dataset layout.

## Code and development

```text
bivosia/                  # project root
  pyproject.toml          # package metadata and dependencies
  README.md
  LICENSE
  docs/
  requirements/           # optional model dependencies
  tests/
  bivosia/                # importable package
    __init__.py
    config.py
    analysis/
    dataset/
      io/
      annotations/
      storage/
      workflow/
    processing/
      analysis/
      segmentation/
      storage/
      remote/
    segmentation/
    ui/
    run/
```

- Imports remain `bivosia.xxx`; the outer folder is the project, not another Python package.
- Package version comes from `bivosia/__init__.py`. The `bivosia` command launches the application.
- Wheels include the help pages, model requirement files and UI graphics/animation.
  Editable installs read the documentation directly from this checkout.
- Build a source archive and wheel from the project root:

```bash
python -m pip install build
python -m build
```


| Folder | Responsibility |
|---|---|
| `dataset/` | Dataset layout, recordings and data-specific readers. |
| `dataset/io/` | Reader interfaces, format registry, adapters and array transport. |
| `dataset/annotations/` | `keyframes.py` manages segmentation annotations; `masks.py` manages mask selection, review and identity-link annotations. |
| `dataset/storage/` | Locations, inventory, file operations and remote readers. |
| `dataset/workflow/` | Processing settings, input validation, job records and completion status. |
| `segmentation/` | Segmentation adapters. |
| `analysis/` | Calculations, quality, detection, revision and linking. |
| `processing/` | Shared job lifecycle, workflow execution, workers and logging. |
| `processing/analysis/` | Recording/block analysis, progress and outputs. |
| `processing/segmentation/` | Key-frame prediction, video segmentation and model installation. |
| `processing/storage/` | File/folder transfers, consolidation and input synchronization. |
| `processing/remote/` | Shared SSH client, request dispatch and persistent sessions. |
| `ui/` | Shared widgets, presentation and interaction. |
| `run/` | Application and command-line entry points. |
| `docs/` | User guides and implementation notes. |
| `requirements/` | Optional model dependency files. |

- Shared names, formats, storage layout and defaults live in `config.py`.
- Use absolute `bivosia` imports. Shared UI components own layout and styling.
- **Key frames** are segmentation references; **seed frames** belong to ZoD detection.
- Dataset-specific choices belong in `dataset.json`; explicit processing options override defaults.
- Extra raw mask features: Hu features default off (`--hu-features` enables them);
  convex-hull measures default on (`--no-convex-hull` disables them); spectral
  features are independently enabled with `--fft-features`.
- Historical code remains outside the project in the workspace's `_previous/` folder
  and is not distributed. Its viewer reference is `_previous/posthoc/docs/analysis.md`;
  see [implementation notes](IMPLEMENTATION.md) for the packaged internals.

### Python usage and focused checks

Install with `python -m pip install -e .` from the project root, then start Python
from any directory. Imports continue to use `bivosia`, for example:

```python
from bivosia.dataset.layout import Dataset

dataset = Dataset('/path/to/dataset')
print(dataset.file('key_frames'))
```

```bash
python -m doctest bivosia/analysis/mask_features.py
```

- Prefer small executable docstring examples and focused checks.
- GUI/recording-dependent examples use `doctest: +SKIP`; substitute your own inputs.
- Qt checks need a display or `QT_QPA_PLATFORM=offscreen`; checks must preserve saved annotations.

### Local examples and archived code

- The standard viewer/editor defaults to `data/analysis` beside the source checkout, or under the current directory for an installed package.
- The threshold viewer is archived at `../../_previous/run/run_zone_threshold.py`.
- Analysis and editing do not require segmentation models or a GPU.

## License

BiVOSIA uses [GPL-3.0-only](../LICENSE). Third-party software, model weights and
archived components retain their own licenses and notices.
