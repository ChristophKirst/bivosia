<img src="../bivosia/ui/assets/bivosia-icon.svg" width="64" alt="BiVOSIA icon" />

# BiVOSIA guide

**Video → Key frames → VOS → Mask analysis → ZoD detection and linking → Export**

[Get started](#installation) · [Workflow](#datasets-and-the-application-workflow) ·
[Examples](#command-line-examples-and-batch-processing) · [Authors](#authors) · [Paper](#paper)

## Installation

Use **Python 3.10+**. From the project folder containing `pyproject.toml`:

```bash
python -m pip install -e .
python -m bivosia.run.run_app
```

- `-e .` keeps this checkout editable; use `pip install .` for a regular installation.
- After installation, `bivosia` or the module command starts the app from any directory.
- **Manual annotation, viewing and analysis:** no segmentation model needed.
- **SAM2 / Cutie:** use **Install** beside **Model**, then restart the local app.
- **SAM3:** install its package and authorized checkpoint manually.

[Installation details](REFERENCE.md#installation) · [Remote setup](REMOTE_SETUP.md)

## Datasets and the application workflow

Choose **File → New dataset** or **Open dataset**, then open **Workflow**.

| Step | What to do |
|---|---|
| **1 · Video** | Select the recording through the row’s gear; **Open Video** plays it. |
| **2 · Key frames** | Label the animals at both crop endpoints. |
| **3 · VOS** | Track masks forward and backward with Cutie. |
| **4 · Mask analysis** | Open its window, select Local or SSH, then Run to calculate agreement. |
| **5 · ZoD detection and linking** | Detect and review disagreements; verify animal identities. |
| **6 · Export** | Validate and save reviewed masks for further analysis. |

**Green** = outputs available · **Amber** = work missing · **Purple** = review needed.

- **View files:** inspect files, arrays and video; preview text or video by clicking.
- **Dataset and video gear:** shows the dataset folder, selects the experimental video and keypoints, and opens **Locations…** for local/remote storage settings.
- **Viewer** in Workflow, or **View analysis** in the main window: open the video with available masks, quality scores and zones.
  Missing results leave their plots empty; mask analysis is not required to open the viewer.
- **Open Masks:** select Local or SSH, set the animal count, and Run.
  Each processing window selects its own execution location.
- **Gear buttons:** open a step’s parameters without its editor. Settings are shared with the editor and saved in the dataset.
- **Run:** check steps in the table’s rightmost column, adjust their settings with the gears, and use **Run** at the bottom of that column. Choose one Local/SSH location in Processing. Required inputs may come from saved results or selected earlier steps.
- **Processing panels:** Keyframes, VOS, Mask analysis and Features have detachable controls. ZoDs uses the same progress, logs and cancellation for Automatic. Drag the Processing title bar out to detach it.
- **Recent datasets:** reopen a dataset. Click a window entry to bring it forward.
- **Viewer menus:** use **File** to change datasets, open **Workflow**, **View files**, or the main BiVOSIA window. **Help** has the same documentation and About dialog as the main window.
- **Text size:** Ctrl+plus/minus adjusts the font and remembers it for the next launch. Ctrl+0 restores the system default.

<img src="../bivosia/ui/assets/bivosia-graphic.png" width="520" alt="Forward and reverse masks compared at the same frame" />

## Key-frame annotation

1. Select a frame and label animals with **Paint**, or place **SAM2/SAM3** prompts.
2. Inspect proposals; accept or reject them. Use **Paint / Erase** to correct masks.
3. **Save** both crop endpoints with consistent animal IDs.

**Save** includes accepted masks, not pending proposals. **Undo** (Ctrl+Z) reverses edits.
Optional **Clean up** creates a separate preview; its gear sets the contour-area limits
and overlap kernel. **Save cleaned** preserves the original annotations.

Key-frame parameters appear above the frame list. The detachable **Processing** panel
at the bottom left selects Local/SSH and shows prediction progress and logs.
Start predictions with the mask editor’s prediction buttons.

[Annotation controls and cleanup](REFERENCE.md#key-frame-annotation)

## Video-object segmentation

1. Select **Cutie**, **Local / SSH**, and the forward/backward directions.
2. Choose reference frames: directional start, both endpoints, or all labeled frames.
3. **Run**, follow the console, then inspect the output.

- **Internal size:** inference resolution. **Memory every:** memory-update interval.
- **Jobs:** reconnect to saved jobs. **Stop:** request cancellation.
- Full-recording mask analysis requires complete mask coverage.

[Jobs, scripts and output formats](REFERENCE.md#video-object-segmentation)

## Zone review, identity linking and export

After **Open Masks → Run**, choose **Open ZoDs**.

For automatic processing, select **ZoD detection and linking** and **Export** in Workflow.
Automatic direction choices and links are saved as a new annotation, without marking them manually reviewed.

1. **Preview** detection, then **Automatic** to generate and save directions and links.
2. In **Zone annotation**, choose **forward**, **reverse** or **invalid**. **Next →** navigates; **Approve →** marks the zone reviewed and advances.
3. In **Linking annotation**, **Next →** checks identity assignments; **Approve →** confirms them.
4. **Validate** in Linking annotation, then select **Export** in Workflow and **Run**.

**Critical only** in either annotation panel filters **Next →** and **Approve →** navigation.
It includes unresolved choices, invalid zones, incomplete chosen directions, and automatic links with warnings or invalid bridges.
Ordinary automatic results are skipped. Approving a link acknowledges its warnings; invalid zones remain available for inspection.

**View → Quality / Features** opens the optional plot panels, hidden by default. All detachable panels can be closed and reopened through **View**. The editor shares the viewer’s **File**, **Settings** and **Help** menus.

The detection and annotation panels are detachable. Click a plotted annotation to select its zone and update the fields. **Update** applies Start/End changes; the lock button controls plot editing with Alt+drag.

Changed annotations need a new export. Source masks remain unchanged.
Opening the editor or viewer loads saved links without generating proposals or changing review status.
With an annotation open, **Automatic** asks whether to create a new file or fill unresolved links while preserving zones, drafts and approvals. Progress, logs and cancellation appear in **Processing**; cancelled calculations save no partial results.

[Review and export details](REFERENCE.md#zone-review-identity-linking-and-export)

## Mask features and UMAP

- Open **Workflow → Open features** with a validated annotation.
- **Mask source: Automatic** uses a matching export if available, otherwise forward/reverse masks plus the saved annotation. You can select either source explicitly.
- Set the **Zernike radius** to contain each animal; **Run** calculates features and UMAP.
- Masks exceeding the radius are skipped individually, with frame and animal warnings in the console.
- **UMAP dimensions** selects 2D (default) or 3D. Each embedding is saved separately.
- Select **Start frame**, **End frame** and **Every Nth frame** before extraction. Defaults use all valid masks.
- **Downsample factor** defaults to 1; 2 halves width and height before preprocessing. Radius and other pixel-based settings use the downsampled resolution.
- UMAP and video sit side by side, with annotations and sample dots below. Click a point or dot to outline its mask; click a mask to highlight its sample. Selecting a frame highlights its samples.
- **Color by** switches between a shared sample gradient and animal identities.
- Zoom/pan the 2D plot, then **Show masks** to open a separate grid of representative masks. Empty cells stay blank.
- **Preprocess masks** and **Z-score features before UMAP** are optional, initially off.

[Parameters and saved results](REFERENCE.md#mask-features-and-umap)

## Keypoint alignment

- In **Dataset settings → Select keypoints…**, select a SLEAP analysis HDF5 file
  and enter its original image height and width. Save validates the file, copies
  it into the dataset's configured keypoints folder if needed, and updates `dataset.json`.
- Open **Keypoint alignment**, choose the annotation and keypoints, then run.
- SLEAP analysis files are supported. Invalid frames are skipped; ties are recorded.

Example settings:

```json
{
  "keypoints": {
    "format": "sleap",
    "file": "keypoints.h5",
    "image_shape": [1536, 2048]
  }
}
```

[Alignment options](REFERENCE.md#keypoint-alignment)

## Remote installations

1. Follow [Remote setup](REMOTE_SETUP.md) to prepare Python and copy BiVOSIA to the host.
2. **Settings → Remote installations…:** enter **Host**, **Python** and **Code**.
   Optionally set a **Default data folder** for dataset subfolders.
3. Select **SSH** in the editor’s processing panel; install models on that target if needed.

**View files → Locations…** manages mounted folders and remote dataset locations.
- **Processing:** select Local or SSH for an editor job, or for the whole automatic workflow.
- **View files:** right-click files, categories or job folders to copy, move or delete their contents.
- **Consolidate…:** copy dataset files to one location; choose which copy to use for conflicts. Job records stay where they ran.
- Run checks the selected location. Missing inputs open the dataset viewer for synchronization; the job continues after copying and any conflict choices.
- Remote masks can be viewed frame by frame without downloading full files.
- Progress and logs appear in the processing panel. Workflow availability includes remote results.
- **SAM** sends only its current image and prompts. Registration does not copy code.

## Data formats and storage

```text
dataset/
  dataset.json
  recording/         video
  masks/             forward, reverse and reviewed masks
    key_frame/       key-frame masks and annotations
  annotations/       ZoD review and identity links
  keypoints/         optional pose data
  analysis/          measurements, results and job logs
```

| Masks | Best suited to |
|---|---|
| **HDF5** | Block access; labels and validity together. |
| **NPY** | Memory mapping; optional `.valid.npy` companion. |
| **NPZ** | Compression; arrays load into memory. |

Masks use `(frames, height, width)`; **0 = background**, positive IDs = animals.
External storage roots use the same folder layout.

[Exact filenames, validity and storage settings](REFERENCE.md#data-formats-and-storage)

## Command-line examples and batch processing

Replace the path and animal count with your recording's values.

**Annotate and track**

```bash
python -m bivosia.run.run_key_frame_editor /path/to/dataset --animals 2 --model sam2
python -m bivosia.run.run_vos /path/to/dataset
```

**Analyze and review**

```bash
python -m bivosia.run.run_analysis quantify /path/to/dataset --n-animals 2
python -m bivosia.run.run_zone_revision /path/to/dataset
```

Quantification replaces existing quantification outputs; masks and annotations stay unchanged.

[Batch processing and more commands](REFERENCE.md#command-line-examples-and-batch-processing)

## Code and development

```python
from bivosia.dataset.layout import Dataset

dataset = Dataset('/path/to/dataset')
print(dataset.file('key_frames'))
```

- The outer `bivosia/` folder holds `pyproject.toml`, documentation and tests.
  The inner `bivosia/` folder is the Python package; imports remain `bivosia.xxx`.
- `config.py` collects dataset naming, storage and processing defaults.
- **Key frames** guide segmentation; **seed frames** belong to ZoD detection.
- [Code organization](REFERENCE.md#code-and-development) · [Implementation notes](IMPLEMENTATION.md)

## Authors

**BiVOSIA software:** Kara Quine, Shuyu Wang and Christoph Kirst.

Contact: [christoph.kirst@ucsf.edu](mailto:christoph.kirst@ucsf.edu)

## Paper

Shuyu Wang, Kara Quine, Audrey Jordan, Shreya Dasari, Devanand S. Manoli and Christoph Kirst.
**Identity-stable multi-animal tracking using bidirectional segmentation with object-level memory.**
*bioRxiv* (2025), preprint. [Read the paper](https://www.biorxiv.org/content/10.1101/2025.09.22.677949v1) ·
[DOI: 10.1101/2025.09.22.677949](https://doi.org/10.1101/2025.09.22.677949)

## License

[GPL-3.0-only](../LICENSE). Third-party software and model weights retain their own licenses.
