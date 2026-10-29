<img src="../bivosia/ui/assets/bivosia-icon.svg" width="64" alt="BiVOSIA icon" />

# BiVOSIA guide

**Video → Key frames → VOS → Mask analysis → Review & link → Export**

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
| **1 · Video** | Select the experimental recording. |
| **2 · Key frames** | Label the animals at both crop endpoints. |
| **3 · VOS** | Track masks forward and backward with Cutie. |
| **4 · Mask analysis** | Calculate agreement between the two directions. |
| **5 · Review & link** | Review disagreements and verify animal identities. |
| **6 · Export** | Validate and save reviewed masks for further analysis. |

**Green** = outputs available · **Amber** = work missing · **Purple** = review needed.

- **View files:** inspect files, arrays and video; preview text or video by clicking.
- **Console:** follow job logs and progress, or request cancellation.
- **Recent datasets:** reopen a dataset. Click a window entry to bring it forward.

<img src="../bivosia/ui/assets/bivosia-graphic.png" width="520" alt="Forward and reverse masks compared at the same frame" />

## Key-frame annotation

1. Select a frame and label animals with **Paint**, or place **SAM2/SAM3** prompts.
2. Inspect proposals; accept or reject them. Use **Paint / Erase** to correct masks.
3. **Save** both crop endpoints with consistent animal IDs.

**Save** includes accepted masks, not pending proposals. Optional **Clean up**
creates a separate preview; **Save cleaned** preserves the original annotations.

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

After **Mask analysis → Run**, open **ZoD annotation and linking → Open editor**.

1. **Preview** detection, then **Start** or resume a saved annotation.
2. Review each Zone of Disagreement (ZoD): choose **forward**, **reverse** or **invalid**.
3. **Next link →** checks identity assignments; **Approve →** confirms them.
4. **Validate → Export…** saves the reviewed masks. **Result** opens the export.

Changed annotations need a new export. Source masks remain unchanged.

[Review and export details](REFERENCE.md#zone-review-identity-linking-and-export)

## Mask features and UMAP

- Open **Workflow → Mask features** with a reviewed consensus export.
- Set the **Zernike radius** to contain each animal; **Run** calculates features and UMAP.
- **Preprocess masks** and **Z-score features before UMAP** are optional, initially off.

[Parameters and saved results](REFERENCE.md#mask-features-and-umap)

## Keypoint alignment

- Add keypoints and their original image resolution to `dataset.json`.
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
3. Select **Run on → SSH** in the editor; install models on that target if needed.

**View files → Locations…** manages mounted folders and remote dataset locations.
**SAM** sends the current image and prompts. **VOS → Run** uploads the video and saved
key frames first; identical files are reused and conflicting files are never overwritten.
**Check** reports required uploads without transferring them. Registration does not copy code.

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
