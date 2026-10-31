# BiVOSIA implementation notes

Status: initial refactor implemented, 2026-10-04. Algorithms, recording-scale
processing and dataset writers are separated. Viewer/editor imports and archived
callers use the new modules directly; superseded modules have been removed.
The key-frame editor and first VOS interface were added on 2026-10-05.
VOS supports local and SSH background jobs, script export and a progress console.
Real Cutie inference awaits its checkpoint; external-tool correction remains
future work. Automatic direction proposals support the Methods rule and two
simpler missing-mask rules.

## Responsibilities

`analysis` contains the result-producing algorithms. Functions accept masks,
arrays or annotation records and return results. They do not open datasets,
create worker pools, parse command-line arguments or configure logging.

`processing` runs these algorithms on recordings: load, process in blocks,
report progress, handle cancellation and save. `dataset` owns paths, readers,
writers and annotation persistence. The CLI and GUI call the same processing
functions. Colors, display ranges, downsampling and interaction stay in `ui`.

Implemented layout:

```text
bivosia/
  analysis/
    masks.py          validation, relabeling, cleanup and preprocessing
    mask_features.py  geometry, contour and Zernike descriptors, temporal features
                      and frame summaries
    mask_quality.py   IoU, assignments, margins, switches and direction comparisons
    zones_of_disagreement.py evidence, seeds, expansion, merging,
                             zone summaries, interval edits and mask directions
    mask_linking.py   boundary matches, identity propagation and validation
  processing/
    analysis/
      alignment.py    keypoint-to-mask alignment over a recording
      keypoints.py    keypoint displacement over a recording
      mask_quality.py basic mask quantification over a recording
      mask_export.py validated mask output generation
      mask_features.py detailed feature output generation
      mask_shape.py  sampled shape descriptors and embeddings
      zones_of_disagreement.py detection, direction choices and automatic linking
    segmentation/
      keyframes.py   prompted key-frame prediction and its remote service
      vos.py         bidirectional video-mask propagation
      model_install.py segmentation packages and model weights
    storage/
      file_operations.py individual file transfers
      folder_operations.py folder transfers and cleanup
      consolidation.py dataset consolidation and required-input synchronization
    remote/
      client.py       SSH requests and installation checks
      server.py       request dispatch to the appropriate subsystem
      session.py      persistent connections, timeouts and cleanup
    jobs.py           input checks, job submission, execution and monitoring
    workflow.py       workflow dependencies and step execution
    parallel.py       shared block-processing utilities
    log.py            logging setup and progress reporting
  dataset/            dataset layout, recordings and data-specific readers
    recording.py      loaded recording and ZoD/linking access
    io/               interfaces, format registry, adapters and array transport
    annotations/      key-frame and mask annotation state and persistence
      keyframes.py    key-frame masks, prompts and editing state
      masks.py        mask selection, validity, review, identity links and inputs
    storage/          locations, inventory, file operations and remote readers
    workflow/         processing settings, input validation, job records and status
  segmentation/       image/video interfaces and SAM/Cutie backends
  ui/                 viewers, editor and future console
  run/                thin entry points
  config.py           dataset hierarchy, file names, formats and shared defaults
```

`analysis/zones_of_disagreement.py` owns zone detection, shared missing-mask
event classification, zone summaries and revision interval rules. UI adapters
only turn those results into display records. `dataset/recording.py` connects
the analysis to a loaded recording. Identity linking remains in
`analysis/mask_linking.py`, including independent frame transitions inside a zone.

The default detector combines missing-mask, assignment-switch and low-quality
seeds, expands to recovery, then merges intervals using an inclusive frame gap.
Missing masks and switches remain explicit flags; the displayed quality is the
minimum finite frame quality, without replacing it by negative infinity.
The `direction_method` parameter selects `missing_masks`, `missing_mask_counts`
or `temporal_quality` (default), as listed in `config.AUTO_DIRECTION_METHODS`.
The first preserves the conservative complete-direction rule. The latter two
reject a zone if either direction is missing more than one mask in any frame,
then choose fewer missing masks; equal nonzero totals are invalid. Without
missing masks, the count-only rule chooses forward and the default Methods rule
chooses the higher minimum temporal assignment quality; equal scores or no
finite quality are invalid. The editor saves the method in the annotation
header. Loaded zone records retain their saved choices; automatic interval
edits use the session's method and manually chosen directions are preserved.

Pixel-label replacement lives in `analysis/masks.py`, identity validation and
preferred-mask selection in `analysis/mask_linking.py`, and output streaming in
`processing/analysis/mask_export.py` using dataset writers. `Linker` operates on a supplied
mask accessor and evidence arrays; `RevisionRecording` in `dataset/recording.py` connects it to the
loaded recording.

## Point and mask alignment

Keypoint displacement is calculated independently of mask validity and assignments,
using the alignment's selected nodes in original source-image pixels.
`analysis/keypoints.py`
averages Euclidean displacement over nodes with finite xy in both consecutive
frames for each track; no common nodes gives positive infinity. The frame-pair
score is the maximum across tracks. `processing/analysis/keypoints.py` computes blocks
with one overlapping frame so no boundary transition is lost.
`analysis/keypoint_displacement.h5` stores `individual[t, track]` and `maximal[t]`
for the transition t to t+1, plus units, source image shape, node/track names
and `excluded_keypoints`. The loader verifies that the exclusions match the
alignment. The viewer's All keypoints display toggle does not change this metric.
No transition beyond the final supplied frame is used.
Skeleton detection and finite-point counts use the same selected nodes as
alignment and displacement. A track with finite coordinates only at excluded
nodes is missing. The viewer and exported frame annotations
share this selection through `KeypointInputs.selected_node_ids`. Raw per-node evidence
and the optional All keypoints overlay retain the source nodes.
The standard and comparison viewers load these results into a synchronized
Keypoint displacement dock below Zones, visible by default. The shared graph
shows finite displacement and marks positive infinity at the top with ∞;
the final frame has no following-frame value. Its View-menu entry and docking
behavior reuse the existing panels. No mask or keypoint swap results change.

`config.py` separates `DATASET_LAYOUT` (folders and file names), `DATA_FORMATS`
(data-category readers, writers and suffixes), `KEY_FRAME_BACKENDS` (SAM2/SAM3)
and `VOS_BACKENDS` (Cutie). `ANALYSIS_DEFAULTS['mask_quality']['count_dtype']`
sets the pixel-count dtype shared by overlap calculation and result loading.
`ANALYSIS_DEFAULTS['zones_of_disagreement']['theta_quality']` supplies the `0.60` seed default to the detector,
processing, linking and viewers; explicit thresholds and saved annotation settings
retain their supplied values.
Both defaults dictionaries use module names as their top-level keys.
`ANALYSIS_DEFAULTS` groups detection thresholds in `zones_of_disagreement`, assignment
fallback, tie tolerances and dilation in `alignment`, counts and temporal intersections
in `mask_quality`, mask-linking tie tolerance in `mask_linking`, and optional Hu,
convex-hull and FFT features in `mask_features`.
Zernike settings live in `mask_features['zernike']`; optional z-scoring belongs to `embedding`.
`analysis/masks.py` contains validation and mask manipulation. Cleanup and preprocessing defaults live under
`ANALYSIS_DEFAULTS['masks']['cleanup']` and `ANALYSIS_DEFAULTS['masks']['preprocessing']`.
`analysis/mask_features.py` groups geometry and presence, contour and temporal features,
Zernike descriptors, frame summaries and feature validation. `analysis/mask_quality.py`
groups overlaps, assignment quality, temporal comparisons and recording comparisons.
`analysis/mask_linking.py` uses these comparisons to establish identities across segments.
`include_hu_features=False` skips both signed-log Hu values and Hu-based temporal
shape comparisons; `max_shape_change_i1` is omitted when its source is absent.
Raw and derived output column names record the selected features.
`include_convex_hull=True` and `include_fft_features=False` independently control
convex-hull excess and spectral calculations, including their derived measures.
Spectral sampling and frequency parameters live under `mask_features['fft_features']`
regardless of whether calculation is enabled.
`PROCESSING_DEFAULTS['analysis']` groups batch/block sizes and worker counts under
`mask_quality`, `mask_features`, `mask_shape`, `alignment` and `keypoints`.
`PROCESSING_DEFAULTS['jobs']` holds job timing intervals.
`mask_shape` also holds frame-sampling defaults. Saved workflow operation identifiers
such as `quantify`, `detect_link` and `shape_features` identify jobs independently
of their implementing modules.
`PROCESSING_DEFAULTS['remote']['session']['timeout_seconds']` sets the shared
request timeout for remote mask readers and image segmentation.
`None` workers means automatic selection;
alignment defaults to one worker. CLI switches override only the options explicitly
supplied, so omitted options retain the configured defaults. `JOB_TERMINAL_STATES`
remains a fixed job-state definition.
Processing passes the selected dtype to its workers and checks pixel-count capacity
before writing. `dataset/io/registry.py` resolves an explicit format or a
unique suffix within the requested category. Read-only formats have no writer
entry. Masks support HDF5, NPY and NPZ; points use SLEAP, videos use OpenCV,
analysis arrays use NPY, documents use JSON and structured result containers use
HDF5. Container access does not replace the result-specific schemas and validators.
Mask adapters stage their output; the shared mask transaction owns locks,
exclusive destinations and rollback. Video buffering consumes the video reader
interface, preserving RGB processing and BGR display conventions.

Model definitions contain their own checkpoint and parameter defaults. Cutie's
entry also owns its encoder weight names, reference modes, controls and help text.
SAM2's configuration is a SAM2 parameter. `segmentation/models.py` handles lazy
construction and availability checks; `ImageSegmenter` and `VideoSegmenter`
define distinct inference interfaces. The SAM2/SAM3 specializations share prompt
inference and image-embedding reuse. The editors consume registry definitions and
model metadata without branching on backend names.

`dataset/io/interfaces.py` defines `KeypointReader`, `MaskReader`, `VideoReader` and
`FileMetadata`. Points and masks expose frame-indexed data with dimensions and
explicit validity; video readers expose decoded images and frame rate.
Reader implementations use a source prefix, such as `Hdf5MaskReader` and
`SleapKeypointReader`. Editable annotation state uses `Session` or `Draft`;
`dataset/annotations/masks.py` groups `RevisionSession`, saved annotation loading
and `AnnotationInputs`, which owns the analysis inputs needed for annotation and
export. It manages mask-selection and identity-link annotations;
`dataset/masks.py` reads and writes the mask arrays themselves.
`KeypointInputs` combines a coordinate reader with optional alignment and
displacement results. The reader itself contains source data and coordinate
metadata, not analysis results or node-selection settings.

`AlignmentResult`, `MaskMatchResult`, `LinkResult` and `EmbeddingResult` stay
with their algorithms in `analysis`. `ZodEvidence` and `ZodResult` hold detection
evidence and output there.
`ZodAnnotation` holds one editable ZoD interval, its mask choice and review status;
`ZodView` is its display representation in `ui`. Recording-wide partitions use
`Segment`, including intervals outside ZoDs. `MaskShapeResults` in
`dataset/mask_features.py` locates and validates saved mask-shape descriptors and
embeddings; it does not hold the numerical result arrays.
`MaskShapeDescriptors` contains loaded descriptor rows, frame/identity indices
and provenance. `MaskShapeEmbedding` combines an `EmbeddingResult` with those
sample indices and provenance. The same numerical embedding type is returned
by calculation and file loading.
`RevisionRecording` specializes `Recording` with ZoD detection and linking access;
both live in `dataset/recording.py` and own the loaded recording resources.
`RecordingStreams` groups video and mask readers for playback.
`SleapKeypointReader` converts SLEAP's
track-major axes on read. `Hdf5MaskReader`, `NpyMaskReader` and `NpzMaskReader` specialize
`MaskReader` for HDF5, memory-mapped NPY and compressed NPZ files. `NumpyMaskReader`
adapts supplied arrays. `open_masks` selects the file reader for analysis and
video loading. `dataset/masks.py` also selects exported or annotation-based masks
and writes mask outputs. `dataset/io/numpy.py` and `dataset/io/hdf5.py` each
contain generic format operations and mask adapters. NPY stores partial frame
validity in an optional `.valid.npy` sidecar; HDF5 and NPZ keep
validity with the labels. Historical hand annotations are converted by
`_previous/run/import_hand_annotations.py`, with source-version information
retained as provenance rather than an importer name.

Optional `keypoints/keypoints.h5` retains the SLEAP analysis file in its native
format. `dataset.json` annotates its source `keypoints.image_shape` and optional
`invalid_frames`. The reader exposes frame-major (frame, track, node, xy)
coordinates, names and frame validity through `KeypointReader`; a single point uses
one node. Requested slices are limited to the dataset's mask frame count.
Readers do not retain an open point-file handle. The annotated point
`image_shape` and mask array dimensions describe their coordinate systems.
The shared point reader derives the scale
between them (0.25 for 1536×2048 points and 384×512 masks), applies it on read,
and exposes the target `image_shape` alongside `source_image_shape`. Alignment
itself does not convert coordinate systems. Original coordinates are preserved.
`analysis/keypoint_alignment.h5` stores zero-based
assignments (-1 unmatched), validity, tie flags and selected mask source codes,
along with overlap scores and mask-presence flags for scored frames.
Its annotation fingerprint prevents displaying results against changed labels.
The file also retains boolean `overlaps[frame, track, node, mask]` for every
node, including excluded nodes, against each independently dilated selected
mask. `point_usable[frame, track, node]` records finite coordinates that round
inside the image on valid alignment frames. Invalid frames are not evaluated;
their false entries must be interpreted together with `valid`. Node/track names,
image dimensions, coordinate rounding, dilation and selected mask source are
stored with the result. Scores can be recalculated from these arrays using
`analysis.alignment.point_overlap_scores` after choosing nodes, without
reading the masks again. Changing dilation or coordinate rounding requires
recomputing containment from the source masks and points.

`analysis/alignment.py` rounds supplied mask-pixel coordinates to the nearest integer
(half-integers to even), then scores independently dilated masks by C²/K:
C contained points out of K finite in-bounds points. It maximizes the total C²/K score;
zero-overlap pairs remain unmatched.
Equal optima leave ambiguous pairs unmatched. Optional previous-assignment
fallback retains compatible pairs, propagates initial ties backward from the
first non-tied valid frame, and preserves every tie flag.

`processing/analysis/alignment.py` selects the annotation's mask direction, combines
mask and point validity, and runs blocks with shared progress/cancellation.
The `align` CLI command uses the selected annotation; `--fallback-previous`
enables temporal fallback, which requires consistent mask identities.
`--exclude-keypoints NAME [NAME ...]` (the processing function's
`exclude_keypoints` argument) omits named nodes from both C and K during
scoring. Unknown names are rejected. The exclusion list is stored as JSON in
the alignment file's `excluded_keypoints` attribute and shared by skeleton
detection and displacement. Source coordinates retain all nodes; the viewer
shows selected nodes by default, with an optional All keypoints overlay.
The existing standard viewer and `MaskDisplayControl` expose a keypoint switch
and Original label / Assigned label / Both colors when keypoints are present.
Both draws the original inner disk and assigned outer disk when available.
Assigned colors appear in the annotation-selected mask pane.
The shared keypoint controls hide nodes listed in the saved alignment's
`excluded_keypoints` by default. The All keypoints checkbox includes those
nodes in the overlay. Without an alignment there are no saved exclusions.
This display choice applies to all color modes and does not change analysis.
Saved assignments use original keypoint track slots and local mask labels.
`analysis/keypoints.py` applies the existing mask identity links to each keypoint
track's assigned mask. Consecutive resolved frames are compared across
unresolved frames. Resolved means valid, untied and assigned for every mask
animal. A changed track must be assigned in both endpoints, which must belong
to the same connected component. Appearing or disappearing tracks, local mask
relabeling and F/R source changes alone do not mark normal keypoint swaps.
Each swap zone includes both resolved endpoints p through q and counts once.
Touching zones remain separate events. A zone spanning dataset-invalid frames
is shown as Swap over invalid zone when both endpoints have linked identities.
Leading/trailing unresolved frames have no pair of resolved anchors; unchanged
endpoint assignments do not create a zone. Unlinked components are not compared.
The event output retains zone intervals, one swap marker at each later endpoint,
per-zone invalid-span flags and valid-frame coverage for error-frame counts.
Partial resolved swaps compare successive unambiguous keypoint track assignments
for each linked mask, across unresolved or invalid gaps within the same linked
component. At least one endpoint must be unresolved as a whole frame. Tied
frames contribute only pairs shared by all optimal assignments, excluding
temporal fallback. Several masks changing over the same endpoint pair count
as one event; different endpoint pairs remain separate, including when they
share a later endpoint. Full and partial swap zones can overlap and are counted
separately. Both include their endpoints, and frame-error totals use the union
of their valid-frame coverage. Saved point-to-mask assignments are unchanged.
The Keypoints row shows swaps, partial resolved swaps, no overlap, combined
ties/unassigned and missing skeleton bands in that order,
using the same `IntervalBand` model as hand annotations. The combined band has
distinct colors for ties only, unassigned only and both. No overlap means a
detected skeleton has zero overlap score for every mask. Unassigned means a
detected skeleton has positive overlap but no assigned mask. Ties retain
the matcher's saved flags even with temporal fallback. Missing skeleton marks
fewer detected skeletons than expected animals on valid alignment frames. A
skeleton is detected when it has at least one finite xy pair among its
analysis-selected nodes, independently of mask overlap. The existing
`processing/analysis/keypoints.py` block reader supplies counts to the analysis; the
viewer calculates them once on opening. This display flag does not change
alignment. Both timelines show the same five subrows. Partial resolved swaps
are dark blue, directly below the normal blue swaps. Invalid crossings are
light purple for normal swaps and the existing purple for partial swaps.
Both swap subrows span the centers of their endpoint frames, inset by 0.05
frames at each end to separate successive events. Longer zones stay continuous
through their interior. Dark-blue vertical borders follow the inset edges.
The shared interval-band renderer draws borders after the fills in the detail
plot and overview; recorded inclusive intervals and event counts are unchanged.

`_previous/run/compare_keypoint_alignment.py` runs the archived two-vole scoring
and assignment functions verbatim on the same annotation-selected masks and
linked identities. It retains the original floor conversion and pair decisions,
stores their raw outputs and native counters separately, and uses the shared
event definitions for a comparison table. Only the evaluation copy removes
forced zero-overlap pairs, so these cannot count as verified identities. Original
swap counters use their original gap labels; the shared comparison uses
resolved endpoint pairs within one linked component. Historical execution and
comparison adapters stay in `_previous`; the active algorithms are unchanged.
The comparison runner also saves the original per-node containment array,
including its NaNs, in `original_point_overlaps.npz`, with selected sources and
local-to-linked mask mappings. Completed native assignments and logs are reused
unchanged. Nonfinite original scores are retained and prevent only the optional
shared-definition evaluation; native classical counts remain available.
`_previous/run/run_keypoint_comparisons.py` runs the recordings listed in the
hand-annotation summary, using explicitly selected mask frame ranges and native SLEAP
files and the existing hand importer. It resumes completed dataset stages,
records individual failures and writes the per-recording counts to `batch.json`.
Total swaps count normal swaps plus partial events whose transition spans do
not overlap a normal swap; sharing only an endpoint does not merge events.

`_previous/run/run_keypoint_comparison_viewer.py` opens a separate comparison
viewer with a Classical row at the bottom of both zone plots. Its historical
reader loads the original log's Localized Swaps, Post-Gap Swaps and four
remainder categories verbatim, including the original nontrivial classification
of undefined-score frames. It does not recompute or reclassify those events.
The standard viewer accepts optional additional `RowSpec` objects; its normal
rows and keypoint overlays are unchanged. The comparison uses the existing
interval bands, legend checkboxes and synchronized timelines.

Auto, Fail and Seed share one ZoD row: Auto fills the full row height,
missing-mask and Switch colors occupy the middle third, and Seed occupies
the bottom third, using `IntervalBand`. Overview and detail show the same
tracks; the editor uses the same configuration.
The standard viewer initially allocates 20% of the timeline split to the
overview and 80% to the detail plot, reducing the overview's previous share by 60%.
Switch frames mark changes in the forward-to-reverse assignment between
consecutive valid frames. Both incident frames remain marked, as in the
existing calculation. These are distinct from hand-annotated swaps within
forward or reverse masks. The detector uses `switch_frames` and `switch_seeds`;
the persisted signal is `mask_switch.npy`.
Quality and feature plots reuse `GraphPanel` for their legends and graphs.
In the standard viewer they occupy separate `DockPanel` windows, start hidden,
and are shown through the View menu. All timeline ranges and playheads remain
synchronized while docked, detached or hidden.
Feature controls are supplied as its legend, retaining their current behavior.
Zone list, Video, Zones, Quality and Features use the editor's shared
`PanelHost` and `DockPanel` for detaching, resizing and redocking, with visibility
actions in View. The shared dock style renders titles in bold.
The zone list and its existing ZoD controls occupy the Zone list dock; Video contains
only the video panes and their playback/display controls. The zone list starts
beside Video, with the zone timelines below both in the Zones dock. On opening,
the Zones dock height is reduced to 60% of the initial layout, subject to its
content's minimum height.
Zone selection stays synchronized when detached.
The shared `FramePlayer` starts playback at the current cursor. A checked loop
applies only when playback starts inside its inclusive bounds. Starting outside
continues without looping through the recording's end, even when passing through
the loop interval. Moving the cursor outside during playback also disables
looping for that run. Loop brackets and the checkbox are retained; pressing Play
again reevaluates the cursor position. Editor prefetch uses the same effective
playback bounds and loop state.
File → Open... uses the same dataset loader as the command-line
viewer and replaces the current window only after a successful load. Canceling
or selecting an unreadable dataset keeps the current recording open.
All GUI launchers set the application icon from `ui/assets/bivosia-icon.svg`;
windows inherit it. The main dataset tree paints `ui/assets/bivosia-graphic.png`
at low opacity beneath its rows, preserving the graphic's aspect ratio.
Help → About BiVOSIA shows project information, the full-color application graphic,
and a bundled 30-second masked six-vole preview below it. The content scrolls
when it exceeds the available screen height. `ui/assets/about.gif` is a
480×360, 15 fps loop taken from
frames 54000–58499 (18:00–19:30 at 50 fps), played at 3× speed, from the existing
`six_vole_linked_r4571/six_vole_linked_pastel.mp4` render. The saved render plan
and exported masks mark all 4500 source frames valid, with all six identities
present. Qt's `QMovie` plays the preview only while the information window is open.
The standard viewer's Colors selector switches between local Labels and the
editor's Global link preview. Both viewers share `global_label_colors` and
component palettes. Approved links and unambiguous automatic boundary matches
connect identities; unresolved boundaries keep separate palettes. The same
raw-mask color lookup colors assigned keypoints, while original keypoint disks
retain their track colors. This display option does not modify alignment,
approve links, or change the keypoint identity-event tracks, which always use links.
Linked zones can be shown in both timelines with the existing editor component
row. The shared row assigns a distinct shade from a blue gradient to each
connected component. Golden-ratio spacing distributes the shades across
component order so nearby components remain visually separated. Repeated
intervals of one component retain the same shade in both timelines. Invalid
intervals stay empty, and component labels are retained. Mask and keypoint
identity palettes remain separate from this linked-zone shading.
The standard viewer uses the shared `ZonePanel`'s checkable legend: each available
row has a checkbox before its title and color keys (Hand, ZoD, Keypoints, Links).
The controls replace the separate Linked zones checkbox and affect both the
overview and detail timelines. Hidden rows retain their legend controls, and
visibility choices survive row-data refreshes. Links starts hidden as before.
Automatic boundary markers use the shared Auto color, without text or quality
overlays on the timeline.

Reviewed zones retain optional `forward_valid`, `reverse_valid`, `forward_swap`
and `reverse_swap` hand annotations in the current session format. Swap values
are event counts; an omitted or null swap annotation means zero swaps.
The shared hand-row renderer uses four stacked strips: forward validity, reverse
validity, preferred direction and combined directional swaps. It preserves the existing interval selection,
overview and legend widgets. Geometry edits clear validity judgments and swap
annotations for the new interval; Undo and imported source records preserve
the original annotations.
The Hand legend groups color-first keys as valid (F, R, invalid), preferred
(F, R, none), and swap (F, R, B), separated by vertical bars. Validity retains
its separate F/R strips and green direction colors, with purple for invalid.
Swaps use light blue for F, darker blue for R and darkest blue for both in one
strip, consistently with the legend. Preferred direction keeps its existing
colors. Missing validity is labeled none; its legend key is omitted when every
displayed zone has both validity annotations. Preferred none means no chosen
mask direction. The swap strip is marked only for nonzero event counts.
Fail uses mask keys F, R, B followed by switch and, when present, mask+switch.
Keypoints likewise places color keys before ties and changes. Auto zones use
one shared orange color in the viewer and editor, distinct from blue Switch frames.

## Linking versus relabeling

Linking determines which mask identities correspond across frames or regions.
For a left-to-right raw-label permutation `h` and a left raw-to-global mapping
`g_left`, propagation produces:

```text
g_right[h(i)] = g_left[i]
```

Relabeling applies the resulting mapping to a mask image. With zero-based
mapping indices and mask pixels using 0 for background and 1..N for objects:

```text
output[p] = 0                         if input[p] == 0
output[p] = g[input[p] - 1] + 1        otherwise
```

Relabeling changes IDs, not segmentation geometry. Preferred F/R selection,
invalid-frame handling and identity-consistency rules remain analysis
functions; the processing layer applies their results and writes the outputs.

`prepare_links` combines approved boundary mappings with automatic matches of
the selected endpoint masks. An automatic match must be complete, have positive
IoU for every selected pair, and have a unique optimal assignment. Invalid gaps
and ambiguous/incomplete matches remain separate until reviewed. Manual or
moved-anchor drafts also remain pending. Interior detector failures do not
prevent matching otherwise valid boundary endpoints; chosen-mask completeness
is checked separately before export.

For endpoint IoUs `I` and selected permutation `h`, automatic-link quality is
the existing signed margin calculation:

```text
q = min_i min(I[i,h(i)] - max_{j != h(i)} I[i,j],
              I[i,h(i)] - max_{k != i} I[k,h(i)])
```

The editor's boundary status reports quality alongside Auto without adding a threshold. Viewers,
validation and export use the same boundary results and identity propagation.
Automatic results do not change saved hand annotations or approval records.

## Pipeline

| Step | Responsibility | Scope |
|---|---|---|
| 1. Mask generation | Key-frame editing and video propagation | `segmentation` backends, executed through `processing/segmentation/vos.py` |
| 2. Mask quantification | Integer intersections and per-direction areas in one mask pass; derive IoU, Hungarian assignments, quality and switches from those counts | `analysis/mask_features.py` and `analysis/mask_quality.py`, executed through `processing/analysis/mask_quality.py` |
| 3. Detection and automatic direction | Detect zones, summarize flags and choose mask direction using the selected method | `analysis/zones_of_disagreement.py` |
| 4. Zone editing | Revise intervals, select masks and approve links | Existing UI calls analysis algorithms; session state/history stays in `dataset` |
| 5. Validated outputs | Validate links and generate relabeled masks and optional detailed features | Shared export/feature processing functions |
| 6. Correct external outputs | SLEAP or other tool outputs | Later; no implementation now |

Core feature functions can operate on raw or validated masks without duplicate
implementations. Detailed features are not a prerequisite for basic detection.

The first pass saves exact counts in `mask_intersections.npy`, with shape
`(frames, animals, animals)`, and `mask_areas.npy`, a uint32 array with axes
`(direction, frame, raw object ID)`, direction order forward/reverse and IDs 1..N.
Directional missing counts come from zero areas, including frames whose entire
IoU matrix is NaN. Areas include pixels overlapping the other pass's background.
Detailed component/contour features remain a separate pass.

With `temporal_intersections=True` (CLI: `--temporal-intersections`), the same pass writes
`mask_temporal_intersections.npy` with shape
`(2, frames-1, animals, animals)`. All three count files use uint32. Entry `[d, t, i, j]`
is the intersection between raw ID `i+1` at `t` and ID `j+1` at `t+1` within direction
`d` (forward/reverse). Both directions use chronological frame order.
Each processing block reads one preceding frame to include its first transition.
The compact `mask_temporal_quality.npy` array, shape `(2, frames-1)`, is always
written: temporal IoU matrices use the same Hungarian assignments and signed
row/column margins as the forward/reverse quality calculation. Entry `[d,t]`
compares frames `t` and `t+1`; zone quality takes the minimum finite value for
transitions starting inside the zone. The last recording frame has no successor.
The `temporal_intersections` option defaults to False and controls only retention
of the full temporal count matrices. Zone detection itself is unchanged.
The loader retains integer missing counts and loads temporal quality when
available. Selecting a temporal proposal without this output requests a new
mask-analysis run instead of substituting the conservative rule.

Unions and normalized IoU are derived from the count arrays, not persisted:
`union = A + B - I`, `IoU = I / union`. Counts are promoted to int64 before
arithmetic. Frames exceeding uint32 pixel capacity are rejected before processing.
Either area being zero produces NaN IoU.
Assignments and quality use the same formulas as before and run over count-array
blocks without reading masks again. `dataset.mask_quality.MaskOverlapReader` owns the
memory maps and calls the shared conversion in `analysis/mask_quality.py`.
Processing and GUI entry points share a `Dataset` instance; `Recording(dataset)`
adds loaded evidence and video resources. Historical normalized-array readers
and processing entry points remain under `_previous`.

## Extracting the existing code

| Current code | Destination |
|---|---|
| `calculate_iou_matrix.frame_iou_matrix` | `analysis/mask_quality.py` |
| Assignments, missing-mask measures, margins and switch calculations in `detection.py` | Comparison calculations in `analysis/mask_quality.py`; zone detection in `analysis/zones_of_disagreement.py` |
| Block loops and file processing in the IoU, match and quality calculators | `processing/analysis/mask_quality.py`, using dataset I/O |
| `zone_revision.py` | `analysis/zones_of_disagreement.py` |
| `zone_linking.py` and export identity validation | `analysis/mask_linking.py` |
| `global_relabel.py` | Pixel mapping in `analysis/masks.py`; identity/source rules in `analysis/mask_linking.py`; streaming in `processing/analysis/mask_export.py` |
| `revision_data.py` | Numerical linking rules in analysis; loaded-recording orchestration/caches in `dataset/recording.py` |
| Feature and measure calculators | Geometry, descriptors and frame summaries in `analysis/mask_features.py`; forward/reverse comparisons in `analysis/mask_quality.py`; recording-scale execution in `processing/analysis/mask_features.py` and `processing/analysis/mask_shape.py` |
| `parallel.py` | Worker utilities in `processing/parallel.py`; progress reporting in `processing/log.py` |
| Embedded `main()` functions | Thin entry points in `run` where needed |

Preserve formulas and output values during extraction. Temporal calculations
must include adjacent frames across block boundaries, including permutation
changes at the first frame of a block. Do not duplicate formulas in processing,
loaders or viewers.

## Logging

Use standard Python logging. `processing/log.py` owns shared setup and the
progress reporter; a generic `utils` package is unnecessary at this stage.
The CLI or GUI configures handlers once. Modules use named loggers, and
processing calls attach context through `extra` fields. This separates records
from their text formatting. [Python logging reference](https://docs.python.org/3.10/library/logging.html)

Processing conventions:

- `INFO`: stage start, rate-limited progress and successful completion.
- `WARNING`: recoverable issues requiring attention.
- `DEBUG`: optional execution details.
- Raise failures to the execution boundary, which logs the exception once.
- Progress records contain `stage`, `completed` and `total`; completion records
  may contain a dataset-relative `result_path`.
- Report progress approximately once per second and at completion. Count
  completed blocks, not the largest returned frame index.
- Logging is informational: return values and exceptions determine success.
  Cancellation must not emit a successful-completion record.

The parent process reports completed work. If workers also emit logs, use
`QueueHandler`/`QueueListener` to collect them rather than letting processes
write to one log handler independently. Keep record fields serializable.
[Python logging handlers](https://docs.python.org/3.10/library/logging.handlers.html#queuehandler)

A future GUI handler forwards records through a Qt signal; the receiving slot
updates the console on the GUI thread. No Qt objects enter analysis or
processing functions. [Python Qt logging example](https://docs.python.org/3.10/howto/logging-cookbook.html#a-qt-gui-for-logging)

## Interactive logs: future option, not an implementation task yet

Structured records are the useful foundation. A terminal can show text while
the GUI renders the same record with an **Open folder** or **Open result** link.
Processing code supplies a result reference, not HTML or a shell command.

Qt's `QTextBrowser` supports rich text and link-click signals. An eventual
console can handle clicks itself, resolving a dataset-relative result against
the active dataset. `QDesktopServices.openUrl()` can delegate a local folder or
file to the desktop. An internal result link could instead call an existing
viewer. These actions need UI handlers; HTML formatting alone does not connect
them to the application. [QTextBrowser](https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QTextBrowser.html),
[QDesktopServices](https://doc.qt.io/qtforpython-6/PySide6/QtGui/QDesktopServices.html)

Escape message text before HTML rendering. Disable automatic link navigation
and accept only known, user-clicked actions; never execute arbitrary commands
from log content. Persist result paths relative to their dataset, not as
machine-specific absolute links.

Rich offers a standard logging handler for styled terminal output and can
export recorded console output as HTML. This is useful for optional saved
reports, but does not provide BiVOSIA-specific actions automatically. Do not
add it as a dependency for the initial logging refactor.
[Rich logging](https://rich.readthedocs.io/en/stable/logging.html),
[Rich HTML export](https://rich.readthedocs.io/en/stable/console.html#exporting)

## Implementation order and checks

1. Extract array/mask algorithms and rename revision/linking modules. Update
   imports directly; no old-path compatibility wrappers.
2. Separate recording-scale processing from dataset I/O and core calculations.
3. Replace processing prints with the shared logging/progress reporter. Keep
   existing progress controls working through an adapter to the same counts.
4. Update existing entry points and viewers to use the shared processing code.
5. Check numerical equivalence, signed margins, missing masks, temporal block
   boundaries, linking/relabeling, cancellation and final progress reporting.
   Check imports with no Qt dependency in analysis/processing and open the
   existing viewers without saving annotation changes.

The initial refactor was checked with 25 doctest examples, one small synthetic
parallel-processing run, two existing export/link-preview checks, and off-screen
loads of the six-vole editor and standard viewer. No test suite was added.
The dataset description and saved hand annotation checksums were unchanged.

`run/run_analysis.py` provides `quantify` and `features` commands. Existing
viewer/editor launchers configure the same logging setup. Progress records
carry stage and completed/total counts; dataset-relative result references
are supplied when the caller knows the dataset root.

Retain Python 3.10, concise module examples and the agreed module headers.
Preserve datasets and original annotations. Keep historical formats, converters,
old IoU/alternative-IoU support and comparison GUIs exclusively in `_previous`.
Keep changes focused on the requested workflow; external-tool correction and
a pipeline framework are outside the current scope.

## Key-frame editor

`run/run_key_frame_editor.py` opens video-only datasets. `ui/key_frame_editor.py` owns
frame selection and previews; `ui/mask_editor.py` edits one frame in a reusable
window. Both use the existing video renderer, navigation, opacity, timeline,
legend and splitters. The existing player and identity palettes now live in
`ui/player.py` and `ui/colors.py`, shared with the revision viewer.
`ui/mask_canvas.py` handles image interaction and brush/selection overlays;
`ui/mask_tools.py` supplies the exclusive mode buttons.

`analysis/keyframes.py` owns frame proposals, crop/label validation, prompt conversion
and mask editing. Initial proposals are evenly spaced and include both crop
endpoints. When labels already exist, proposals are spaced between fixed labels
and endpoints. Only unlabeled interior frames move. Changing endpoints changes
the processing crop, not the source video or existing labeled frame numbers.
Cropped-out labels remain saved and reappear when included again.
Add inserts the current video frame; the timeline's right-click menu inserts
the clicked frame. Neither operation redistributes existing proposals.

`config.py:DATASET_LAYOUT` owns the complete folder hierarchy, file names,
templates and inherited storage preferences. `dataset/paths.py` resolves
that declaration, including metadata companions and parameterized job folders.
`DATA_FORMATS` owns the NPY mask validity companion, NPZ scratch array and JSON
temporary-file patterns. Readers, writers and mask discovery share those declarations.
`FILE_PATTERNS` centralizes shared partial and lock filenames.

`dataset/annotations/keyframes.py` owns annotation state, mask history and persistence.
Its two files live together in `masks/key_frame/`, separate from ZoD/revision
annotations:

- `masks/key_frame/key_frames.json`: crop, selected source frames, animal count, prompts
  and model identifier. Progress and label status are derived, not duplicated.
- `masks/key_frame/key_frames.h5`: `frames` (K original frame numbers) and `masks` (K,H,W uint8
  integer label images), chunked per frame with lossless LZF compression.
- No RGB images, extracted JPEGs or persistent thumbnail caches. Video frames
  are read directly; small preview images are held in memory.
- File locations come from `Dataset`; names come from `config.py`. Each file
  is replaced atomically. A failed HDF5 commit restores the previous JSON.
- Save writes the current frame's label; unsaved edits prompt on navigation or
  close. Mask Undo/Redo is in memory, limited to 20 operations per editing frame.
- Prediction returns the highest finite-score candidate for each prompted
  animal. `analysis/keyframes.py` selects candidates and composes the preview.
  Overlapping proposals favor the higher SAM score, then the smaller ID on
  ties. Animals without predictions retain their current masks.
- The default SAM action predicts all prompted animals; Predict selected runs
  one animal while retaining the other proposals. Both reuse the same image
  embedding. Editing points invalidates only that animal's pending proposal.
- Predictions remain separate from edited masks. Select mode picks an animal
  by clicking its displayed mask, including after zoom/pan. Accept selected
  applies exactly that animal's visible region; Reject selected removes its
  proposal. Approve all accepts the remaining proposals. Each acceptance is
  undoable; Save is blocked while any proposal awaits review. Discard leaves
  accepted masks and prompts unchanged. Navigation prompts before dropping a
  preview. No predicted pixels are saved without acceptance.
- Paint and Erase work on both accepted masks and pending previews. Preview
  edits use the same `MaskDraft` stroke history, remain unapproved and survive
  per-animal review. Accepted regions touched during preview editing become
  proposals too, so Discard never changes accepted pixels. Brush edits produce
  disjoint proposal masks; their original SAM scores remain unchanged and
  newly touched IDs use 0 until another prediction. Acceptance/rejection or a
  new prediction starts a fresh preview edit history.
- Exclusive Select, Prompt, Paint, Erase and Pan buttons choose the interaction.
  Paint/Erase use left-drag and a hover circle scaled from the image-pixel
  radius into the current zoom. Each stroke is one Undo step. Left/Right
  cycles animal IDs, excluding background.

`segmentation/backends/sam.py` lazily loads either SAM2 or SAM3, reusing one image
embedding for repeated prompts. `processing/segmentation/keyframes.py` performs prompt prediction
with standard logging. The UI runs it in a worker thread. Upstream source and
checkpoints stay in `external/sam/`. The explicit installer supports SAM2 and
Cutie; SAM3 requires a manually installed package and checkpoint. Inference
does not download models. Cutie execution is separate from this editor.
The Model selector updates the open mask editor without discarding masks or
prompts. `ui/key_frame_model.py` checks available models through the segmentation
adapter and grays out entries with missing checkpoints or failed imports.
Tooltips show the reason. Checks refresh on opening the selector or choosing
a checkpoint. Device selection uses working CUDA or CPU automatically; an
explicit device argument is respected. Model controls are disabled during
prediction. Manual editing remains available even without a usable SAM model.

The editor's Local/SSH choice reuses the saved installation selector, omitting
the remote dataset field. `RemoteImageSegmenter` in `segmentation/image.py`
implements the image predictor interface through `processing/remote/session.py` and
`processing/segmentation/keyframes.py`. One SSH process retains the existing SAM adapter
and its current image embedding. Compressed NumPy arrays preserve RGB pixels,
point prompts, candidate masks and scores exactly; decoding disables pickle.
Availability checks run asynchronously on the chosen host. Requests have a
300-second timeout, and closing a session interrupts pending pipe operations.
Model/target changes and window closure release the process. A failed prediction
clears the session so the next request sends the image again. Automatic installation
uses the same target; SAM3 remains manual.

SAM requirements are separate from the basic viewer dependencies. Both need
compatible PyTorch/TorchVision, Pillow, tqdm and iopath. SAM2 also needs
hydra-core/OmegaConf. SAM3 additionally uses timm, einops, ftfy, regex,
typing_extensions, huggingface_hub, pycocotools and psutil. The checked upstream
SAM3 source imports pkg_resources, supplied by setuptools below version 81.
Notebook/training extras are not required for key-frame editing.

The workstation's python314 environment currently has only the basic viewer/
analysis dependencies. SAM3 documents Python >=3.12 but its checked dependency
metadata pins numpy>=1.26,<2; NumPy 1.26 supports Python 3.9–3.12, not 3.14.
Do not downgrade NumPy or bypass this pin implicitly. SAM runtime installation
and real-model verification in python314 remain to be resolved; SAM3 also needs
an authorized checkpoint download. Availability checks import installed SAM
packages when a checkpoint exists; loading its weights waits until prediction.

Focused checks covered mask editing/Undo/Redo, mouse dragging and locked frames,
crop preservation, external storage, save/reload, failed-save rollback, unsaved
navigation and prediction delivery using a small test predictor. The six-vole
video's first and last frames opened without saving any dataset changes. Existing
video-rendering and timeline checks passed after extracting the shared widgets.
Real SAM inference has not yet been verified in python314.
SAM2.1 Tiny inference was verified on a six-vole frame in the local Python 3.10
environment on CPU; SAM3 inference remains unverified.
SAM2.1 Tiny candidates, scores and metadata also matched local inference exactly
through the persistent subprocess transport on a synthetic image. Focused checks
covered image reuse, connection failure, timeout, cancellation and host selection.
Actual SSH-server inference remains unverified.

## Video-object segmentation and background jobs

The first VOS interface is implemented. `segmentation/image.py` and
`segmentation/video.py` define independent image and sequence interfaces;
`segmentation/backends` contains SAM and Cutie adapters. VOS reference selection
and model state stay in segmentation; frame/label/coverage validation is in
`analysis/keyframes.py`. `processing/segmentation/vos.py` executes interval runs,
using the dataset video reader and transactional mask writer. Source RGB frames
are decoded directly; reverse traversal uses bounded forward-decoded blocks.
No JPEGs, reversed clips or intermediate PNG masks are generated.

The GUI reuses the existing player, video renderer, zoom group, opacity control,
key-frame thumbnails, timeline/legend and splitter widgets. Model parameters use a
shared form. Model adapters supply controls and reference capabilities; naming
and defaults remain in config.py. The default is independent starting-key-frame
forward/reverse runs, native output and internal size 480. Both ends and All key frames
preload permanent references; transient sensory state is reset before propagation.
Reference IDs must represent the same animals across the selected key frames.

`processing/jobs.py` starts detached Python workers. `processing/remote/client.py`
uses OpenSSH to reach `processing/remote/server.py`, which dispatches requests to
the existing job runner and dataset services. Remote Linux workers use a systemd user service when available to survive SSH logout.
There is no cluster adapter, permanent SAM service or web server. Job records/logs are owned by `dataset/workflow/jobs.py`.
Saved remote installations use application preferences through
`ui/remote_settings.py`. The shared `RemoteTarget` selector remembers each local
dataset's selected installation; remote directories belong to the dataset description. Registration checks core imports and the protocol
through the remote client and server without accessing a dataset or loading model weights.
Exported scripts call the same runner, normally or with --background. SSH inputs
are structured JSON; credentials stay in SSH. Submission IDs prevent duplicate
launch on repeated requests. Workers read key-frame inputs into memory on startup.
Status is written atomically with a heartbeat, and cancellation is cooperative.
Reconnection does not restore interrupted model state. Output locks prevent
concurrent writers, existing outputs are refused, and incomplete files remain
separate. Dataset transfers belong to the dataset viewer. Consolidation compares copies,
collects conflict choices, verifies published bytes and writes a self-contained destination description.

`dataset/storage/remote_reader.py` owns one persistent SSH session for a mask pair. The server uses
`open_masks()` and its registered HDF5/NPY/NPZ adapters. Frames are transferred on demand;
recording plots use temporary snapshots of the much smaller numerical analysis products.
`dataset/storage/inventory.py` scans local, mounted and registered SSH locations;
`dataset/workflow/status.py` combines their available results and workflow status.
Analysis windows submit jobs to
the selected installation. The shared console polls logs asynchronously.

Masks retain source indices and an explicit validity array outside the crop.
Full-recording analysis currently rejects cropped output. Focused checks cover
frame order, key frames, references, crop validity, cancellation, output protection,
background failure reporting, script execution, remote file transfers and off-screen UI.
Remote mask reads and a small remote analysis job were verified on the registered Linux workstation.
Non-Linux hosts remain unverified.
