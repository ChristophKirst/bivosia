"""Key-frame selection, prompt validation and integer-mask editing.

Examples
--------
>>> import numpy as np
>>> from bivosia.analysis.key_frames import add_frame, move_frame, paint_mask, propose_frames
>>> propose_frames(0, 100, 5)
[0, 25, 50, 75, 100]
>>> propose_frames(0, 100, 4, locked=(40,))
[0, 40, 70, 100]
>>> move_frame([0, 40, 100], 40, 45, (0, 100), ())
[0, 45, 100]
>>> add_frame([0, 40, 100], 25, (0, 100))
[0, 25, 40, 100]
>>> add_frame([0, 40, 100], 40, (0, 100))
[0, 40, 100]
>>> mask = np.zeros((5, 5), dtype=np.uint8)
>>> paint_mask(mask, (2, 2), (2, 2), 1, 2)[2, 2].item()
2
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

import numpy as np
import cv2


def validate_crop(start, end, frame_count):
    """Validate inclusive source-frame bounds for bidirectional segmentation."""
    if not 0 <= start < end < frame_count:
        raise ValueError("Crop must contain at least two frames within the video")


def fixed_frames(crop, labeled):
    """Return crop endpoints and all labeled frames inside the crop."""
    return sorted({*crop, *(f for f in labeled if crop[0] <= f <= crop[1])})


def propose_frames(start, end, count, locked=()):
    """Space proposals between fixed labels and endpoints.

    Allocate each additional frame to the gap with the widest resulting
    spacing, then distribute that gap's proposals evenly. Fixed labels are
    never moved, including when the requested number of frames changes.
    """
    fixed = fixed_frames((start, end), locked)
    if start >= end or not len(fixed) <= count <= end - start + 1:
        raise ValueError(f"Frames must be between {len(fixed)} and {end - start + 1}")
    gaps = np.diff(fixed)
    extra = np.zeros(len(gaps), dtype=int)
    for _ in range(count - len(fixed)):
        spacing = np.where(extra < gaps - 1, gaps / (extra + 1), -1)
        extra[int(np.argmax(spacing))] += 1
    result = [fixed[0]]
    for left, right, number in zip(fixed, fixed[1:], extra):
        result.extend(np.rint(np.linspace(left, right, int(number) + 2)[1:]).astype(int).tolist())
    return result


def add_frame(frames, frame, crop):
    """Insert a key frame inside the inclusive crop without moving other frames."""
    if not crop[0] <= frame <= crop[1]:
        raise ValueError("Proposed frames must be inside the crop")
    return sorted({*frames, frame})


def move_frame(frames, source, target, crop, locked):
    """Move an unlabeled interior proposal without changing any other frame."""
    if source not in frames:
        raise ValueError("Select a proposed frame first")
    if source in crop or source in locked:
        raise ValueError("Labeled frames and crop boundaries cannot be moved")
    if not crop[0] < target < crop[1]:
        raise ValueError("Proposed frames must be inside the crop")
    if target != source and target in frames:
        raise ValueError("This frame is already selected")
    return sorted(target if f == source else f for f in frames)


def cropped_frames(frames, crop, locked):
    """Keep interior selections and add the new crop endpoints."""
    return sorted({*fixed_frames(crop, locked), *(f for f in frames if crop[0] <= f <= crop[1])})


def validate_mask(mask, shape, animals):
    """Check image geometry and animal IDs; return an independent uint8 mask."""
    mask = np.asarray(mask)
    if not 1 <= animals <= 255:
        raise ValueError("Animals must be between 1 and 255")
    if mask.shape != tuple(shape) or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("Mask must be an integer array matching the video frame")
    if mask.size and (mask.min() < 0 or mask.max() > animals):
        raise ValueError("Mask contains IDs outside the animal range")
    return mask.astype(np.uint8, copy=True)


def validate_prompts(prompts, shape, animals):
    """Check source-image x/y coordinates and animal IDs (0 = background)."""
    height, width = shape
    for point in prompts:
        if (len(point) != 3 or not all(np.isfinite(v) for v in point)
                or not 0 <= point[0] < width or not 0 <= point[1] < height
                or point[2] != int(point[2]) or not 0 <= point[2] <= animals):
            raise ValueError("Invalid key-frame prompt")


def point_prompts(prompts, animal):
    """Use the chosen animal as positive; other IDs and background as negative."""
    points = np.asarray(prompts, dtype=float).reshape(-1, 3)
    labels = (points[:, 2] == animal).astype(np.int32)
    if animal <= 0 or not labels.any():
        raise ValueError("Place a positive point for the selected animal")
    return points[:, :2].astype(np.float32), labels


def prompted_animals(prompts, animal=None):
    """Return animal IDs with positive prompts, excluding background.

    >>> from bivosia.analysis.key_frames import prompted_animals
    >>> prompted_animals([[1, 2, 3], [4, 5, 0], [6, 7, 1]])
    [1, 3]
    """
    animals = sorted({int(point[2]) for point in prompts if point[2] > 0})
    if not animals:
        raise ValueError("Place at least one animal point before prediction")
    if animal is not None:
        if animal not in animals:
            raise ValueError("Place a positive point for the selected animal")
        return [animal]
    return animals


def best_mask(masks, scores):
    """Select the candidate with the highest finite prediction score.

    >>> import numpy as np
    >>> from bivosia.analysis.key_frames import best_mask
    >>> best_mask(np.array([[[False]], [[True]]]), [0.2, 0.9])[0].item()
    True
    """
    masks, scores = np.asarray(masks, dtype=bool), np.asarray(scores, dtype=float)
    if masks.ndim != 3 or scores.shape != (len(masks),) or not np.isfinite(scores).any():
        raise ValueError("Segmenter returned invalid masks or scores")
    index = int(np.argmax(np.where(np.isfinite(scores), scores, -np.inf)))
    return masks[index].copy(), float(scores[index])


def combine_masks(mask, predictions):
    """Replace predicted animal regions, preserving animals without predictions.

    Predictions map animal IDs to (mask, score). Overlaps go to the higher
    score; equal scores favor the smaller animal ID.

    >>> import numpy as np
    >>> from bivosia.analysis.key_frames import combine_masks
    >>> base = np.array([[1, 0, 3]], dtype=np.uint8)
    >>> proposals = {1: (np.array([[0, 1, 1]]), 0.7),
    ...              2: (np.array([[1, 1, 0]]), 0.9)}
    >>> combine_masks(base, proposals).tolist()
    [[2, 2, 3]]
    """
    result = mask.copy()
    result[np.isin(result, list(predictions))] = 0
    for animal, (candidate, score) in sorted(predictions.items(), key=lambda item: (-item[1][1], item[0])):
        candidate = np.asarray(candidate, dtype=bool)
        if candidate.shape != mask.shape or not 1 <= animal <= 255 or not np.isfinite(score):
            raise ValueError("Invalid predicted mask, animal ID or score")
        result[candidate & (result == 0)] = animal
    return result


def accept_masks(mask, preview, animals):
    """Apply only selected animal regions exactly as shown in a mask preview.

    >>> import numpy as np
    >>> from bivosia.analysis.key_frames import accept_masks
    >>> accept_masks(np.array([[1, 2, 0]], dtype=np.uint8),
    ...              np.array([[0, 1, 2]], dtype=np.uint8), [1]).tolist()
    [[0, 1, 0]]
    """
    if preview.shape != mask.shape:
        raise ValueError("Preview and mask dimensions differ")
    result = mask.copy()
    result[np.isin(result, animals)] = 0
    selected = np.isin(preview, animals)
    result[selected] = preview[selected]
    return result


def edited_predictions(predictions, before, after):
    """Retain edited preview regions as proposals, including touched accepted IDs.

    Proposal masks become disjoint so recomposing them preserves brush edits.

    >>> import numpy as np
    >>> from bivosia.analysis.key_frames import edited_predictions
    >>> before = np.array([[1, 2]], dtype=np.uint8)
    >>> after = np.array([[1, 0]], dtype=np.uint8)
    >>> result = edited_predictions({1: (before == 1, .8)}, before, after)
    >>> sorted(result)
    [1, 2]
    >>> result[2][0].tolist()
    [[False, False]]
    """
    changed = before != after
    animals = set(predictions) | set(np.unique(before[changed])) | set(np.unique(after[changed]))
    return {int(animal): (after == animal, predictions[animal][1] if animal in predictions else 0.0)
            for animal in sorted(animals) if animal > 0}


def paint_mask(mask, start, end, radius, animal):
    """Paint a continuous round stroke; animal 0 erases pixels."""
    result = mask.copy()
    first, last = tuple(map(round, start)), tuple(map(round, end))
    cv2.line(result, first, last, int(animal), max(1, 2 * int(radius)))
    cv2.circle(result, last, int(radius), int(animal), -1)
    return result


def validate_key_frames(crop, frames, masks, prompts, shape, frame_count, animals):
    """Check a key-frame annotation before loading or writing either data file."""
    validate_crop(*crop, frame_count)
    if not 1 <= animals <= 255:
        raise ValueError("Animals must be between 1 and 255")
    if frames != sorted(set(frames)) or not set(crop).issubset(frames):
        raise ValueError("Key frames must be unique, sorted and include crop endpoints")
    if any(not crop[0] <= f <= crop[1] for f in frames):
        raise ValueError("Selected frame outside crop")
    for frame, mask in masks.items():
        if not 0 <= frame < frame_count:
            raise ValueError("Saved mask frame is outside the video")
        validate_mask(mask, shape, animals)
        if crop[0] <= frame <= crop[1] and frame not in frames:
            raise ValueError("A labeled frame inside the crop is missing from the selection")
    for frame, points in prompts.items():
        if not 0 <= int(frame) < frame_count:
            raise ValueError("Prompt frame is outside the video")
        validate_prompts(points, shape, animals)
