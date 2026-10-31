"""Boundary matching, identity propagation and permutation algebra.

Examples
--------
>>> from bivosia.analysis.mask_linking import inverse, swap_endpoint
>>> inverse((2, 0, 1))
(1, 2, 0)
>>> swap_endpoint((0, 1), "right", 0, 1)
(1, 0)
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import hashlib
import json
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from bivosia.config import ANALYSIS_DEFAULTS
from bivosia.analysis.mask_quality import frame_iou_matrix, frame_quality_measures, optimal_assignments
from bivosia.analysis.mask_features import mask_presence
from bivosia.analysis.masks import relabel_mask
from bivosia.analysis.zones_of_disagreement import segments, zone_has_missing_masks


def permutation(values, n):
    values = tuple(int(x) for x in values)
    if len(values) != n or sorted(values) != list(range(n)):
        raise ValueError("A link must be a one-to-one permutation of all mask labels")
    return values


def inverse(values):
    return tuple(int(x) for x in np.argsort(values))


def transport(mapping, old_to_new_left, old_to_new_right):
    """new_h = C_right ◦ old_h ◦ inverse(C_left). Labels here are zero-based."""
    n = len(mapping)
    mapping = permutation(mapping, n)
    left = inverse(permutation(old_to_new_left, n))
    right = permutation(old_to_new_right, n)
    return tuple(right[mapping[left[i]]] for i in range(n))


def swap_endpoint(initial, side, source_label, target_label):
    """Swap two assignments relative to the supplied initial permutation."""
    initial = permutation(initial, len(initial))
    if not 0 <= source_label < len(initial) or not 0 <= target_label < len(initial):
        raise ValueError("Mask label outside permutation")
    if side == "left":
        return inverse(swap_endpoint(inverse(initial), "right", source_label, target_label))
    if side != "right":
        raise ValueError("Endpoint side must be left or right")
    result = list(initial)
    partner = inverse(initial)[target_label]
    result[source_label], result[partner] = target_label, initial[source_label]
    return tuple(result)


@dataclass(frozen=True)
class MaskMatchResult:
    """A proposed mask-label mapping with its ambiguity warnings and quality."""

    mapping: tuple[int, ...]
    warnings: tuple[str, ...]
    quality: float


def match_masks(left, right, n):
    if left.shape != right.shape:
        raise ValueError("Masks must use the same image coordinates")
    matrix, _left_area, _right_area = frame_iou_matrix(left, right, n)
    mapping = tuple(int(i) for i in optimal_assignments(matrix[None])[0])
    clean = np.nan_to_num(matrix, nan=0., posinf=0., neginf=0.)
    assigned = clean[np.arange(n), mapping]
    warnings = []
    if not np.isfinite(matrix).all():
        warnings.append("Missing mask at a linking anchor")
    if np.any(assigned <= 0):
        warnings.append("Some proposed pairs have zero pixel overlap")
    # Detect tied optima by excluding each assigned pair in turn.
    best = float(assigned.sum())
    tolerance = ANALYSIS_DEFAULTS['mask_linking']['tie_absolute_tolerance']
    for i, j in enumerate(mapping):
        alternate = clean.copy()
        alternate[i, j] = -1e9
        rows, cols = linear_sum_assignment(alternate, maximize=True)
        if float(alternate[rows, cols].sum()) >= best - tolerance:
            warnings.append("Hungarian proposal is ambiguous (tied assignments)")
            break
    quality, _, _, _ = frame_quality_measures(matrix[None], np.asarray(mapping)[None])
    return MaskMatchResult(mapping, tuple(warnings), float(quality[0]))


@dataclass(frozen=True)
class Boundary:
    id: str
    left: object
    right: object
    context_regions: tuple | None = None

    @property
    def context(self):
        regions = self.context_regions or (self.left, self.right)
        return hashlib.sha256(json.dumps([asdict(s) for s in regions], sort_keys=True).encode()).hexdigest()

    @property
    def skipped_frames(self):
        return self.right.start - self.left.end - 1


def _link_partition(zones, frame_count, link_frames):
    """Split identity propagation at t→t+1 without changing annotation zones."""
    frames = sorted(set(link_frames))
    if any(not isinstance(t, int) or not 0 <= t < frame_count - 1 for t in frames):
        raise ValueError("Linking boundaries require 0 ≤ t < last frame")
    parts, parents = [], {}
    for region in segments(zones, frame_count):
        cuts = [region.start, *(t + 1 for t in frames if region.start <= t < region.end), region.end + 1]
        for start, stop in zip(cuts, cuts[1:]):
            key = region.id if start == region.start else f"{region.id}:after:{start - 1}"
            part = replace(region, id=key, start=start, end=stop - 1)
            parts.append(part)
            parents[key] = region
    return parts, parents


def link_segments(zones, frame_count, link_frames=()):
    return _link_partition(zones, frame_count, link_frames)[0]


def boundaries(zones, frame_count, link_frames=()):
    parts, parents = _link_partition(zones, frame_count, link_frames)
    valid = [s for s in parts if s.source != "invalid"]
    result = []
    for a, b in zip(valid, valid[1:]):
        left, right = parents[a.id], parents[b.id]
        key = f"frame:{a.end}" if a.end in link_frames and b.start == a.end + 1 else f"{left.id}>{right.id}"
        result.append(Boundary(key, a, b, (left, right)))
    return result


def initial_draft(boundary):
    return {"left_frame": boundary.left.end, "right_frame": boundary.right.start,
            "left_source": boundary.left.source, "right_source": boundary.right.source,
            "mapping": None, "manual": False}


def automatic_link(job, mask, n):
    """Propagate the incoming pass through invalid frames before matching passes.

    Adjacent valid endpoints use their direct optimal IoU assignment. Across
    an invalid interval, labels stay in the incoming pass until the next valid
    frame, where its masks are matched to the outgoing pass at that same frame.
    """
    left_frame = job.right.start if job.skipped_frames else job.left.end
    match = match_masks(mask(job.left.source, left_frame), mask(job.right.source, job.right.start), n)
    draft = dict(initial_draft(job), left_frame=left_frame, mapping=list(match.mapping),
                 canonical_mapping=list(match.mapping))
    return {'context': job.context, 'status': 'automatic', 'draft': None, 'approved': None,
            'automatic': draft, 'quality': match.quality if math.isfinite(match.quality) else None,
            'warnings': list(match.warnings)}


def validate_draft(draft, n, frame_count):
    if not all(0 <= draft[f"{side}_frame"] < frame_count for side in ("left", "right")):
        raise ValueError("Link frames must lie inside the video")
    for side in ("left", "right"):
        if draft[f"{side}_source"] not in ("forward", "reverse"):
            raise ValueError("Linking anchors require a valid mask direction")
    if draft.get("mapping") is not None:
        permutation(draft["mapping"], n)


def approved_mapping(job, record, n, frame_count):
    approved = record.get("approved")
    if record.get("status") != "approved" or record.get("context") != job.context or not approved:
        return None
    validate_draft(approved, n, frame_count)
    return permutation(approved["canonical_mapping"], n)


def propagate_identities(partition, jobs, mappings, n):
    """Propagate supplied boundary permutations; absent links separate components."""
    valid = [s for s in partition if s.source != "invalid"]
    graph = {s.id: [] for s in valid}
    errors = []
    for job in jobs:
        if job.id not in mappings:
            continue
        h = permutation(mappings[job.id], n)
        graph[job.left.id].append((job.right.id, h))
        graph[job.right.id].append((job.left.id, inverse(h)))
    maps, components = {}, {}
    for root in valid:
        if root.id in maps:
            continue
        component = len(set(components.values()))
        maps[root.id] = tuple(range(n))
        components[root.id] = component
        pending = [root.id]
        while pending:
            current = pending.pop()
            for target, h in graph[current]:
                candidate = tuple(maps[current][i] for i in inverse(h))
                if target in maps:
                    if maps[target] != candidate:
                        errors.append(f"Inconsistent identity constraints at {target}")
                else:
                    maps[target] = candidate
                    components[target] = component
                    pending.append(target)
    return maps, components, errors


@dataclass(frozen=True)
class LinkResult:
    partition: tuple
    maps: dict
    components: dict
    regions: tuple  # inclusive start, end, connected component
    errors: tuple
    jobs: tuple
    automatic: dict  # boundary ID -> accepted automatic MaskMatchResult


@dataclass(frozen=True)
class BoundaryLinks:
    partition: tuple
    jobs: tuple
    mappings: dict
    n: int
    errors: tuple
    automatic: dict

    def resolve(self, current_id=None, current_mapping=None):
        mappings = dict(self.mappings)
        if current_id is not None and current_mapping is not None:
            mappings[current_id] = permutation(current_mapping, self.n)
        maps, components, errors = propagate_identities(self.partition, self.jobs, mappings, self.n)
        regions = []
        for part in self.partition:
            if part.id not in components:
                continue
            component = components[part.id]
            if regions and regions[-1][1] + 1 == part.start and regions[-1][2] == component:
                regions[-1] = (regions[-1][0], part.end, component)
            else:
                regions.append((part.start, part.end, component))
        automatic = dict(self.automatic)
        if current_id in automatic and current_mapping is not None:
            if tuple(current_mapping) != automatic[current_id].mapping:
                automatic.pop(current_id)
        return LinkResult(self.partition, maps, components, tuple(regions),
                          self.errors + tuple(errors), self.jobs, automatic)


def prepare_links(session, boundary_match=None):
    """Resolve saved approved/automatic links, optionally proposing missing links.

    Without a callback, unresolved boundaries remain unlinked. With a callback,
    adjacent-boundary proposals use the selected masks at their endpoints.
    Their quality is the minimum signed row/column IoU margin of the selected
    assignment. Invalid bridges and incomplete or tied matches need review.
    Manual or moved-anchor drafts remain pending until approved.
    """
    partition = tuple(link_segments(session.zones, session.frame_count, session.link_frames))
    jobs = tuple(boundaries(session.zones, session.frame_count, session.link_frames))
    mappings, automatic, errors = {}, {}, []
    for job in jobs:
        try:
            record = session.links.get(job.id, {})
            mapping = approved_mapping(job, record, session.n, session.frame_count)
            if (mapping is None and record.get('status') == 'automatic'
                    and record.get('context') == job.context):
                proposal = record['automatic']
                validate_draft(proposal, session.n, session.frame_count)
                mapping = permutation(proposal['canonical_mapping'], session.n)
                automatic[job.id] = MaskMatchResult(mapping, tuple(record.get('warnings', ())),
                                          float(record['quality']) if record.get('quality') is not None else float('nan'))
            draft = record.get('draft')
            edited = draft is not None and (draft.get('manual') or any(
                draft[key] != value for key, value in initial_draft(job).items()
                if key not in ('mapping', 'manual')))
            if mapping is None and boundary_match is not None and not job.skipped_frames and not edited:
                match = boundary_match(job)
                if not match.warnings and np.isfinite(match.quality):
                    mapping = match.mapping
                    automatic[job.id] = match
            if mapping is not None:
                mappings[job.id] = permutation(mapping, session.n)
        except (ValueError, KeyError, TypeError) as error:
            errors.append(f"Invalid boundary {job.id}: {error}")
    return BoundaryLinks(partition, jobs, mappings, session.n, tuple(errors), automatic)


def unresolved_links(session):
    """Select unlinked boundaries without replacing drafts or approvals."""
    links = prepare_links(session)
    return tuple(job for job in links.jobs if job.id not in links.mappings
                 and session.links.get(job.id, {}).get('draft') is None
                 and session.links.get(job.id, {}).get('approved') is None)


def critical_links(links):
    """Select unresolved links and automatic links with warnings or invalid bridges.

    Use saved assignments and diagnostics. A valid manual approval acknowledges
    existing warnings; an invalidated approval remains unresolved.
    """
    result = []
    for job in links.jobs:
        match = links.automatic.get(job.id)
        if job.id not in links.mappings or (match is not None and (
                match.warnings or not np.isfinite(match.quality) or job.skipped_frames)):
            result.append(job)
    return tuple(result)


def validation_errors(session, links):
    """Check reviewed zones, automatic/approved links and chosen-mask completeness."""
    accepted = {'reviewed', 'automatic'} if session.header.get('mode') == 'automatic' else {'reviewed'}
    errors = [f"Zone {i + 1} ({z.start}–{z.end}) has not been reviewed"
              for i, z in enumerate(session.zones) if z.status not in accepted]
    resolved = links.resolve()
    errors.extend(resolved.errors)
    errors.extend(f'Boundary needs review at {job.left.end} → {job.right.start}'
                  for job in links.jobs if job.id not in links.mappings)
    if len(set(resolved.components.values())) > 1:
        errors.append("Valid segments are not all linked to the first identity component")
    for z in session.zones:
        if zone_has_missing_masks(z, session.missing_forward, session.missing_reverse,
                                  automatic=session.header.get('mode') == 'automatic'):
            errors.append(f"Zone {z.start}–{z.end}: chosen direction still contains missing masks; split or mark invalid")
    return errors


class Linker:
    """Revise identity matches using supplied masks and framewise evidence.

    The mask accessor accepts a direction and frame and returns a label array.
    """

    def __init__(self, mask, n, quality, missing, switches):
        self.mask = mask
        self.n = n
        self.quality = quality
        self.missing = missing
        self.switches = switches
        self.count = len(quality)

    @lru_cache(maxsize=32)
    def conversion(self, old, new, frame, seed):
        if old == new:
            return tuple(range(self.n)), []
        match = match_masks(self.mask(old, frame), self.mask(new, frame), self.n)
        warnings = list(match.warnings)
        if (self.missing[frame] or self.switches[frame]
                or (np.isfinite(self.quality[frame]) and self.quality[frame] < seed)):
            warnings.append(f"F/R conversion at frame {frame} meets the detector's seed condition")
        return match.mapping, warnings

    def draft(self, job, record, seed):
        draft = dict(record.get("draft") or record.get("approved") or record.get('automatic') or initial_draft(job))
        draft.pop("canonical_mapping", None)
        warnings = list(draft.get("transport_warnings", []))
        match = match_masks(self.mask(draft["left_source"], draft["left_frame"]),
                            self.mask(draft["right_source"], draft["right_frame"]), self.n)
        warnings.extend(match.warnings)
        if draft.get("mapping") is None:
            draft["mapping"] = list(match.mapping)
        validate_draft(draft, self.n, self.count)
        warnings.extend(self.anchor_warnings(job, draft, seed))
        return draft, list(dict.fromkeys(warnings))

    def change_anchors(self, job, draft, seed, **changes):
        new = {**draft, **changes}
        validate_draft(new, self.n, self.count)
        warnings = []
        conversions = []
        for side in ("left", "right"):
            mapping, messages = self.conversion(draft[f"{side}_source"], new[f"{side}_source"], new[f"{side}_frame"], seed)
            conversions.append(mapping)
            warnings.extend(messages)
        if draft.get("manual") or any(draft[f"{s}_source"] != new[f"{s}_source"] for s in ("left", "right")):
            new["mapping"] = list(transport(draft["mapping"], *conversions))
            warnings.extend(match_masks(self.mask(new["left_source"], new["left_frame"]),
                                        self.mask(new["right_source"], new["right_frame"]), self.n).warnings)
        else:
            match = match_masks(self.mask(new["left_source"], new["left_frame"]),
                                self.mask(new["right_source"], new["right_frame"]), self.n)
            new["mapping"] = list(match.mapping)
            warnings.extend(match.warnings)
        warnings.extend(self.anchor_warnings(job, new, seed))
        return new, list(dict.fromkeys(warnings))

    def anchor_warnings(self, job, draft, seed):
        warnings = []
        if job.skipped_frames:
            warnings.append(f"Bridge over {job.skipped_frames:,} invalid frames; verify identities manually")
        for side, segment in (("left", job.left), ("right", job.right)):
            source, frame = draft[f"{side}_source"], draft[f"{side}_frame"]
            _, messages = self.conversion(segment.source, source, frame, seed)
            warnings.extend(messages)
            labels = set(np.unique(self.mask(source, frame)))
            if not set(range(1, self.n + 1)) <= labels:
                warnings.append(f"Missing mask at {side} anchor {frame}")
            if labels - set(range(self.n + 1)):
                warnings.append(f"Unexpected raw mask labels at {side} anchor")
        return warnings

    def transport_source_edit(self, job, record, old_job, seed):
        draft, _ = self.draft(job, record, seed)
        changes = {f"{side}_source": getattr(job, side).source for side in ("left", "right")
                   if getattr(old_job, side).source != getattr(job, side).source}
        draft, warnings = self.change_anchors(job, draft, seed, **changes)
        draft["manual"] = True
        draft["transport_warnings"] = warnings
        return draft

    def canonical_mapping(self, job, draft, seed):
        left, lw = self.conversion(job.left.source, draft["left_source"], draft["left_frame"], seed)
        right, rw = self.conversion(job.right.source, draft["right_source"], draft["right_frame"], seed)
        result = [inverse(right)[draft["mapping"][left[i]]] for i in range(self.n)]
        return result, list(dict.fromkeys(lw + rw + self.anchor_warnings(job, draft, seed)))

    @lru_cache(maxsize=512)
    def boundary_match(self, job):
        """Return the endpoint mask assignment and its signed IoU-margin quality."""
        return match_masks(self.mask(job.left.source, job.left.end),
                           self.mask(job.right.source, job.right.start), self.n)

    def clear(self):
        self.conversion.cache_clear()
        self.boundary_match.cache_clear()


def automatic_zone_frames(session):
    """Identify automatic zones where one missing animal is permitted."""
    frames = np.zeros(session.frame_count, dtype=bool)
    if session.header.get('mode') == 'automatic':
        for zone in session.zones:
            if zone.status == 'automatic':
                frames[zone.start:zone.end + 1] = True
    return frames


def preferred_mask(mask, mapping, n, *, allow_missing=False):
    """Validate the chosen mask and apply its saved global identity mapping."""
    if (~mask_presence(mask, n)).sum() > int(allow_missing):
        raise ValueError("Chosen masks are missing; split the zone or mark it invalid")
    return relabel_mask(mask, mapping)


def linked_frame(forward, reverse, source, mapping, n, *, quality=None, switch=False,
                 seed=ANALYSIS_DEFAULTS['zones_of_disagreement']['theta_quality'], allow_missing=False):
    """Return relabeled directions, source code and directional validity."""
    empty = np.zeros(forward.shape, dtype=np.uint16)
    if source == "invalid":
        return {"preferred": empty, "forward": empty, "reverse": empty}, 2, (False, False)
    masks = {"forward": forward, "reverse": reverse}
    raw = masks[source]
    preferred = preferred_mask(raw, mapping, n, allow_missing=allow_missing)
    other = "reverse" if source == "forward" else "forward"
    match = match_masks(raw, masks[other], n)
    uncertain = quality is not None and (not np.isfinite(quality) or quality < seed)
    other_valid = not (match.warnings or uncertain or switch)
    other_mask = empty
    if other_valid:
        other_map = tuple(mapping[i] for i in inverse(match.mapping))
        other_mask = relabel_mask(masks[other], other_map)
    values = {"preferred": preferred, source: preferred, other: other_mask}
    code = 0 if source == "forward" else 1
    valid = (True, other_valid) if code == 0 else (other_valid, True)
    return values, code, valid
