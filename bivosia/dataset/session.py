"""Revision state, persistence, undo history and local link invalidation.

Examples
--------
>>> from bivosia.dataset.session import RevisionSession
>>> import numpy as np
>>> missing = np.zeros(20, dtype=bool)
>>> session = RevisionSession.create(
...     {"seed": 0.6, "recover": 0.9, "gap": 0}, [(5, 9)], missing, missing, 2)
>>> session.edit(session.zones[0].id, "source", source="reverse")
>>> session.zones[0].source
'reverse'
>>> session.undo()
>>> session.zones[0].source
'forward'
"""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import hashlib
import json

from bivosia.analysis.linking import boundaries, initial_draft, permutation, validate_draft
from bivosia.analysis.revision import RevisionZone, edit_zones, new_zone, revision_fingerprint, validate_zones
from bivosia.dataset.jsonio import annotation_filename, atomic_json


class RevisionSession:
    def __init__(self, header, automatic, zones, missing_forward, missing_reverse, n, link_frames=()):
        self.frame_count, self.n = len(missing_forward), int(n)
        if len(missing_reverse) != self.frame_count or self.n < 2:
            raise ValueError("Invalid frame or mask counts")
        self.header = deepcopy(header)
        self.automatic = tuple((int(a), int(b)) for a, b in automatic)
        if any(not 0 <= a <= b < self.frame_count for a, b in self.automatic):
            raise ValueError("Auto zone outside the recording")
        self.zones = validate_zones(zones, self.frame_count)
        self.link_frames = set(link_frames)
        self.missing_forward, self.missing_reverse = missing_forward, missing_reverse
        self.links = {}
        self.undo_stack, self.redo_stack = [], []
        self.revision = 0
        self.path = None
        self._disk_digest = None
        self.dirty = True
        self.export = None
        self._sync_links()

    @classmethod
    def create(cls, header, automatic, missing_forward, missing_reverse, n):
        return cls(header, automatic, [new_zone(a, b, missing_forward, missing_reverse) for a, b in automatic],
                   missing_forward, missing_reverse, n)

    @property
    def jobs(self):
        return boundaries(self.zones, self.frame_count, self.link_frames)

    def set_link_frame(self, frame, enabled):
        """Add/remove an independent boundary; leave all zone records untouched."""
        if not isinstance(frame, int) or not 0 <= frame < self.frame_count - 1:
            raise ValueError("Choose a transition before the last frame")
        if enabled:
            existing = next((j for j in self.jobs if j.left.end == frame and j.right.start == frame + 1), None)
            if existing is not None:
                return existing.id
        frames = self.link_frames | {frame} if enabled else self.link_frames - {frame}
        jobs = boundaries(self.zones, self.frame_count, frames)
        job = next((j for j in jobs if j.id == f"frame:{frame}"), None)
        if enabled and job is None:
            raise ValueError("Both frames must be valid; use the existing bridge across an invalid zone")
        before = self._state()
        self.link_frames = frames
        self._sync_links()
        self._finish(before, "Add linking boundary" if enabled else "Remove linking boundary")
        return job.id if job else None

    def _sync_links(self):
        # Removed links remain in undo history.
        updated = {}
        for job in self.jobs:
            record = self.links.get(job.id)
            if record is None:
                record = {"context": job.context, "status": "unseen", "draft": None, "approved": None}
            elif record["context"] != job.context:
                record = deepcopy(record)
                record["context"] = job.context
                record["status"] = "needs_review"
                # Geometry changes retain the proposal but require review.
                old = record.get("draft") or record.get("approved")
                draft = initial_draft(job)
                if old and old.get("mapping") is not None:
                    draft.update(mapping=old["mapping"], manual=True)
                    # These directions identify the masks used by the saved permutation.
                    for side in ("left", "right"):
                        draft[f"{side}_source"] = old[f"{side}_source"]
                        draft[f"{side}_frame"] = old[f"{side}_frame"]
                record["draft"] = draft
            updated[job.id] = record
        self.links = updated

    def _state(self):
        return {"zones": {z.id: z.record() for z in self.zones}, "links": deepcopy(self.links),
                "link_frames": {str(t): t for t in self.link_frames}}

    def _finish(self, before, label, history=True):
        after = self._state()
        changes = []
        for kind in before:
            for key in before[kind].keys() | after[kind].keys():
                a, b = before[kind].get(key), after[kind].get(key)
                if a != b:
                    changes.append({"kind": kind, "id": key, "before": a, "after": b})
        if not changes:
            return
        if history:
            self.undo_stack.append({"label": label, "changes": changes})
            self.undo_stack = self.undo_stack[-20:]
            self.redo_stack.clear()
        self.revision += 1
        self.dirty = True

    def edit(self, zone_id, operation, *, link_transport=None, **kwargs):
        before = self._state()
        old_jobs = {job.id: job for job in self.jobs}
        self.zones = edit_zones(self.zones, zone_id, operation, frame_count=self.frame_count,
                               missing_forward=self.missing_forward, missing_reverse=self.missing_reverse, **kwargs)
        try:
            self._sync_links()
            if link_transport is not None:
                for job in self.jobs:
                    old = old_jobs.get(job.id)
                    if old and (old.left.source, old.right.source) != (job.left.source, job.right.source):
                        self.links[job.id]["draft"] = link_transport(job, self.links[job.id], old)
        except BaseException:
            self.zones = validate_zones([RevisionZone(**z) for z in before["zones"].values()], self.frame_count)
            self.links = before["links"]
            raise
        self._finish(before, f"Zone {operation}")

    def visit_zone(self, zone_id):
        before = self._state()
        self.zones = [replace(z, status="draft") if z.id == zone_id and z.status == "unseen" else z for z in self.zones]
        self._finish(before, "Visit zone", history=False)

    def set_draft(self, job_id, draft, *, history=True):
        validate_draft(draft, self.n, self.frame_count)
        before = self._state()
        record = self.links[job_id]
        if record.get("draft") == draft:
            return
        record["draft"] = deepcopy(draft)
        record["status"] = "draft"
        self._finish(before, "Edit boundary link", history)

    def approve(self, job_id, canonical_mapping):
        job = next(j for j in self.jobs if j.id == job_id)
        record = self.links[job_id]
        draft = record.get("draft")
        if not draft or draft.get("mapping") is None:
            raise ValueError("This boundary has no linking proposal")
        validate_draft(draft, self.n, self.frame_count)
        before = self._state()
        approved = deepcopy(draft)
        approved["canonical_mapping"] = list(permutation(canonical_mapping, self.n))
        record.update(approved=approved, status="approved", context=job.context)
        self._finish(before, "Approve boundary")

    def _restore(self, command, side):
        state = self._state()
        for change in command["changes"]:
            value = deepcopy(change[side])
            if value is None:
                state[change["kind"]].pop(change["id"], None)
            else:
                state[change["kind"]][change["id"]] = value
        self.zones = validate_zones([RevisionZone(**z) for z in state["zones"].values()], self.frame_count)
        self.link_frames = set(state["link_frames"].values())
        self.links = state["links"]
        self._sync_links()
        self.revision += 1
        self.dirty = True

    def undo(self):
        if self.undo_stack:
            command = self.undo_stack.pop()
            self._restore(command, "before")
            self.redo_stack.append(command)

    def redo(self):
        if self.redo_stack:
            command = self.redo_stack.pop()
            self._restore(command, "after")
            self.undo_stack.append(command)

    def fingerprint(self):
        return revision_fingerprint(self.header, (zone.record() for zone in self.zones),
                                    self.links, self.link_frames)

    def save(self, path=None):
        target = Path(path or self.path).resolve()
        expected = annotation_filename(self.header["seed"], self.header["recover"])
        if not target.name.endswith(expected):
            raise ValueError(f"Filename must end in {expected} (optional labeler prefix)")
        if target.exists():
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if self.path != target or self._disk_digest != digest:
                raise ValueError("File already exists or was changed by another session. Open it or choose another filename.")
        payload = {"schema": 1, "header": self.header, "auto_zones": self.automatic,
                   "zones": [z.record() for z in self.zones], "links": self.links,
                   "link_frames": sorted(self.link_frames),
                   "history": {"undo": self.undo_stack, "redo": self.redo_stack},
                   "revision": self.revision, "export": self.export}
        atomic_json(target, payload)
        self.path = target
        self._disk_digest = hashlib.sha256(target.read_bytes()).hexdigest()
        self.dirty = False

    @classmethod
    def load(cls, path, missing_forward, missing_reverse, n):
        path = Path(path).resolve()
        raw = path.read_bytes()
        payload = json.loads(raw)
        if payload.get("schema") != 1:
            raise ValueError("Not a revision/linking annotation file")
        session = cls(payload["header"], payload["auto_zones"], [RevisionZone(**z) for z in payload["zones"]],
                      missing_forward, missing_reverse, n, payload.get("link_frames", ()))
        session.links = payload["links"]
        for record in session.links.values():
            for key in ("draft", "approved"):
                draft = record.get(key)
                if draft is not None:
                    validate_draft(draft, session.n, session.frame_count)
                    if "canonical_mapping" in draft:
                        permutation(draft["canonical_mapping"], session.n)
        session._sync_links()
        session.undo_stack = payload.get("history", {}).get("undo", [])[-20:]
        session.redo_stack = payload.get("history", {}).get("redo", [])[-20:]
        session.revision = payload.get("revision", 0)
        session.export = payload.get("export")
        session.path, session._disk_digest = path, hashlib.sha256(raw).hexdigest()
        session.dirty = False
        return session
