"""
ledgerhound.knowledge.dataset_card
================================

DatasetCard -- V3-011: one versioned object per dataset, merging what today
lives as unlinked fragments (the profiler digest, ad-hoc decisions,
findings) into a single thing that:

  - is created once per dataset, keyed by its resolved absolute path
  - is loaded on every subsequent session that touches the same data, so
    the session does NOT re-profile from scratch (the digest is
    deterministic for fixed data -- recomputing it wastes time)
  - is updated (appended to, never overwritten) as sessions add decisions,
    findings, and run references
  - gets pinned into the LLM's context on load, alongside the live-schema
    pin (V3-003), so "we've seen this dataset before" is visible from the
    first turn

This is the spine later drift-monitoring (Phase 6) and the KB (research
report #2) hang off -- see docs/v3.0/V3_GAP_ANALYSIS_AND_ROADMAP.md, gap G11.

Storage: one JSON + one MD file per dataset under
``<project>/.ledgerhound/dataset_cards/<key>.{json,md}``, where ``key`` is a
short stable hash of the resolved absolute data path. Cards require a
project -- persistence across sessions is the entire point; with no
project, callers simply don't use one and fall back to today's
in-session-only caching (mirrors how ``analyst.books`` degrades).

Quirks are deliberately NOT model-writable in v1: per the KB anti-junk-
drawer principle (an LLM auto-appending "quirks" every turn becomes noise,
not signal), ``add_quirk`` exists for a human-curated workflow (a future
``ledgerhound card`` CLI command) but no tool exposes it to the model yet.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

CARDS_DIR = "dataset_cards"
SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def card_key(path: str | Path) -> str:
    """Stable key for a data path -- the same file gets the same key across
    every future session, which is what makes cross-session reuse work."""
    resolved = str(Path(path).resolve())
    return hashlib.sha1(resolved.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


@dataclass
class DatasetCard:
    schema_version: int = SCHEMA_VERSION
    key: str = ""
    path: str = ""                    # resolved absolute path this card describes
    created_at: str = ""
    updated_at: str = ""
    digest: dict = field(default_factory=dict)             # last profiler Digest.to_dict()
    quirks: list[str] = field(default_factory=list)         # human-curated free text
    reference_distributions: dict = field(default_factory=dict)  # per-col baseline (Phase 6 drift)
    decisions: list[dict] = field(default_factory=list)     # [{ts, decision, rationale}]
    run_ids: list[str] = field(default_factory=list)        # run folder names touching this data
    finding_refs: list[str] = field(default_factory=list)   # short recorded-finding claims
    framing: dict = field(default_factory=dict)             # V3-023: L0Framing.to_dict(), {} if none captured
    last_eval: dict = field(default_factory=dict)            # V3-024: {pr_auc, roc_auc, calibration_flagged, ...} baseline for drift-sweep eval comparison
    data_sha256: str = ""                                     # V3-026: content hash the card was last profiled/consolidated against
    stale: bool = False                                       # V3-026: True once a staleness check finds data_sha256 no longer matches
    stale_reason: str = ""
    # V3-053: the Librarian. This card IS the "book" -- one per dataset,
    # already keyed by card_key. `stories` adds the two levels underneath:
    # a STORY is one span of a specific data_sha256 (a new content hash
    # under the same path opens a new story automatically -- no user
    # action needed, and this is also what makes cross-period confirmation
    # (V3-051 C4) possible: two stories on the same book are directly
    # diffable). A CHAPTER is a declared approach from chapter_taxonomy.py,
    # dedup'd per (story, chapter) so the same approach on UNCHANGED data
    # is refused, but automatically reopens on a new story.
    # Shape: [{story_id, data_sha256, first_seen, row_count,
    #          chapters: {name: {status, page_count, pages: [str, ...],
    #                             redone: [{ts, reason}, ...]}}}]
    stories: list[dict] = field(default_factory=list)
    # 2026-09-19, team knowledge base: the name this dataset has for EVERY
    # analyst (trino:<table> / s3://<bucket>/<key> / sha256:<content>), unlike
    # `key`/`path`, which are this machine's file path.
    dataset_id: str = ""
    # Answers to questions about the data asked after profiling (what a
    # label's classes mean, the entity, the event time, code columns...):
    # [{key, question, answer, by, ts}]. Append-only -- a changed answer is
    # a new entry; the latest per key is current. Shared with the team, so
    # nobody is asked the same question about the same dataset twice.
    clarifications: list[dict] = field(default_factory=list)

    # -- (de)serialisation ----------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetCard":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def new(cls, path: str | Path) -> "DatasetCard":
        resolved = str(Path(path).resolve())
        now = _now()
        return cls(key=card_key(path), path=resolved, created_at=now, updated_at=now)

    # -- mutation (append, never overwrite) ------------------------------------

    def set_digest(self, digest: dict) -> None:
        self.digest = digest
        self.updated_at = _now()

    def add_quirk(self, text: str) -> None:
        text = (text or "").strip()
        if text and text not in self.quirks:
            self.quirks.append(text)
            self.updated_at = _now()

    def add_decision(self, decision: str, rationale: str = "") -> None:
        decision = (decision or "").strip()
        if not decision:
            return
        self.decisions.append({"ts": _now(), "decision": decision,
                               "rationale": (rationale or "").strip()})
        self.updated_at = _now()

    def add_clarification(self, key: str, question: str, answer: str, by: str = "") -> None:
        key, answer = (key or "").strip(), (answer or "").strip()
        if not key or not answer:
            return
        self.clarifications.append({"key": key, "question": (question or "").strip(),
                                    "answer": answer, "by": by, "ts": _now()})
        self.updated_at = _now()

    def current_clarifications(self) -> dict[str, dict]:
        """Latest answer per key (older answers stay in the history)."""
        out: dict[str, dict] = {}
        for c in sorted(self.clarifications or [], key=lambda c: str(c.get("ts", ""))):
            if c.get("key"):
                out[c["key"]] = c
        return out

    def add_run(self, run_id: str) -> None:
        if run_id and run_id not in self.run_ids:
            self.run_ids.append(run_id)
            self.updated_at = _now()

    def set_framing(self, framing_dict: dict) -> None:
        """Record L0 framing (ledgerhound.framing.l0.L0Framing.to_dict()) on
        this dataset card. Same status as add_quirk/add_dependency
        elsewhere in this codebase: a primitive for a trusted caller, not
        an LLM tool in v1."""
        self.framing = dict(framing_dict)
        self.updated_at = _now()

    def add_finding(self, claim: str) -> None:
        claim = (claim or "").strip()[:200]
        if claim and claim not in self.finding_refs:
            self.finding_refs.append(claim)
            self.updated_at = _now()

    def set_reference_distribution(self, column: str, stats: dict) -> None:
        self.reference_distributions[column] = stats
        self.updated_at = _now()

    def set_data_hash(self, sha256: str) -> None:
        """Record the content hash (ledgerhound.runtime.events.sha256_file)
        this card was last profiled/consolidated against, and clear any
        prior staleness flag -- a fresh hash capture means "this is the
        current baseline," not "still stale." A primitive for a trusted
        caller (ledgerhound.memory.consolidation), same status as
        set_reference_distribution/set_framing."""
        self.data_sha256 = sha256
        self.stale = False
        self.stale_reason = ""
        self.updated_at = _now()

    def flag_stale(self, reason: str) -> None:
        self.stale = True
        self.stale_reason = reason
        self.updated_at = _now()

    # -- Librarian: story / chapter (V3-053) -----------------------------

    def find_story(self, data_sha256: str) -> dict | None:
        """Look up a story by content hash across ALL stored stories, not
        just the most recent one. Load-bearing: if this only checked the
        latest story, alternating between two data versions (e.g. April,
        then May, then April again) would silently stop deduping against
        the April story the moment May was loaded, because the "current"
        hash would no longer match it."""
        for s in self.stories:
            if s.get("data_sha256") == data_sha256:
                return s
        return None

    def open_story(self, data_sha256: str, *, row_count: int = 0) -> dict:
        """Return the existing story for this content hash, or start a new
        one. A new content hash under the same book (same resolved path)
        opens a new story automatically -- no user action needed, per the
        agreed dedup rule (auto-allow re-running a chapter when the data
        has genuinely changed)."""
        story = self.find_story(data_sha256)
        if story is not None:
            return story
        story = {
            "story_id": f"story_{len(self.stories) + 1}",
            "data_sha256": data_sha256,
            "first_seen": _now(),
            "row_count": row_count,
            "chapters": {},
        }
        self.stories.append(story)
        self.updated_at = _now()
        return story

    def chapter_status(self, data_sha256: str, chapter: str) -> dict | None:
        story = self.find_story(data_sha256)
        if story is None:
            return None
        return story.get("chapters", {}).get(chapter)

    def open_chapter(self, data_sha256: str, chapter: str, *,
                     row_count: int = 0) -> tuple[bool, dict]:
        """Try to open `chapter` against the story for `data_sha256`.

        Returns (allowed, info). `allowed` is False only when this EXACT
        (story, chapter) pair already has pages -- i.e. this chapter was
        already run against this exact data version. A different content
        hash always opens (a new story is created if needed), which is the
        agreed "auto-allow when the data changed" rule and doubles as the
        mechanism for cross-period confirmation (two stories on one book,
        directly diffable).
        """
        story = self.open_story(data_sha256, row_count=row_count)
        chapters = story.setdefault("chapters", {})
        existing = chapters.get(chapter)
        if existing and existing.get("page_count", 0) > 0:
            return False, existing
        chapters[chapter] = existing or {
            "status": "open", "page_count": 0, "pages": [], "redone": [],
        }
        chapters[chapter]["status"] = "open"
        self.updated_at = _now()
        return True, chapters[chapter]

    def redo_chapter(self, data_sha256: str, chapter: str, reason: str) -> dict:
        """Explicit override: proceed even though (story, chapter) already
        has pages and the data has NOT changed. Always recorded, never
        silent -- the caller's event log should also log a `chapter_redo`
        event; this just keeps the card's own record honest."""
        story = self.open_story(data_sha256)
        chapters = story.setdefault("chapters", {})
        entry = chapters.setdefault(chapter, {
            "status": "open", "page_count": 0, "pages": [], "redone": [],
        })
        entry.setdefault("redone", []).append({"ts": _now(), "reason": reason})
        entry["status"] = "open"
        self.updated_at = _now()
        return entry

    def add_chapter_page(self, data_sha256: str, chapter: str,
                         page_summary: str) -> int:
        """Append one page (append-only, never rewritten). Returns the new
        page number (1-indexed)."""
        story = self.open_story(data_sha256)
        entry = story.setdefault("chapters", {}).setdefault(chapter, {
            "status": "open", "page_count": 0, "pages": [], "redone": [],
        })
        entry.setdefault("pages", []).append(page_summary.strip()[:500])
        entry["page_count"] = len(entry["pages"])
        entry["status"] = "has_pages"
        self.updated_at = _now()
        return entry["page_count"]

    def add_percentile_snapshot(self, data_sha256: str, chapter: str,
                                feature: str, percentiles: dict) -> None:
        """V3-055: record a STRUCTURED percentile snapshot (flat
        {label: value}, e.g. ``percentile_table(...)['a']``) for `feature`
        under this story/chapter -- separate from the prose `pages` list
        on purpose, so the free-text progress notes and the numeric record
        a stability check can actually compare against don't get tangled
        into one schema. Append-only, most-recent-wins on lookup."""
        story = self.open_story(data_sha256)
        entry = story.setdefault("chapters", {}).setdefault(chapter, {
            "status": "open", "page_count": 0, "pages": [], "redone": [],
            "snapshots": [],
        })
        entry.setdefault("snapshots", []).append({
            "feature": feature, "percentiles": dict(percentiles), "ts": _now(),
        })
        self.updated_at = _now()

    def find_percentile_snapshot(self, data_sha256: str, chapter: str,
                                 feature: str) -> dict | None:
        """Most recent snapshot for `feature` under (story for
        data_sha256, chapter), or None if nothing was ever recorded."""
        story = self.find_story(data_sha256)
        if story is None:
            return None
        snapshots = story.get("chapters", {}).get(chapter, {}).get("snapshots", [])
        for snap in reversed(snapshots):
            if snap.get("feature") == feature:
                return snap
        return None

    def library_summary(self) -> str:
        """One compact block for `/library`: story/chapter counts, no page
        bodies -- this is the index, not the content."""
        lines = [f"{len(self.stories)} stor{'y' if len(self.stories) == 1 else 'ies'}"]
        for s in self.stories:
            n_ch = len(s.get("chapters", {}))
            n_pages = sum(c.get("page_count", 0) for c in s.get("chapters", {}).values())
            lines.append(f"  - {s['story_id']} ({s.get('data_sha256', '?')[:10]}...): "
                        f"{n_ch} chapter(s), {n_pages} page(s)")
        return "\n".join(lines)

    def set_last_eval(self, eval_scores: dict) -> None:
        """Record the eval scores (pr_auc/roc_auc/calibration_flagged/...)
        a drift sweep (V3-024) diffs the next run's scores against. A
        primitive for a trusted caller (ledgerhound.monitor.drift_sentinel),
        same status as set_reference_distribution/set_framing."""
        self.last_eval = dict(eval_scores)
        self.updated_at = _now()

    # -- presentation -----------------------------------------------------------

    def summary_block(self, max_chars: int = 1800) -> str:
        """Compact block for pinning into LLM context on load."""
        lines = [f"DATASET CARD -- seen before, {len(self.run_ids)} prior "
                 f"session(s) on this exact file:"]
        if self.stale:
            lines.append(f"  ** STALE: {self.stale_reason} **")
        if self.digest:
            ov = self.digest.get("overview", {})
            dq = self.digest.get("data_quality", {})
            if ov:
                lines.append(f"  last profiled: {ov.get('rows', '?')} rows x "
                            f"{ov.get('columns', '?')} cols, "
                            f"target={ov.get('target', 'none')}, "
                            f"quality={dq.get('score', '?')}/100 "
                            f"(do not re-profile -- reuse this)")
        if self.quirks:
            lines.append("  known quirks:")
            for q in self.quirks[-8:]:
                lines.append(f"    - {q}")
        if self.decisions:
            lines.append("  prior decisions:")
            for d in self.decisions[-5:]:
                extra = f" ({d['rationale']})" if d.get("rationale") else ""
                lines.append(f"    - {d['decision']}{extra}")
        if self.finding_refs:
            lines.append("  prior findings:")
            for f in self.finding_refs[-8:]:
                lines.append(f"    - {f}")
        block = "\n".join(lines)
        return block[:max_chars]

    def to_markdown(self) -> str:
        lines = [f"# Dataset Card -- {Path(self.path).name}", ""]
        lines += [f"- Path: `{self.path}`",
                  f"- Key: {self.key}",
                  f"- Created: {self.created_at}",
                  f"- Updated: {self.updated_at}",
                  f"- Sessions: {len(self.run_ids)}", ""]
        if self.stale:
            lines += [f"**STALE**: {self.stale_reason}", ""]
        if self.digest:
            ov = self.digest.get("overview", {})
            dq = self.digest.get("data_quality", {})
            lines += ["## Last known digest", "",
                      f"- Rows: {ov.get('rows', '?')}",
                      f"- Columns: {ov.get('columns', '?')}",
                      f"- Target: {ov.get('target', 'none')}",
                      f"- Quality score: {dq.get('score', '?')}/100", ""]
        if self.clarifications:
            lines += ["## What the data means (answered by the team)", ""]
            for k, c in self.current_clarifications().items():
                who = f" -- {c.get('by')}" if c.get("by") else ""
                lines.append(f"- **{k}**: {c.get('answer')}{who}")
            lines.append("")
        if self.quirks:
            lines += ["## Known quirks", ""]
            lines += [f"- {q}" for q in self.quirks] + [""]
        if self.decisions:
            lines += ["## Decision history", ""]
            for d in self.decisions:
                extra = f" -- {d['rationale']}" if d.get("rationale") else ""
                lines.append(f"- [{d['ts']}] {d['decision']}{extra}")
            lines.append("")
        if self.reference_distributions:
            lines += ["## Reference distributions (drift baselines)", ""]
            for col, stats in self.reference_distributions.items():
                lines.append(f"- `{col}`: {stats}")
            lines.append("")
        if self.finding_refs:
            lines += ["## Findings", ""]
            lines += [f"- {f}" for f in self.finding_refs] + [""]
        if self.run_ids:
            lines += ["## Runs", ""]
            lines += [f"- {r}" for r in self.run_ids] + [""]
        return "\n".join(lines)


class DatasetCardStore:
    """Load/save cards under a project's ``.ledgerhound/dataset_cards``."""

    def __init__(self, project):
        self.project = project
        self.dir = Path(project.dotdir) / CARDS_DIR

    def _paths(self, key: str) -> tuple[Path, Path]:
        return self.dir / f"{key}.json", self.dir / f"{key}.md"

    def load(self, path: str | Path) -> DatasetCard | None:
        json_path, _ = self._paths(card_key(path))
        if not json_path.exists():
            return None
        try:
            text = _retry_io(lambda: json_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except OSError:
            # Still locked by another session after the retries: the card
            # is healthy, just busy. Never quarantine it for that -- the
            # caller starts a fresh copy and merge-on-save folds it back in.
            return None
        try:
            return DatasetCard.from_dict(json.loads(text))
        except Exception:
            # K-07: an unreadable card used to return None here, and the
            # caller then created a NEW card and saved it over this file --
            # silently erasing every story, chapter and decision recorded
            # on this dataset. Move the damaged file aside first so it can
            # be recovered by hand; the new card starts fresh next to it.
            self._quarantine(json_path)
            return None

    @staticmethod
    def _quarantine(json_path: Path) -> None:
        try:
            import uuid
            # Second-resolution alone collides when two loads hit the same
            # corrupt card within a second -- the second replace() would
            # overwrite the first backup.
            stamp = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                     + "-" + uuid.uuid4().hex[:6])
            json_path.replace(json_path.with_name(
                f"{json_path.stem}.corrupt-{stamp}.json"))
        except Exception:  # noqa: BLE001 -- best effort; never block a load
            pass

    def load_or_create(self, path: str | Path) -> DatasetCard:
        return self.load(path) or DatasetCard.new(path)

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        """Write to a temp file in the same folder, then rename over the
        target -- a crash mid-write leaves the old card intact instead of a
        half-written JSON (which K-07 showed costs the whole history)."""
        import uuid
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            # Windows refuses to rename over a file another session has open
            # for reading (WinError 5/32) -- momentary, so retry briefly
            # instead of dropping this save.
            _retry_io(lambda: tmp.replace(path))
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass

    def save(self, card: DatasetCard) -> Path:
        """Merge-on-save. Two sessions on the same dataset each hold their own
        copy of the card; a plain overwrite let the second save erase the
        first one's new pages, decisions and findings. Under a short lock,
        the card on disk is re-read and merged with this one (cards only
        ever grow -- see merge_cards), written atomically, and `card` is
        updated in place so this session also sees the other's additions."""
        self.dir.mkdir(parents=True, exist_ok=True)
        json_path, md_path = self._paths(card.key)
        with _card_lock(self.dir / f"{card.key}.lock"):
            if json_path.exists():
                try:
                    text = _retry_io(lambda: json_path.read_text(encoding="utf-8"))
                except FileNotFoundError:
                    text = None
                # Any other OSError propagates: writing without having read
                # the card on disk would erase the other session's work.
                if text is not None:
                    try:
                        disk = DatasetCard.from_dict(json.loads(text))
                    except Exception:  # noqa: BLE001 -- genuinely corrupt
                        disk = None
                        self._quarantine(json_path)   # keep it for recovery
                    if disk is not None:
                        merged = merge_cards(disk, card)
                        card.__dict__.update(merged.__dict__)
            self._atomic_write(json_path, json.dumps(card.to_dict(), indent=2))
            self._atomic_write(md_path, card.to_markdown())
        return md_path


# -- concurrent-session safety -------------------------------------------------

def _retry_io(fn, *, attempts: int = 40, delay: float = 0.05):
    """Run a file operation, retrying Windows sharing violations.

    A read or rename that races another process's open handle fails with
    PermissionError (WinError 5/32) for a few milliseconds. That is not a
    real error, so it is retried for up to about 2 s. FileNotFoundError and
    every other error are raised at once."""
    import time
    for i in range(attempts):
        try:
            return fn()
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(delay)


class _card_lock:
    """Tiny cross-platform lock file (O_CREAT|O_EXCL). Held only for one
    re-read + merge + write. A lock older than STALE_S is from a crashed
    session and is broken; after WAIT_S the write goes ahead unlocked --
    a card update is never worth hanging a session over."""

    WAIT_S = 10.0
    STALE_S = 30.0

    def __init__(self, path: Path):
        self.path = path
        self.held = False

    def __enter__(self):
        import os
        import time
        deadline = time.monotonic() + self.WAIT_S
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                self.held = True
                return self
            except (FileExistsError, PermissionError):
                # PermissionError: on Windows a lock file that its holder is
                # deleting ("delete pending") refuses O_EXCL with access
                # denied -- it is busy, not unlockable, so wait like any
                # other held lock rather than writing unlocked.
                try:
                    if time.time() - self.path.stat().st_mtime > self.STALE_S:
                        self.path.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    return self
                time.sleep(0.05)
            except OSError:
                return self

    def __exit__(self, *exc):
        if self.held:
            try:
                self.path.unlink(missing_ok=True)
            except OSError:
                pass
        return False


def _union(a: list, b: list, key=lambda x: json.dumps(x, sort_keys=True, default=str)) -> list:
    seen, out = set(), []
    for item in list(a or []) + list(b or []):
        k = key(item)
        if k not in seen:
            seen.add(k)
            out.append(item)
    return out


def _merge_chapter(a: dict, b: dict) -> dict:
    out = dict(a)
    out.update({k: v for k, v in b.items()
                if k not in ("pages", "redone", "snapshots", "page_count", "status")})
    pages = _union(a.get("pages", []), b.get("pages", []), key=str)
    out["pages"] = pages
    out["page_count"] = len(pages)
    out["redone"] = _union(a.get("redone", []), b.get("redone", []))
    if a.get("snapshots") or b.get("snapshots"):
        out["snapshots"] = _union(a.get("snapshots", []), b.get("snapshots", []))
    out["status"] = "has_pages" if pages else (b.get("status") or a.get("status") or "open")
    return out


def merge_cards(disk: "DatasetCard", mine: "DatasetCard") -> "DatasetCard":
    """Combine two copies of one card. Cards are append-only (stories,
    chapter pages, decisions, findings, runs, quirks only ever grow), so the
    merge is a union; single-valued fields keep `mine` (the session saving
    now) unless it has nothing there."""
    d, m = disk.to_dict(), mine.to_dict()
    out = dict(d)
    for k in ("quirks", "run_ids", "finding_refs"):
        out[k] = _union(d.get(k, []), m.get(k, []), key=str)
    out["decisions"] = _union(d.get("decisions", []), m.get("decisions", []),
                              key=lambda x: (x.get("ts"), x.get("decision")))
    out["clarifications"] = _union(
        d.get("clarifications", []), m.get("clarifications", []),
        key=lambda x: (x.get("key"), x.get("ts"), x.get("by"), x.get("answer")))
    out["dataset_id"] = m.get("dataset_id") or d.get("dataset_id") or ""
    out["reference_distributions"] = {**d.get("reference_distributions", {}),
                                      **m.get("reference_distributions", {})}
    for k in ("digest", "framing", "last_eval"):
        out[k] = m.get(k) or d.get(k) or {}
    for k in ("data_sha256", "stale", "stale_reason", "path", "key", "created_at"):
        out[k] = m.get(k) if m.get(k) not in (None, "") else d.get(k)
    out["updated_at"] = max(str(d.get("updated_at") or ""), str(m.get("updated_at") or ""))

    stories = [dict(s) for s in d.get("stories", [])]
    by_sha = {s.get("data_sha256"): s for s in stories}
    taken = {s.get("story_id") for s in stories}
    for s in m.get("stories", []):
        sha = s.get("data_sha256")
        if sha in by_sha:
            base = by_sha[sha]
            chapters = dict(base.get("chapters", {}))
            for name, ch in (s.get("chapters") or {}).items():
                chapters[name] = _merge_chapter(chapters.get(name, {}), ch)
            base["chapters"] = chapters
        else:
            new = dict(s)
            if new.get("story_id") in taken:        # created in parallel
                n = len(stories) + 1
                while f"story_{n}" in taken:
                    n += 1
                new["story_id"] = f"story_{n}"
            taken.add(new.get("story_id"))
            stories.append(new)
            by_sha[sha] = new
    out["stories"] = stories
    return DatasetCard.from_dict(out)
