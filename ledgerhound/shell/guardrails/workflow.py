"""Gates on the analysis workflow: chapters and methodology phases."""

from __future__ import annotations


class ChapterDedupeGate:
    """The same chapter on UNCHANGED data (same content hash as when it last
    produced pages) is refused, showing what was already found. The dedupe
    state lives on the dataset card (DatasetCard.open_chapter); this class
    owns the refusal."""

    @staticmethod
    def refusal(chapter: str, info: dict) -> str:
        pages = info.get("pages", [])
        found = "\n".join(f"  - {p}" for p in pages[-5:])
        return (
            f"REFUSED: chapter '{chapter}' already has {info.get('page_count', 0)} "
            f"page(s) recorded against this EXACT data version.\n"
            f"What was already found:\n{found}\n\n"
            "If you have a real reason to redo it anyway (the first "
            "pass was buggy, wrong assumption, etc.), call "
            "redo_chapter(chapter, reason) -- that is recorded, not "
            "silent. If the data has genuinely changed since, re-load "
            "it first; a new content hash opens a new story "
            "automatically and this chapter reopens on its own.")


class PhaseReportGate:
    """V3-045: a methodology phase does not close without an artifact.
    Advancing a phase that saved no report is refused unless forced (a
    forced close is allowed and recorded; no stub report is written, since
    a file nobody wrote would claim coverage that does not exist)."""

    @staticmethod
    def check(phase_id: str, reports_by_phase: dict, force: bool) -> str:
        if force or reports_by_phase.get(phase_id):
            return ""
        return (
            f"REFUSED: phase '{phase_id}' has no saved report.\n"
            "Every phase leaves an artifact -- call save_report with this "
            "phase's findings before advancing, so the run folder shows "
            "what each phase actually produced.\n"
            "If this phase genuinely has nothing to report, "
            "advance_phase(force=True) moves on and records the gap.")
