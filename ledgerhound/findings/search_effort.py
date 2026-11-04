"""
ledgerhound.findings.search_effort
===============================

V5-002. How hard did we look before we found this?

A finding recorded after three comparisons and a finding recorded after two
hundred are not equally believable, and until now nothing in the ledger could
tell them apart. `record_finding` already refuses restatements and catches
predicate mismatches -- both checks about *what* is claimed. Neither asks how
many attempts it took to arrive at it.

That question matters more here than almost anywhere. Fraud runs near 1 in
100,000 (README). When the real signal is that rare, the pool of
plausible-looking noise is enormous, and a search long enough will always
surface something. The forty-seven comparisons that showed nothing are never
written down; only the three that looked striking are.

**The chapter is the family.** A chapter is already a *declared approach* from
a fixed taxonomy (`knowledge/chapter_taxonomy.py`), which is exactly the
boundary a family of tests needs: the analyst says up front what they are
looking for, and every attempt inside that declaration belongs to one family.
Free-text grouping would be gameable; the taxonomy is not.

What this does NOT do
---------------------
It cannot see tests run outside `record_finding`. An analyst who computes
thirty comparisons in `run_python` and only ever records the winner is outside
the accounting; the cell counter narrows this (thirty cells against one
finding is visible) but does not close it. No multiplicity method can correct
for tests it was never told about, and pretending otherwise would be worse
than the gap.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ledgerhound.analysis.multiplicity import adjust as adjust_pvalues

# Status a recorded finding carries out of this module.
ESTABLISHED = "established"      # survived correction within its family
EXPLORATORY = "exploratory"      # did not -- recorded, but not as fact
UNQUANTIFIED = "unquantified"    # no p-value supplied; effort recorded only

# Attempts in one chapter beyond which the search is worth remarking on even
# when no p-value was supplied. Not a refusal: the honest response to a long
# search is to say how long it was, not to forbid it.
EFFORT_NOTE_THRESHOLD = 8

# Analysis cells per recorded finding beyond which the ratio itself is the
# signal -- many computations, one written-down result.
CELLS_PER_FINDING_THRESHOLD = 15


@dataclass
class FamilyRecord:
    """One chapter's accumulated search."""
    chapter: str
    n_attempted: int = 0        # record_finding calls, refusals included
    n_recorded: int = 0
    n_refused: int = 0
    n_cells: int = 0            # run_python executions while this was open
    pvalues: dict = field(default_factory=dict)   # claim key -> raw p
    # Status as last assessed, per claim. Kept because a family GROWS: a claim
    # established when it was the only test in the chapter may stop being
    # established once thirty more are run, and the ledger has to be able to
    # say so rather than leaving the first verdict standing forever.
    statuses: dict = field(default_factory=dict)  # claim key -> status

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Assessment:
    status: str
    family_size: int            # tests in this family carrying a p-value
    n_attempted: int
    n_cells: int
    qvalue: float | None = None
    pvalue: float | None = None
    # Claims established EARLIER in this chapter that this new test knocked
    # back. Surfaced, never silent: a demotion the analyst does not see is a
    # correction that did not happen.
    demoted: list = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class SearchEffortLedger:
    """Per-chapter test families for one run. Never raises: a failure to
    account for search effort must not cost the analyst their finding."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.families: dict[str, FamilyRecord] = {}
        self._load()

    # -- persistence -------------------------------------------------------
    def _load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            for chapter, d in (raw or {}).items():
                self.families[chapter] = FamilyRecord(**d)
        except Exception:  # noqa: BLE001
            self.families = {}

    def save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps({c: f.to_dict() for c, f in self.families.items()},
                           indent=2),
                encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    # -- accounting --------------------------------------------------------
    def _family(self, chapter: str) -> FamilyRecord:
        key = chapter or "(no chapter)"
        if key not in self.families:
            self.families[key] = FamilyRecord(chapter=key)
        return self.families[key]

    def note_cell(self, chapter: str) -> None:
        """One analysis cell ran. Cheap proxy for tests we cannot see."""
        self._family(chapter).n_cells += 1
        self.save()

    def note_attempt(self, chapter: str) -> int:
        f = self._family(chapter)
        f.n_attempted += 1
        self.save()
        return f.n_attempted

    def note_refused(self, chapter: str) -> None:
        self._family(chapter).n_refused += 1
        self.save()

    # -- the judgement -----------------------------------------------------
    def assess(self, chapter: str, claim: str,
               pvalue: float | None = None) -> Assessment:
        """Record this finding against its family and say what it may be
        called. Call AFTER `note_attempt`.

        Re-corrects the whole chapter each time, so an earlier claim can be
        demoted by a later test. `Assessment.demoted` lists any that were.
        """
        f = self._family(chapter)
        f.n_recorded += 1

        demoted: list[str] = []
        if pvalue is not None:
            key = claim[:200]
            f.pvalues[key] = float(pvalue)
            # Re-correct the WHOLE family, not just this claim. A family grows,
            # and the earlier verdicts were reached against a smaller one: a
            # claim that was the only test in its chapter is established on a
            # family of one, and stays established forever unless something
            # revisits it. That is the order-dependence this re-assessment
            # exists to remove -- the 200th look must be able to demote the
            # 1st, or "how hard did we look" never actually binds.
            mult = adjust_pvalues(f.pvalues)
            for other in f.pvalues:
                was = f.statuses.get(other)
                now = ESTABLISHED if mult.is_rejected(other) else EXPLORATORY
                if was == ESTABLISHED and now != ESTABLISHED and other != key:
                    demoted.append(other)
                f.statuses[other] = now
            q = mult.q(key)
            status = f.statuses[key]
        else:
            q, status = None, UNQUANTIFIED
            f.statuses[claim[:200]] = status

        self.save()
        return Assessment(
            status=status, family_size=len(f.pvalues),
            n_attempted=f.n_attempted, n_cells=f.n_cells,
            qvalue=q, pvalue=pvalue, demoted=demoted,
            note=self._note(f, status, q, pvalue, demoted),
        )

    def statuses(self, chapter: str) -> dict:
        """Current status of every p-valued claim in a chapter, after all
        corrections applied so far."""
        return dict(self._family(chapter).statuses)

    def _note(self, f: FamilyRecord, status: str,
              q: float | None, p: float | None,
              demoted: list | None = None) -> str:
        lines: list[str] = []

        for claim in (demoted or []):
            lines.append(
                f"DEMOTED by this test: \"{claim[:80]}\" was recorded as "
                f"established earlier in this chapter and no longer survives "
                f"correction now that {len(f.pvalues)} tests have been "
                f"declared here. Restate it as exploratory wherever it has "
                f"already been reported.")

        if status == EXPLORATORY:
            lines.append(
                f"MARKED EXPLORATORY -- not established. p={p:.4g} does not "
                f"survive correction across the {len(f.pvalues)} tests "
                f"declared in chapter '{f.chapter}' (adjusted q={q:.4g}). "
                f"With that many looks, a result this strong is expected by "
                f"chance. Report it as a lead, not a fact; confirm it on a "
                f"different period or sample before it becomes a feature.")
        elif status == ESTABLISHED:
            lines.append(
                f"Survives correction across {len(f.pvalues)} tests in this "
                f"chapter (p={p:.4g}, q={q:.4g}).")

        if f.n_attempted >= EFFORT_NOTE_THRESHOLD:
            lines.append(
                f"Search effort: this is attempt {f.n_attempted} in chapter "
                f"'{f.chapter}' ({f.n_refused} refused as restatements). "
                f"State the number of comparisons made alongside the finding "
                f"-- a reader cannot judge it otherwise.")

        if (f.n_recorded and f.n_cells
                >= CELLS_PER_FINDING_THRESHOLD * f.n_recorded):
            lines.append(
                f"{f.n_cells} analysis cells have run in this chapter for "
                f"{f.n_recorded} recorded finding(s). Most of what was "
                f"computed is not written down, so the tests this correction "
                f"can see are a fraction of the tests actually made.")

        return " ".join(lines)

    def summary(self) -> str:
        if not self.families:
            return "No search effort recorded."
        lines = ["Search effort by chapter:"]
        for name, f in sorted(self.families.items()):
            lines.append(
                f"  {name}: {f.n_recorded} recorded / {f.n_attempted} attempted"
                f" ({f.n_refused} refused), {f.n_cells} cells,"
                f" {len(f.pvalues)} declared p-values")
        return "\n".join(lines)
