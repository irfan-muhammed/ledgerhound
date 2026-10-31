"""
sentinel.memory.fact_memory
===========================

FactMemory -- the structured numeric spine of long-term "analyst instinct".

While LongitudinalMemory gives *associative* recall ("have we seen a pattern
like this?"), FactMemory gives *quantitative* recall: it accumulates the
grounded `metrics` every run computes into per-(dataset, fact) time series,
maintains a rolling baseline + spread, and classifies each new observation as:

  NEW         first time we've seen this fact
  CONFIRMING  consistent with history -> raises confidence, tightens baseline
  DRIFT       deviates beyond tolerance -> flagged anomaly (the gold signal)

This is exactly how a long-tenured analyst's instinct works: repeated exposure
builds an expectation; a deviation from that expectation is what fires the gut.

Trust guarantees (this is memory, so correctness is non-negotiable):
* Only numeric facts that carry their `computation` are stored, so any fact can
  be RE-VERIFIED by re-running it against current data -- memory that can
  recompute itself cannot rot into hallucination.
* Confidence only grows with INDEPENDENT confirmations (distinct run_ids).
* Drift is detected statistically (z-score + relative tolerance), never by an
  LLM's opinion.

Storage: one JSON file per memory root (human-readable, diff-able, dependency
free). Keyed by ``dataset::fact_name``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

Classification = Literal["NEW", "CONFIRMING", "DRIFT"]
Status = Literal["new", "stable", "drifting", "contradicted"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Observation:
    value: float
    run_id: str
    as_of: str


@dataclass
class FactRecord:
    dataset: str
    name: str
    computation: str = ""          # re-runnable; enables re-verification
    unit: str = ""
    observations: list[Observation] = field(default_factory=list)
    status: Status = "new"
    first_seen: str = ""
    last_confirmed: str = ""
    last_verified: str = ""        # last time recomputation reproduced the value

    # -- derived statistics --
    @property
    def key(self) -> str:
        return f"{self.dataset}::{self.name}"

    @property
    def n(self) -> int:
        return len(self.observations)

    @property
    def values(self) -> list[float]:
        return [o.value for o in self.observations]

    @property
    def baseline(self) -> float:
        v = self.values
        return sum(v) / len(v) if v else 0.0

    @property
    def stdev(self) -> float:
        v = self.values
        if len(v) < 2:
            return 0.0
        m = sum(v) / len(v)
        return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))

    @property
    def confidence(self) -> float:
        """Grows with the number of DISTINCT runs that confirmed this fact."""
        distinct = len({o.run_id for o in self.observations})
        return round(1.0 - 1.0 / (distinct + 1), 3)   # 1->.5, 3->.75, 9->.9

    def zscore(self, value: float) -> float:
        s = self.stdev
        if s <= 0:
            return 0.0
        return (value - self.baseline) / s

    def to_dict(self) -> dict:
        d = asdict(self)
        d["baseline"] = round(self.baseline, 6)
        d["stdev"] = round(self.stdev, 6)
        d["confidence"] = self.confidence
        d["n"] = self.n
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FactRecord":
        obs = [Observation(**o) for o in d.get("observations", [])]
        return cls(
            dataset=d["dataset"], name=d["name"],
            computation=d.get("computation", ""), unit=d.get("unit", ""),
            observations=obs, status=d.get("status", "new"),
            first_seen=d.get("first_seen", ""), last_confirmed=d.get("last_confirmed", ""),
            last_verified=d.get("last_verified", ""),
        )


@dataclass
class Consolidation:
    """Result of observing one fact in a run."""
    key: str
    classification: Classification
    value: float
    baseline: float
    stdev: float
    zscore: float
    confidence: float
    is_anomaly: bool
    message: str


class FactMemory:
    """Per-dataset numeric fact store with consolidation + drift detection."""

    def __init__(self, store_path: str | Path,
                 drift_sigma: float = 3.0, rel_tol: float = 0.10):
        self.path = Path(store_path)
        self.drift_sigma = drift_sigma   # z beyond this => drift
        self.rel_tol = rel_tol           # OR relative change beyond this => drift
        self._facts: dict[str, FactRecord] = {}
        self._load()

    # -- consolidation ----------------------------------------------------

    def observe(self, dataset: str, name: str, value: float,
                run_id: str, computation: str = "", unit: str = "",
                as_of: str | None = None) -> Consolidation:
        """Record one observation and classify it (NEW/CONFIRMING/DRIFT)."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"FactMemory only stores numeric facts; got {value!r}")
        as_of = as_of or _now()
        key = f"{dataset}::{name}"
        rec = self._facts.get(key)

        if rec is None:
            rec = FactRecord(dataset=dataset, name=name, computation=computation,
                             unit=unit, first_seen=as_of)
            rec.observations.append(Observation(value, run_id, as_of))
            rec.last_confirmed = as_of
            rec.status = "new"
            self._facts[key] = rec
            return Consolidation(key, "NEW", value, value, 0.0, 0.0,
                                 rec.confidence, False,
                                 f"New fact '{name}' = {value} (first observation)")

        # classify against EXISTING history (before adding this obs)
        baseline = rec.baseline
        stdev = rec.stdev
        z = rec.zscore(value)
        rel = abs(value - baseline) / (abs(baseline) if abs(baseline) > 1e-12 else 1.0)
        is_drift = (abs(z) >= self.drift_sigma) if stdev > 0 else (rel > self.rel_tol)

        # append observation, update provenance
        rec.observations.append(Observation(value, run_id, as_of))
        if computation and not rec.computation:
            rec.computation = computation
        if unit and not rec.unit:
            rec.unit = unit

        if is_drift:
            rec.status = "drifting"
            rec.last_confirmed = rec.last_confirmed  # not a confirmation
            msg = (f"DRIFT: '{name}' = {value} vs baseline "
                   f"{baseline:.4g} (z={z:.1f}, {rel:.0%} change)")
            cls: Classification = "DRIFT"
            anomaly = True
        else:
            rec.last_confirmed = as_of
            if rec.n >= 3:
                rec.status = "stable"
            msg = (f"CONFIRMING: '{name}' = {value} consistent with baseline "
                   f"{baseline:.4g} (z={z:.1f})")
            cls = "CONFIRMING"
            anomaly = False

        return Consolidation(key, cls, value, rec.baseline, rec.stdev, z,
                             rec.confidence, anomaly, msg)

    def consolidate_run(self, dataset: str, run_id: str,
                        metrics: dict, computations: dict | None = None,
                        ) -> list[Consolidation]:
        """Consolidate a whole run's grounded `metrics` dict at once.
        Returns the consolidation result for each numeric metric."""
        computations = computations or {}
        out = []
        for name, value in (metrics or {}).items():
            try:
                fv = float(value)
            except (TypeError, ValueError):
                continue   # only numeric facts enter durable memory
            out.append(self.observe(dataset, name, fv, run_id,
                                    computation=computations.get(name, "")))
        return out

    # -- re-verification (memory that can recompute itself cannot rot) -----

    def reverify(self, dataset: str, df) -> list[dict]:
        """Re-run each stored fact's `computation` against current data.

        Facts that still reproduce their baseline are marked verified; facts
        that no longer reproduce are demoted to 'contradicted'. This makes the
        trust guarantee active: a stale fact cannot keep masquerading as truth.

        Returns one report dict per fact with a `computation`.
        """
        results = []
        for rec in self.facts_for(dataset):
            if not rec.computation:
                continue
            try:
                ns = {"df": df}
                import pandas as pd, numpy as np  # noqa
                ns.update({"pd": pd, "np": np})
                exec(rec.computation, ns)          # trusted stored computation
                val = float(ns.get("result"))
            except Exception as e:
                results.append({"name": rec.name, "reproduced": None,
                                "error": str(e)[:120]})
                continue
            baseline = rec.baseline
            if rec.stdev > 0:
                ok = abs((val - baseline) / rec.stdev) < self.drift_sigma
            else:
                rel = abs(val - baseline) / (abs(baseline) if abs(baseline) > 1e-12 else 1.0)
                ok = rel <= self.rel_tol
            rec.last_verified = _now()
            if ok:
                # don't flip a drifting fact back to stable on a single recheck
                if rec.status == "contradicted":
                    rec.status = "stable" if rec.n >= 3 else "new"
            else:
                rec.status = "contradicted"
            results.append({"name": rec.name, "baseline": baseline,
                            "recomputed": val, "reproduced": ok})
        return results

    def promote(self, min_confidence: float = 0.75, min_obs: int = 3) -> list[str]:
        """Mark recurring, consistent facts as 'stable' (durable truths).
        Returns the keys promoted."""
        promoted = []
        for rec in self._facts.values():
            if (rec.status not in ("drifting", "contradicted")
                    and rec.n >= min_obs and rec.confidence >= min_confidence):
                if rec.status != "stable":
                    rec.status = "stable"
                    promoted.append(rec.key)
        return promoted

    def merge_record(self, rec: "FactRecord") -> "FactRecord":
        """Merge a foreign FactRecord into this store (union observations).

        Used by the Librarian to promote a fact from the private tier into the
        shared tier without losing observation provenance. Idempotent: merging
        the same record twice adds no duplicate observations. Status is left for
        a subsequent promote()/reverify() to (re)derive.
        """
        existing = self._facts.get(rec.key)
        if existing is None:
            self._facts[rec.key] = FactRecord.from_dict(rec.to_dict())
            return self._facts[rec.key]
        seen = {(o.run_id, o.value) for o in existing.observations}
        for o in rec.observations:
            if (o.run_id, o.value) not in seen:
                existing.observations.append(Observation(o.value, o.run_id, o.as_of))
        if rec.computation and not existing.computation:
            existing.computation = rec.computation
        if rec.unit and not existing.unit:
            existing.unit = rec.unit
        return existing

    # -- retrieval --------------------------------------------------------

    def get(self, dataset: str, name: str) -> FactRecord | None:
        return self._facts.get(f"{dataset}::{name}")

    def facts_for(self, dataset: str) -> list[FactRecord]:
        return [r for r in self._facts.values() if r.dataset == dataset]

    def anomalies(self, dataset: str) -> list[FactRecord]:
        return [r for r in self.facts_for(dataset) if r.status == "drifting"]

    def baseline_block(self, dataset: str, max_facts: int = 20) -> str:
        """Prompt-injectable block of historical baselines = the 'instinct'.

        e.g.  fraud_rate: historically 0.024 +/- 0.003 over 18 runs (stable)
        """
        recs = sorted(self.facts_for(dataset), key=lambda r: -r.n)
        recs = [r for r in recs if r.n >= 2][:max_facts]
        if not recs:
            return ""
        lines = ["=== HISTORICAL BASELINES (data memory) ==="]
        for r in recs:
            spread = f" +/- {r.stdev:.4g}" if r.stdev > 0 else ""
            tag = ""
            if r.status == "drifting":
                tag = " [DRIFTING]"
            elif r.status == "contradicted":
                tag = " [CONTRADICTED - no longer reproduces, do not rely on]"
            elif r.status == "stable":
                tag = " [stable]"
            lines.append(f"  {r.name}: historically {r.baseline:.4g}{spread} "
                         f"over {r.n} runs (conf {r.confidence:.0%}){tag}")
        lines.append("Use these to judge whether this run's values are normal "
                     "or anomalous. Flag large deviations.")
        lines.append("=== END BASELINES ===")
        return "\n".join(lines)

    def check_value(self, dataset: str, name: str, value: float) -> dict | None:
        """Is `value` normal vs history? Returns None if no history."""
        rec = self.get(dataset, name)
        if rec is None or rec.n < 2:
            return None
        z = rec.zscore(value)
        return {
            "baseline": rec.baseline, "stdev": rec.stdev, "zscore": z,
            "is_anomaly": abs(z) >= self.drift_sigma if rec.stdev > 0 else False,
            "n": rec.n, "confidence": rec.confidence,
        }

    # -- persistence ------------------------------------------------------

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"facts": [r.to_dict() for r in self._facts.values()]}
        self.path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        for d in data.get("facts", []):
            try:
                rec = FactRecord.from_dict(d)
                self._facts[rec.key] = rec
            except Exception:
                continue

    def stats(self) -> dict:
        return {
            "facts": len(self._facts),
            "datasets": len({r.dataset for r in self._facts.values()}),
            "stable": sum(1 for r in self._facts.values() if r.status == "stable"),
            "drifting": sum(1 for r in self._facts.values() if r.status == "drifting"),
        }
