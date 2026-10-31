"""
sentinel.eval.eval_record
==========================

``eval_record/v1`` -- the durable, append-only unit of the eval-first data
flywheel. One record per run, written to MinIO (and optionally a local mirror)
so eval quality can be trended across runs, datasets, and agent versions.

Design
------
* **Owner-scoped private tier.** Records land under ``{owner}/eval/dt=.../`` in
  the data bucket (see the private-vs-shared memory scope decision). Nothing is
  promoted to a shared tier here; that is the Librarian's job later.
* **Process-level fields.** Beyond final metrics, the record carries per-step
  / milestone trajectory data (SOTA eval is process-level: DataClawBench,
  DSAEval). ``milestones`` is a list of {name, reached, detail}.
* **Self-describing + reproducible.** Carries config/manifest hashes and token
  accounting so any number can be tied back to the exact run that produced it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sentinel.identity import current_user

SCHEMA = "eval_record/v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EvalRecord:
    """One run's eval record. Build with :meth:`from_run`, then :meth:`write`."""

    run_id: str
    dataset: str
    schema: str = SCHEMA
    owner: str = field(default_factory=current_user)
    created_at: str = field(default_factory=_now)

    # reproducibility
    config_hash: str = ""
    manifest_hash: str = ""

    # cost
    tokens: int = 0
    llm_calls: int = 0

    # trajectory / integrity (from scorecard.py)
    scorecard: dict[str, Any] = field(default_factory=dict)

    # golden / ground-truth metrics (optional; per-motif P/R/F1/AUC etc.)
    ground_truth_source: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    # process-level milestones: [{"name": str, "reached": bool, "detail": str}]
    milestones: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_run(
        cls,
        run_id: str,
        dataset: str,
        scorecard: dict | None = None,
        *,
        owner: str | None = None,
        config_hash: str = "",
        manifest_hash: str = "",
        tokens: int = 0,
        llm_calls: int = 0,
        ground_truth_source: str = "",
        metrics: dict | None = None,
        milestones: list[dict] | None = None,
    ) -> "EvalRecord":
        return cls(
            run_id=run_id,
            dataset=dataset,
            owner=owner or current_user(),
            config_hash=config_hash,
            manifest_hash=manifest_hash,
            tokens=tokens,
            llm_calls=llm_calls,
            scorecard=scorecard or {},
            ground_truth_source=ground_truth_source,
            metrics=metrics or {},
            milestones=milestones or [],
        )

    # -- persistence ------------------------------------------------------

    def write_local(self, path: str | Path) -> Path:
        """Mirror the record to a local JSONL file (append)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.to_dict(), default=str) + "\n")
        return p

    def write_minio(self, store) -> str:
        """Append to MinIO under the owner-scoped private eval prefix.

        Layout: ``{owner}/eval/dt=YYYY-MM-DD/<uuid>.jsonl`` in the data bucket.
        Returns the object key.
        """
        return store.append_jsonl("data", f"{self.owner}/eval", self.to_dict())
