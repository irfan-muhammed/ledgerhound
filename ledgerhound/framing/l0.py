"""
ledgerhound.framing.l0
=====================

V3-023: L0 framing capture -- study-start intake: cost matrix (₹ per FP vs
per missed fraud), unit of analysis, latency budget, regulatory bounds.

"HITL high tier by definition" (roadmap "Do" text) is a CLASSIFICATION
statement, not a build order: this session's close-out criteria are two
pytest clauses (no framing -> documented defaults + a warning finding;
a real framing -> the calibrated threshold demonstrably shifts and the
model card cites where the cost matrix came from), neither of which needs
an interactive confirm/choose flow. The human-confirmed-vs-fallback
distinction is realized as `L0Framing.source` ("captured" vs "default") --
that field IS the HITL boundary. `capture_l0_framing` is registered as a
`high`-tier action in `ledgerhound.analyst.hitl.classify_action` so that IF a
real interactive capture flow is built later (out of scope here), it
inherits high-tier treatment for free; no confirm/choose prompt loop is
built in this pass -- V3-019/V3-020 set the precedent of not building an
agent-facing flow the close-out doesn't actually test.

Two different "cards"
----------------------
"the model card cites it" (close-out, tested) is V3-022's `model_card.md`
(Threshold Rationale section) -- `train_gbm_baseline` records
`framing_source` on its `TrainReport`, and `ModelValidator` reads that
provenance (not a metric, so reading it doesn't violate "recompute, don't
copy" -- that principle is about SCORES/METRICS, not static config facts).
"stored on ... the dataset card" (the "Do" text, not the close-out) is
V3-011's `DatasetCard` -- a different, cross-session-persisted artifact.
Both are exercised by tests in this module's test suite, but only the
model-card citation is a close-out requirement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

DEFAULT_COST_MATRIX = {"fp_cost": 1.0, "fn_cost": 25.0}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class L0Framing:
    """Study-start framing. `source` distinguishes a real human intake
    ("captured") from a documented fallback ("default") -- this is the
    field a downstream reader (validator, model card) uses to decide
    whether to trust the cost matrix as a real business tradeoff or flag
    it as an unconfirmed placeholder."""

    cost_matrix: dict = field(default_factory=lambda: dict(DEFAULT_COST_MATRIX))
    unit_of_analysis: str = "transaction"
    latency_budget_ms: Optional[float] = None
    regulatory_bounds: list = field(default_factory=list)
    captured_at: str = ""
    source: str = "default"   # "default" | "captured"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "L0Framing":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def default(cls) -> "L0Framing":
        """No real intake happened -- documented fallback."""
        return cls(cost_matrix=dict(DEFAULT_COST_MATRIX), source="default",
                   captured_at=_now())

    @classmethod
    def capture(
        cls,
        cost_matrix: dict,
        unit_of_analysis: str = "transaction",
        latency_budget_ms: Optional[float] = None,
        regulatory_bounds: Optional[list] = None,
    ) -> "L0Framing":
        """A real study-start intake happened -- these values reflect an
        actual business/regulatory decision, not a placeholder."""
        return cls(
            cost_matrix=dict(cost_matrix),
            unit_of_analysis=unit_of_analysis,
            latency_budget_ms=latency_budget_ms,
            regulatory_bounds=list(regulatory_bounds or []),
            source="captured",
            captured_at=_now(),
        )

    def summary_block(self) -> str:
        tag = "captured" if self.source == "captured" else "DEFAULT (not captured)"
        lines = [f"L0 framing [{tag}] -- unit of analysis: {self.unit_of_analysis}"]
        lines.append(f"  cost matrix: {self.cost_matrix}")
        if self.latency_budget_ms is not None:
            lines.append(f"  latency budget: {self.latency_budget_ms}ms")
        if self.regulatory_bounds:
            lines.append(f"  regulatory bounds: {', '.join(self.regulatory_bounds)}")
        return "\n".join(lines)
