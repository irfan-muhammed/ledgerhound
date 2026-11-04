"""
ledgerhound.findings.falsification
===============================

V5-004. Turn a recorded finding into the analysis that would refute it.

Every other chapter in the taxonomy looks for support. That is the direction
bias already runs in, and an agent that has just written a claim is the worst
possible judge of it -- it will find more evidence, because more evidence is
what it went looking for.

Popper (arXiv 2502.09858) validates hypotheses by designing experiments
targeting their *measurable implications* and gathering evidence that could
kill them. This module does the small, mechanical part of that: given a claim,
it names the standard ways a claim of that shape turns out to be wrong, so the
attack is specified before the analyst starts rather than improvised into
another confirmation.

The routes below are the ones that have actually produced retractions in this
project's own history -- a sample artifact read as a population fact, a
comparison across two different grains, a split that leaked, a difference that
is real and too small to matter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# (route name, what it would show, when it applies)
_ROUTES: list[tuple[str, str, str]] = [
    ("subgroup-reversal",
     "The direction reverses inside an obvious segment (channel, period, "
     "grain). A claim that holds overall and flips within every segment is "
     "Simpson's paradox, not a finding.",
     "always"),
    ("period-holdout",
     "The effect is absent in a period the claim was not derived from. If it "
     "only exists in the window it was found in, it is a property of that "
     "window.",
     "always"),
    ("prevalence-shift",
     "The effect disappears once the base rate is corrected to deployment "
     "conditions. A separation that only exists at the sampled 1:1000 ratio "
     "does not survive 1:100,000.",
     "rate"),
    ("grain-restatement",
     "The number changes when recomputed at the other grain. Account-grain "
     "and transaction-grain are different figures; if the claim does not name "
     "its grain it has not been tested.",
     "always"),
    ("leakage-origin",
     "The predictive column is downstream of the label -- a case-management "
     "artifact, a risk score, a post-hoc flag. The strongest features are the "
     "likeliest to be leaks.",
     "predictive"),
    ("effect-size",
     "The difference is statistically detectable and practically irrelevant. "
     "At UPI volume almost any difference reaches significance; state the "
     "effect (KS, AUC, a percentile gap) and ask whether it would change a "
     "decision.",
     "significance"),
    ("sample-frame",
     "The claim is a within-stratum result being read as a population one. "
     "Re-derive it on the full frame, or restate the scope.",
     "always"),
]

_RATE = re.compile(r"\b(rate|share|proportion|percent|%|prevalence|ratio)\b", re.I)
_PREDICTIVE = re.compile(
    r"\b(predict|separat|discriminat|signal|auc|iv|importance|driver)\w*\b", re.I)
_SIGNIFICANCE = re.compile(
    r"\b(significant|p ?[<=]|pvalue|p-value|higher|lower|greater|more|less)\b", re.I)


@dataclass
class FalsificationPlan:
    claim: str
    routes: list[tuple[str, str]] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            "FALSIFICATION PLAN",
            "",
            f"Claim under attack: {self.claim}",
            "",
            "Your job in this chapter is to make this claim FAIL. Do not "
            "gather more support for it -- that is what every other chapter "
            "does, and it is why a wrong finding survives. Work these routes; "
            "each one is a way a claim of this shape has actually turned out "
            "wrong before.",
            "",
        ]
        for i, (name, what) in enumerate(self.routes, 1):
            lines.append(f"{i}. {name}")
            lines.append(f"   {what}")
        lines += [
            "",
            "Record the outcome either way. A claim that survives a real "
            "attempt to kill it is worth more than one that was never "
            "attacked; a claim that dies here has saved a model from being "
            "built on it. Use record_finding for the result, with a pvalue "
            "when a test was run.",
        ]
        return "\n".join(lines)


def plan_for(claim: str) -> FalsificationPlan:
    """Routes worth trying against this particular claim."""
    routes = []
    for name, what, when in _ROUTES:
        if when == "always":
            routes.append((name, what))
        elif when == "rate" and _RATE.search(claim):
            routes.append((name, what))
        elif when == "predictive" and _PREDICTIVE.search(claim):
            routes.append((name, what))
        elif when == "significance" and _SIGNIFICANCE.search(claim):
            routes.append((name, what))
    return FalsificationPlan(claim=claim, routes=routes)
