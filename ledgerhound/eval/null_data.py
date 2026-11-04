"""
ledgerhound.eval.null_data
=======================

V5-009. The evaluation this project should have built before reaching for
BLADE: **false-discovery rate under a ground truth we construct.**

Every benchmark we looked at scores whether an agent's analysis *decisions*
resemble a human analyst's. That is not our claim. Ledgerhound's claim is about
process integrity -- a finding found on the two-hundredth look is not recorded
as fact, a sample-derived number is not restated as a population one, a second
session does not silently redo the first's work. An agent could match human
decisions closely and still fail every one of those.

So: generate data where the answer is known by construction, and count what
gets through.

  * **Null arm.** Labels are assigned at random; no feature carries signal.
    Anything recorded as established is, by construction, false. The ideal
    score is zero, and nobody has to trust our answer key -- there is nothing
    there to find.
  * **Signal arm.** A known number of features carry a real effect. This is
    the half that stops the first half being gamed: a gate that refuses
    everything scores a perfect zero on null data and is worthless.

Reporting one arm without the other is how a discipline mechanism gets sold as
working when it is merely strict. Both, always.

What this measures
------------------
The accounting in `findings/search_effort.py`, end to end, against the exact
failure it was built for. It is deterministic, needs no LLM, and runs in
seconds -- so it can sit in CI and fail a build, which no LLM benchmark can.

What it does not measure is whether a *model* driving the tools behaves well;
that needs the agent loop and is noisier. This measures whether the mechanism
is sound, which has to be true first.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ledgerhound.findings.search_effort import ESTABLISHED, SearchEffortLedger

LABEL = "is_fraud"


def make_null_dataset(n_rows: int = 4000, n_features: int = 60,
                      prevalence: float = 0.03, seed: int = 0
                      ) -> pd.DataFrame:
    """Features and label are independent by construction. Nothing to find."""
    rng = np.random.default_rng(seed)
    data = {f"f{i:02d}": rng.normal(0, 1, n_rows) for i in range(n_features)}
    data[LABEL] = (rng.random(n_rows) < prevalence).astype(int)
    return pd.DataFrame(data)


def make_planted_dataset(n_rows: int = 4000, n_features: int = 60,
                         n_planted: int = 6, effect: float = 0.55,
                         prevalence: float = 0.03, seed: int = 0
                         ) -> tuple[pd.DataFrame, list[str]]:
    """Exactly `n_planted` features carry a real mean shift in the positive
    class. Everything else is noise. Returns (frame, planted column names)."""
    rng = np.random.default_rng(seed)
    label = (rng.random(n_rows) < prevalence).astype(int)
    data = {f"f{i:02d}": rng.normal(0, 1, n_rows) for i in range(n_features)}
    planted = [f"f{i:02d}" for i in range(n_planted)]
    for col in planted:
        data[col] = data[col] + label * effect
    data[LABEL] = label
    return pd.DataFrame(data), planted


def sweep_comparisons(df: pd.DataFrame, label: str = LABEL) -> dict[str, float]:
    """The comparison an analyst actually runs: every feature against the
    label, one test each. This is the search -- and the reason multiplicity
    matters, since it produces one p-value per column by construction."""
    from scipy import stats

    pos = df[df[label] == 1]
    neg = df[df[label] == 0]
    out: dict[str, float] = {}
    for col in df.columns:
        if col == label:
            continue
        a, b = pos[col].dropna(), neg[col].dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        # Mann-Whitney: no normality assumption, which is what V5-003 would
        # push an analyst toward anyway.
        out[col] = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    return out


@dataclass
class ArmScore:
    """One (gating mode x dataset kind) cell."""
    gated: bool
    kind: str                      # "null" | "signal"
    n_tests: int = 0
    n_established: int = 0
    false_discoveries: int = 0     # established but not truly signal
    true_discoveries: int = 0      # established and truly signal
    n_planted: int = 0

    @property
    def fdr(self) -> float:
        """Share of established findings that are false."""
        return (self.false_discoveries / self.n_established
                if self.n_established else 0.0)

    @property
    def power(self) -> float:
        """Share of planted effects recovered."""
        return (self.true_discoveries / self.n_planted
                if self.n_planted else 0.0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fdr"] = self.fdr
        d["power"] = self.power
        return d


def score_once(pvalues: dict[str, float], truth: set[str], *, gated: bool,
               kind: str, alpha: float = 0.05) -> ArmScore:
    """Record every comparison as a finding attempt and count what sticks.

    `gated=False` reproduces what a tool without family accounting does:
    threshold each raw p-value at alpha and call it a finding. `gated=True`
    routes the same p-values through the real ledger.
    """
    score = ArmScore(gated=gated, kind=kind, n_tests=len(pvalues),
                     n_planted=len(truth))

    if not gated:
        established = {f for f, p in pvalues.items() if p < alpha}
    else:
        ledger = SearchEffortLedger(None)
        claims = {}
        for feature, p in pvalues.items():
            claim = f"{feature} separates the two cohorts"
            claims[claim[:200]] = feature
            ledger.note_attempt("interaction-sweep")
            ledger.assess("interaction-sweep", claim, p)
        # Read the FINAL statuses, not the verdict each claim got when it was
        # first assessed. The family grows as the sweep runs, so an early
        # claim can be demoted by a later one -- scoring the assess-time
        # verdict would credit the gate with catching less than it does, and
        # would make the number depend on column order.
        established = {claims[k] for k, v in
                       ledger.statuses("interaction-sweep").items()
                       if v == ESTABLISHED and k in claims}

    for feature in established:
        score.n_established += 1
        if feature in truth:
            score.true_discoveries += 1
        else:
            score.false_discoveries += 1
    return score


@dataclass
class EvalResult:
    n_seeds: int
    scores: list[ArmScore] = field(default_factory=list)

    def cell(self, *, gated: bool, kind: str) -> list[ArmScore]:
        return [s for s in self.scores if s.gated is gated and s.kind == kind]

    def mean(self, *, gated: bool, kind: str, attr: str) -> float:
        cells = self.cell(gated=gated, kind=kind)
        if not cells:
            return 0.0
        return sum(getattr(c, attr) for c in cells) / len(cells)

    def to_dict(self) -> dict:
        return {"n_seeds": self.n_seeds,
                "scores": [s.to_dict() for s in self.scores]}


def run_eval(*, n_seeds: int = 20, n_rows: int = 4000, n_features: int = 60,
             n_planted: int = 6, effect: float = 0.55,
             prevalence: float = 0.03, alpha: float = 0.05) -> EvalResult:
    """Both arms, both gating modes, averaged over seeds.

    Seeds matter: a single draw of null data tells you very little, and the
    quantity of interest (how many false findings survive) is itself a random
    variable. One run would be exactly the anecdote this module exists to
    argue against.
    """
    res = EvalResult(n_seeds=n_seeds)
    for seed in range(n_seeds):
        null_df = make_null_dataset(n_rows, n_features, prevalence, seed)
        null_p = sweep_comparisons(null_df)
        for gated in (False, True):
            res.scores.append(score_once(null_p, set(), gated=gated,
                                         kind="null", alpha=alpha))

        sig_df, planted = make_planted_dataset(
            n_rows, n_features, n_planted, effect, prevalence, seed + 1000)
        sig_p = sweep_comparisons(sig_df)
        for gated in (False, True):
            res.scores.append(score_once(sig_p, set(planted), gated=gated,
                                         kind="signal", alpha=alpha))
    return res


def render(res: EvalResult) -> str:
    lines = [
        "FALSE DISCOVERY UNDER KNOWN GROUND TRUTH",
        f"(mean over {res.n_seeds} seeds)",
        "",
        "NULL data -- no feature carries signal, so every established "
        "finding is false by construction:",
        f"  {'mode':<14}{'established':<14}{'ideal'}",
    ]
    for gated, name in ((False, "ungated"), (True, "gated")):
        m = res.mean(gated=gated, kind="null", attr="n_established")
        lines.append(f"  {name:<14}{m:<14.2f}0")

    lines += ["",
              "SIGNAL data -- known planted effects, so this is the half that "
              "stops the above being won by refusing everything:"]
    planted = res.cell(gated=True, kind="signal")
    n_planted = planted[0].n_planted if planted else 0
    lines.append(f"  {'mode':<14}{'recovered':<14}{'false':<10}"
                 f"{'power':<10}{'FDR'}")
    for gated, name in ((False, "ungated"), (True, "gated")):
        tp = res.mean(gated=gated, kind="signal", attr="true_discoveries")
        fp = res.mean(gated=gated, kind="signal", attr="false_discoveries")
        pw = res.mean(gated=gated, kind="signal", attr="power")
        fdr = res.mean(gated=gated, kind="signal", attr="fdr")
        lines.append(f"  {name:<14}{tp:.2f}/{n_planted:<9}{fp:<10.2f}"
                     f"{pw:<10.1%}{fdr:.1%}")
    return "\n".join(lines)
