"""
sentinel.shell.stat_checks
==========================

V5-003. The eighth `_check_*`: does the statistical test in this cell hold
the assumptions it is built on?

The seven checks that came before are all about *data* -- comparability,
leakage, grain, numeric stability, ambiguous columns, memory, mutation delta.
None of them looks at inference. Fisher-R1 (arXiv 2608.07437) measured what
that leaves open: LLM agents "generate technically valid analyses while making
flawed inferential choices." On their P-Bench, GPT-5.4 reached the right
reject/fail-to-reject conclusion 58.3% of the time but produced a valid
p-value only 30.5% of the time -- right for the wrong reasons roughly half the
times it was right at all.

The code runs. The numbers come out. The conclusion is wrong. Nothing
downstream can tell.

What this catches, and what it does not
---------------------------------------
Static: the cell's source is parsed for `scipy.stats` test calls, and each
known test is matched against the assumptions it requires. Nothing is
executed, nothing is re-run, and the frame is not touched -- this is a cheap
pre-flight, not a re-analysis.

It therefore catches the *stated* misuse -- a t-test where the data is
plainly ordinal, a chi-square on a tiny contingency table, a paired test on
unpaired inputs -- and cannot catch an assumption that only fails once the
data is seen. V5-003's job is the class of error that is visible in the code.

**Advisory by default, blocking for the unambiguous.** A check that fires
false positives gets switched off, and a switched-off check protects nothing.
Only cases where the test is wrong *regardless of the data* raise a refusal;
everything else is annotated and left to the analyst.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

# What each test needs to be valid. `blocking` marks the cases where the
# choice is wrong on its face -- no data could rescue it.
TEST_ASSUMPTIONS: dict[str, dict] = {
    "ttest_ind": {
        "needs": ["both samples approximately normal OR n large enough for CLT",
                  "equal variances unless equal_var=False",
                  "independent observations"],
        "note": ("scipy defaults to equal_var=True (Student's). Real-world "
                 "fraud/non-fraud cohorts almost never have equal variance "
                 "and are wildly different sizes -- pass equal_var=False "
                 "(Welch) unless you have checked, or use mannwhitneyu."),
    },
    "ttest_rel": {
        "needs": ["paired observations in matching order",
                  "differences approximately normal"],
        "note": ("A paired test requires that row i of both inputs is the "
                 "SAME entity. Two cohorts of different length cannot be "
                 "paired; if you meant two independent groups, this is the "
                 "wrong test."),
    },
    "ttest_1samp": {
        "needs": ["approximately normal, or n large"],
        "note": "",
    },
    "pearsonr": {
        "needs": ["linear relationship", "both variables roughly normal",
                  "no dominating outliers"],
        "note": ("Amounts are lognormal-ish and outlier-heavy; Pearson on a "
                 "raw amount column mostly measures the tail. spearmanr is "
                 "usually the honest choice here."),
    },
    "f_oneway": {
        "needs": ["normality within groups", "equal variances across groups",
                  "independent observations"],
        "note": "With unequal variances use kruskal instead.",
    },
    "chi2_contingency": {
        "needs": ["expected count >= 5 in (almost) every cell",
                  "counts, not proportions or already-aggregated rates"],
        "note": ("At fraud prevalence the rare cell is the whole point and is "
                 "exactly the one that breaks the approximation -- check "
                 "expected counts, or use fisher_exact."),
    },
    "ks_2samp": {
        "needs": ["continuous distributions", "independent samples"],
        "note": ("On heavily tied/discrete data the KS null is wrong and the "
                 "p-value is anti-conservative."),
    },
    "mannwhitneyu": {
        "needs": ["independent samples", "ordinal or continuous"],
        "note": "",
    },
    "shapiro": {
        "needs": ["n below a few thousand"],
        "note": ("Shapiro-Wilk rejects normality for essentially ANY real "
                 "sample once n is large -- at UPI scale it will always say "
                 "'not normal' and tells you nothing. Judge the size of the "
                 "departure, not its p-value."),
    },
}

# Tests whose null is about *equality of distribution*, where a large sample
# makes any trivial difference "significant".
_SCALE_SENSITIVE = {"ks_2samp", "chi2_contingency", "shapiro", "ttest_ind",
                    "f_oneway"}

# A p-value compared against a bare literal, i.e. a hard-coded alpha.
_ALPHA_LITERAL = re.compile(
    r"p(?:value|_value|val)?\s*[<>]=?\s*0?\.0\d+", re.IGNORECASE)


@dataclass
class StatIssue:
    test: str
    kind: str
    blocking: bool
    message: str


def _called_tests(tree: ast.AST) -> list[tuple[str, ast.Call]]:
    """Every scipy.stats test call in the cell, as (test_name, node)."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = (fn.attr if isinstance(fn, ast.Attribute)
                else fn.id if isinstance(fn, ast.Name) else None)
        if name in TEST_ASSUMPTIONS:
            found.append((name, node))
    return found


_SIZE_REDUCERS = {"sample", "head", "tail", "nsmallest", "nlargest"}


def _arg_rooted_in(node: ast.Call, var: str) -> bool:
    """True when the call's first positional argument is `var` itself or a
    subscript/attribute chain rooted in it (df.amount, df['amount'],
    df.amount.dropna())."""
    if not node.args:
        return False
    cur = node.args[0]
    while True:
        if isinstance(cur, ast.Name):
            return cur.id == var
        if isinstance(cur, ast.Call):
            # A sampling / truncating call anywhere in the chain means the
            # test runs on a subset of unknown size -- the right way to use
            # shapiro on big data -- so it is not "the loaded frame".
            fn = cur.func
            if isinstance(fn, ast.Attribute) and fn.attr in _SIZE_REDUCERS:
                return False
            cur = fn
        elif isinstance(cur, ast.Subscript):
            if isinstance(cur.slice, ast.Slice):
                return False
            cur = cur.value
        elif isinstance(cur, ast.Attribute):
            if cur.attr in ("iloc", "loc") :
                return False
            cur = cur.value
        else:
            return False


def _kwarg(node: ast.Call, name: str):
    for kw in node.keywords:
        if kw.arg == name:
            return kw
    return None


def check_statistical_assumptions(code: str, *,
                                  n_rows: int | None = None,
                                  seen: set | None = None) -> list[StatIssue]:
    """Inspect `code` for statistical tests used outside their assumptions.

    `seen` suppresses repeats within a session -- the same advice on every
    cell is noise, and noise is what gets a check disabled.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []          # the cell will fail on its own terms

    seen = seen if seen is not None else set()
    issues: list[StatIssue] = []

    def add(issue: StatIssue) -> None:
        key = (issue.test, issue.kind)
        if key in seen:
            return
        seen.add(key)
        issues.append(issue)

    for name, node in _called_tests(tree):
        spec = TEST_ASSUMPTIONS[name]

        # -- the unambiguous cases: wrong regardless of the data -----------
        if name == "ttest_ind" and _kwarg(node, "equal_var") is None:
            add(StatIssue(
                test=name, kind="default_equal_var", blocking=False,
                message=(
                    "ttest_ind() without equal_var=False assumes the two "
                    "groups have EQUAL VARIANCE. Fraud and non-fraud cohorts "
                    "differ in both spread and size, which is the exact case "
                    "Student's t handles badly. Pass equal_var=False (Welch), "
                    "or use mannwhitneyu if you do not want a normality "
                    "assumption at all.")))

        if name == "ttest_rel" and len(node.args) >= 2:
            add(StatIssue(
                test=name, kind="paired_on_two_cohorts", blocking=True,
                message=(
                    "ttest_rel() is a PAIRED test: it requires row i of both "
                    "inputs to be the same entity observed twice. Two "
                    "separate cohorts are not paired, and running this on "
                    "them produces a number with no meaning. Use ttest_ind "
                    "(with equal_var=False) or mannwhitneyu for independent "
                    "groups.")))

        if name == "pearsonr":
            add(StatIssue(
                test=name, kind="pearson_on_skewed", blocking=False,
                message=(
                    "pearsonr() measures LINEAR association and is dominated "
                    "by outliers. " + spec["note"])))

        if name == "shapiro" and (n_rows or 0) > 5000:
            # J-05: `n_rows` is the LOADED frame's size. The test is only
            # certainly meaningless when it runs on that frame (an arg
            # rooted in `df`); on some other variable (a sample, an
            # aggregate) the size is unknown here, so advise, don't refuse.
            add(StatIssue(
                test=name, kind="shapiro_at_scale",
                blocking=_arg_rooted_in(node, "df"),
                message=(
                    f"shapiro() on {n_rows:,} rows will reject normality "
                    f"whatever the data looks like -- the test's power grows "
                    f"with n until every real dataset fails it. The result "
                    f"carries no information at this size. " + spec["note"])))

        if name in _SCALE_SENSITIVE and (n_rows or 0) > 1_000_000:
            add(StatIssue(
                test=name, kind="significance_at_scale", blocking=False,
                message=(
                    f"{name}() on {n_rows:,} rows will return a tiny p-value "
                    f"for a difference of no practical size. At this scale "
                    f"report the EFFECT (KS statistic, AUC, a percentile "
                    f"table) and treat the p-value as decoration.")))

        if spec["note"] and name not in {"pearsonr", "shapiro"}:
            add(StatIssue(
                test=name, kind="assumptions", blocking=False,
                message=(f"{name}() assumes: "
                         + "; ".join(spec["needs"]) + ". " + spec["note"])))

    # A hard-coded alpha next to a test is the multiplicity smell V5-002
    # accounts for -- flag the connection rather than leaving it implicit.
    if _called_tests(tree) and _ALPHA_LITERAL.search(code):
        add(StatIssue(
            test="(alpha)", kind="hardcoded_alpha", blocking=False,
            message=(
                "A p-value compared against a hard-coded threshold. If this "
                "cell is one of several comparisons, that threshold is not "
                "the right bar -- pass the p-value to record_finding(pvalue=) "
                "and let it be corrected across the chapter's other tests.")))

    return issues


def render(issues: list[StatIssue]) -> str:
    """Advisory block, or a refusal if anything blocking fired."""
    if not issues:
        return ""
    blocking = [i for i in issues if i.blocking]
    head = ("REFUSED: a statistical test in this cell cannot be valid as "
            "written." if blocking else "[STATISTICAL ASSUMPTIONS]")
    lines = [head]
    for i in (blocking or issues):
        lines.append(f"  - {i.message}")
    if not blocking:
        lines.append("  Advisory -- the cell ran. Fix the test or say why it "
                     "is fine.")
    return "\n".join(lines)


def is_blocking(issues: list[StatIssue]) -> bool:
    return any(i.blocking for i in issues)
