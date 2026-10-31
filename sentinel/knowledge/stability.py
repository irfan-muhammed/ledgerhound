"""
sentinel.knowledge.stability
=============================

V3-055 (0.5.0 next-steps item 1): the Librarian's story model made two
data versions on one book directly diffable -- opening the same chapter
on a new story (new ``data_sha256``) auto-allows precisely so a later
period can be compared against an earlier one. What it did not do is
automate the comparison itself; without this module that diff had to be
done by hand in the sandbox.

This module is deliberately thin: it looks up two recorded percentile
snapshots (``DatasetCard.add_percentile_snapshot`` /
``find_percentile_snapshot``) and hands them to
``sentinel.analysis.feature_eval.compare_percentile_tables``. No new
statistics live here -- just the lookup-and-report glue, so a missing
snapshot on either side is reported plainly instead of raising.
"""

from __future__ import annotations


def check_cross_period_stability(card, sha_a: str, sha_b: str, chapter: str,
                                 feature: str, *, rel_tol: float = 0.15) -> dict:
    """Compare the recorded percentile snapshot for `feature` under
    `chapter` between the stories for content hashes `sha_a` and `sha_b`
    on `card`. Never raises -- a missing snapshot on either side is
    reported in the result, not an exception, since "we haven't recorded
    that yet" is a normal, expected state, not a bug.
    """
    from sentinel.analysis.feature_eval import compare_percentile_tables

    snap_a = card.find_percentile_snapshot(sha_a, chapter, feature)
    snap_b = card.find_percentile_snapshot(sha_b, chapter, feature)
    if snap_a is None or snap_b is None:
        missing = []
        if snap_a is None:
            missing.append(f"story for {sha_a[:10]}...")
        if snap_b is None:
            missing.append(f"story for {sha_b[:10]}...")
        return {
            "ok": False,
            "reason": (f"no recorded percentile snapshot for '{feature}' "
                      f"under chapter '{chapter}' in: {', '.join(missing)} "
                      "-- record one with record_percentile_snapshot first."),
        }
    result = compare_percentile_tables(
        snap_a["percentiles"], snap_b["percentiles"], rel_tol=rel_tol)
    result.update({"ok": True, "feature": feature, "chapter": chapter,
                   "story_a": sha_a[:10], "story_b": sha_b[:10]})
    return result
