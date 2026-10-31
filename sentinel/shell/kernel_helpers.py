"""
sentinel.shell.kernel_helpers
================================

Makes the analyst's sandbox helpers -- ``check_leakage``,
``evaluate_feature``, ``evaluate_features`` -- exist inside the REAL
IPython kernel, not only in the restricted exec sandbox.

Why this module exists (audit finding, 2026-08-06)
----------------------------------------------------
Those three helpers were injected into ``ConversationalAnalyst._ns``,
which is populated ONLY in ``_tool_run_python``'s exec branch --
``_run_python_kernel`` returns before ever reaching it. So in kernel
mode they simply did not exist. That was not a silent degradation: the
system prompt MANDATES calling them ("Before recommending ANY engineered
feature as valuable, check it actually separates fraud from non-fraud:
evaluate_feature(...)"), so the model would call one, get a bare
``NameError``, get no useful hint back (``_teach_error_text`` passes
``names=[]`` in kernel mode -- it cannot introspect kernel-side
variables), and per the prompt's own "state discipline" rule change
approach -- i.e. fall back to recommending features on judgment alone.
That is precisely the original pilot complaint ("only 6 of the top 50
features mattered") reappearing, silently, on the GPU path -- which is
the DEFAULT path whenever a GPU is present.

The cuDF wrinkle (why this is not a one-line injection)
--------------------------------------------------------
``sentinel.analysis.feature_eval`` is pure pandas/numpy (``pd.qcut``,
``np.histogram``, ...). In kernel mode with ``engine="cudf"``, ``df`` is
a **cuDF** frame, so handing it straight to that code raises rather than
computing. ``_sentinel_host_cols`` therefore moves ONLY the columns a
given call actually needs (feature + target -- two columns, not the whole
frame) to host memory first. At 8-crore scale that is ~1.3 GB for a
two-column float pull, not a full-dataset round trip.

``check_leakage`` takes feature NAMES and metadata, never the frame, so
it needs no conversion at all.

Failure posture: if ``sentinel`` is somehow not importable inside the
kernel process, each helper is still DEFINED but returns an explicit
``{"ok": False, "reason": ...}`` dict. A helper that returns a readable
reason is strictly better than a ``NameError``, which is the exact
failure this module exists to remove.
"""

from __future__ import annotations

# Executed inside the kernel process at startup. Kept as one self-contained
# string (no f-string interpolation) so nothing here can be broken by a
# stray brace in a path or a repr.
ANALYST_HELPERS_SNIPPET = r'''
def _sentinel_host_cols(_df, _cols):
    """Return a pandas DataFrame of just `_cols`, converted from cuDF to
    host memory if needed. Only the requested columns are moved -- never
    the whole frame."""
    _cols = [c for c in _cols if c is not None]
    _sub = _df[_cols] if _cols else _df
    # dask_cudf / dask frames are LAZY -- materialise first, or the pandas-only
    # code downstream (qcut, histogram) receives an unevaluated graph and
    # either raises or silently misbehaves. compute() on a two-column slice
    # is the cheap case; we never materialise the whole frame.
    if hasattr(_sub, "compute") and not hasattr(_sub, "iloc"):
        _sub = _sub.compute()
    elif hasattr(_sub, "compute") and hasattr(_sub, "npartitions"):
        _sub = _sub.compute()
    if hasattr(_sub, "to_pandas"):      # cuDF -> pandas
        _sub = _sub.to_pandas()
    return _sub


def evaluate_feature(df, feature, target, n_bins=10):
    """IV / KS / AUC for one feature vs the label. Works on cuDF or pandas."""
    try:
        from sentinel.analysis.feature_eval import evaluate_feature as _ev
    except Exception as _e:
        return {"feature": feature, "ok": False,
                "reason": "sentinel.analysis.feature_eval not importable in "
                          "this kernel: %s: %s" % (type(_e).__name__, _e)}
    try:
        _host = _sentinel_host_cols(df, [feature, target])
    except Exception as _e:
        return {"feature": feature, "ok": False,
                "reason": "could not read columns %r/%r: %s: %s"
                          % (feature, target, type(_e).__name__, _e)}
    _r = _ev(_host, feature, target, n_bins=n_bins)
    if _r is None:
        return {"feature": feature, "ok": False,
                "reason": "'%s' or '%s' not in df.columns, or the feature is "
                          "entirely null" % (feature, target)}
    _d = _r.to_dict()
    _d["ok"] = True
    return _d


def evaluate_features(df, features, target, n_bins=10, corr_threshold=0.9):
    """Ranked batch version of evaluate_feature (also prunes redundant
    correlated features). Works on cuDF or pandas."""
    try:
        from sentinel.analysis.feature_eval import evaluate_features as _ev
    except Exception as _e:
        return {"ok": False,
                "reason": "sentinel.analysis.feature_eval not importable in "
                          "this kernel: %s: %s" % (type(_e).__name__, _e)}
    _feats = list(features)
    try:
        _host = _sentinel_host_cols(df, _feats + [target])
    except Exception as _e:
        return {"ok": False,
                "reason": "could not read the requested columns: %s: %s"
                          % (type(_e).__name__, _e)}
    return _ev(_host, _feats, target, n_bins=n_bins,
               corr_threshold=corr_threshold).to_dict()


def percentile_table(fraud, nonfraud):
    """V3-051: side-by-side percentile view of one feature, two
    populations -- average, p1..p99. Accepts a cuDF or pandas Series/array
    on either side (a cuDF Series has `.to_pandas()`, not `.columns`)."""
    from sentinel.analysis.feature_eval import percentile_table as _pt

    def _host(x):
        return x.to_pandas() if hasattr(x, "to_pandas") else x
    return _pt(_host(fraud), _host(nonfraud))


def separation_callout(table):
    """V3-051: mechanically derive where two populations diverge, overlap,
    or cross from a percentile_table(...) result."""
    from sentinel.analysis.feature_eval import separation_callout as _sc
    return _sc(table)


def check_leakage(features, label_column=None, **kw):
    """Point-in-time / target-leak vet. Takes feature NAMES (or per-feature
    dicts), never the frame -- so no cuDF conversion is involved."""
    try:
        from sentinel.validation.leakage import FeatureSpec, LeakageSentinel
    except Exception as _e:
        return {"ok": False,
                "reason": "sentinel.validation.leakage not importable in this "
                          "kernel: %s: %s" % (type(_e).__name__, _e)}
    _specs = []
    for _f in features:
        if isinstance(_f, dict):
            _specs.append(FeatureSpec(
                name=_f["name"],
                computed_at=_f.get("computed_at", "unknown"),
                as_of_note=_f.get("as_of_note", ""),
                inputs=list(_f.get("inputs", []) or []),
                is_group_aggregate=bool(_f.get("is_group_aggregate", False)),
                fit_scope=_f.get("fit_scope", "unknown"),
            ))
        else:
            _specs.append(FeatureSpec(name=str(_f)))
    _rep = LeakageSentinel().vet(_specs, label_column=label_column, **kw)
    _rep.display()
    _d = _rep.to_dict()
    _d["vetoed_features"] = _rep.vetoed_features()
    return _d
'''

# The workbench (cached_agg / sample_frame) ships alongside the helpers --
# one injection, one place for the kernel-side analyst toolkit.
try:
    from sentinel.shell.workbench import WORKBENCH_SNIPPET
    ANALYST_HELPERS_SNIPPET = ANALYST_HELPERS_SNIPPET + "\n" + WORKBENCH_SNIPPET
except Exception:  # noqa: BLE001 -- helpers must load even if workbench doesn't
    pass
