"""
Ledgerhound -- Agentic EDA Engine for Fraud Modeling
=================================================

A Python library and CLI that automates Exploratory Data Analysis
for fraud detection pipelines.  It profiles datasets deterministically
(no LLM), then uses a ReAct-based LLM agent to reason over the profile
and generate actionable insights.

Quick start::

    from ledgerhound import Ledgerhound

    engine = Ledgerhound(data="txns.parquet", schema="schema.json")
    digest = engine.profile(target="is_fraud")
    digest.save("digest.yaml")
"""

__version__ = "0.7.9"


def _make_console_utf8_safe() -> None:
    """Force stdout/stderr to UTF-8 so rich output (tables, spinners, panels)
    never crashes on a legacy Windows cp1252 console/pipe. Runs on package
    import so EVERY entry path is covered -- the CLI, `python -m`, and any code
    that imports a ledgerhound submodule directly (e.g. the analyst dispatch).
    ``errors='replace'`` degrades a stray glyph to '?' instead of raising.
    """
    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # py3.7+
        except Exception:
            pass


_make_console_utf8_safe()


# Renamed from Sentinel: adopt SENTINEL_* env vars and move ~/.sentinel
# before any submodule computes a path or reads config at import time
# (identity.py and llm/credentials.py do both).
from ledgerhound import _compat  # noqa: E402

_compat.adopt_legacy_env()
_compat.migrate_home()


def build_info() -> dict:
    """Build provenance so a stale deploy is detectable.

    ``scripts/stamp_build.py`` writes ``ledgerhound/_build.py`` at package/zip
    time with the git short hash + UTC date. When absent (a plain checkout),
    we report ``"dev"`` -- but the moment a build is stamped, ``ledgerhound
    version`` shows exactly which commit is live, so "I unzipped a new
    build but the command is still missing" becomes diagnosable.
    """
    try:
        from ledgerhound import _build  # type: ignore
        return dict(getattr(_build, "BUILD", {})) or {"build": "dev"}
    except Exception:
        return {"build": "dev"}


from ledgerhound.core import Ledgerhound

__all__ = ["Ledgerhound", "build_info"]
