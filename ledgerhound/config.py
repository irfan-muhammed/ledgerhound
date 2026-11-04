"""
ledgerhound.config
================

The ONE place ``main_config.yaml`` gets read from. Everything that used to
be "export this env var before every session" (GPU tuning, Trino,
webui, institutional knowledge dirs, identity, integrations) now lives in
one gitignored file instead -- see ``main_config.example.yaml`` for the
full template.

Precedence, everywhere this module is used: **main_config.yaml wins if the
key is present; an environment variable is the fallback**, not the other
way round. The env var path is kept deliberately -- it's how Kubeflow's
PodDefault injects ``LEDGERHOUND_USER`` in production (see
``ledgerhound.identity``), which nothing here should break -- but for
interactive/local use, main_config.yaml is what you actually edit.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_CACHE: dict[str, dict] = {}


def _config_candidates(path: str | Path | None = None) -> list[Path]:
    """Search order for main_config.yaml:
    1. explicit `path`
    2. $LEDGERHOUND_MAIN_CONFIG (production override)
    3. ./main_config.yaml (cwd -- the file users edit)
    4. repo root / parent of the package (dev checkout)
    5. the template shipped INSIDE the installed package (wheel default,
       owner decision 2026-09-18 -- contains no secrets). Only used when
       nothing user-side exists.
    """
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path))
    else:
        env_path = os.environ.get("LEDGERHOUND_MAIN_CONFIG", "").strip()
        if env_path:
            candidates.append(Path(env_path))
        candidates.append(Path.cwd() / "main_config.yaml")
        candidates.append(Path(__file__).resolve().parent.parent / "main_config.yaml")
        try:
            from importlib.resources import files as _files
            candidates.append(_files("ledgerhound").joinpath("main_config.yaml"))
        except Exception:
            pass
    return candidates


def load_master_config(path: str | Path | None = None) -> dict:
    """Load main_config.yaml once and cache it. See ``_config_candidates``
    for the search order. Missing file -> {}, never an error -- an
    absent/empty main_config.yaml is a valid "using defaults / env vars"
    state, not a bug."""
    cache_key = str(path) if path else "__default__"
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    data: dict = {}
    for c in _config_candidates(path):
        if c.exists():
            data = yaml.safe_load(c.read_text(encoding="utf-8")) or {}
            break
    _CACHE[cache_key] = data
    return data


def clear_cache() -> None:
    """Test helper -- main_config.yaml is cached on first read per path."""
    _CACHE.clear()


def resolved_path(path: str | Path | None = None) -> Path | None:
    """V3-064: which main_config.yaml file (if any) `load_master_config`
    would actually use -- same search order, but returns the PATH, not
    the parsed content. Exists because resolution was previously silent:
    a config filled in correctly but sitting somewhere the search order
    doesn't check (wrong cwd, a non-editable install where the repo-root
    guess resolves into site-packages instead of the real project) failed
    with no visible signal beyond "the value wasn't there." The CLI prints
    this at shell startup so that failure mode is diagnosable."""
    for c in _config_candidates(path):
        if c.exists():
            return c.resolve()
    return None


def search_candidates() -> list[Path]:
    """The paths `resolved_path`/`load_master_config` check, in order --
    for a diagnostic message when NONE of them exist."""
    return _config_candidates()


def get(*keys: str, default: Any = None, path: str | Path | None = None) -> Any:
    """Read a nested key path, e.g. ``get("gpu", "enable_dask_cudf")``.
    Returns `default` if any level is missing or not a dict."""
    node: Any = load_master_config(path)
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k)
    return node if node is not None else default


def get_str(*keys: str, env: str | None = None, default: str = "") -> str:
    """main_config.yaml wins; falls back to the `env` var if given, then
    `default`."""
    v = get(*keys)
    if v is not None and str(v).strip():
        return str(v)
    if env:
        return os.environ.get(env, default)
    return default


def get_bool(*keys: str, env: str | None = None, default: bool = False) -> bool:
    v = get(*keys)
    if v is None and env:
        v = os.environ.get(env)
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() not in ("0", "false", "no", "off", "")


def get_int(*keys: str, env: str | None = None, default: int = 0) -> int:
    v = get(*keys)
    if v is None and env:
        v = os.environ.get(env)
    if v is None or str(v).strip() == "":
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default
