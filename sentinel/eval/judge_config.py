"""
sentinel.eval.judge_config
===========================

Resolve WHICH model acts as the evaluation judge, from one place.

Why this exists
---------------
The judge was configurable two ways that didn't meet:

* ``SENTINEL_JUDGE_API_KEY`` / ``_BASE_URL`` / ``_MODEL`` env vars, read
  (and duplicated) inside ``scripts/validate_e2e.py`` and
  ``scripts/eval_generic_fraud.py``.
* a ``judge:`` section in ``.sentinel/config.yaml``
  (``config_ref`` + ``api_key_env``), writable from the web UI -- which
  **nothing ever read**, so setting it there silently had no effect and
  runs quietly fell back to self-judging.

This module is the single resolver both paths now go through.

Precedence (first hit wins)
---------------------------
1. **SENTINEL_JUDGE_* env vars** -- an explicit, per-run override beats
   stored config, matching how every other secret works here.
2. **The project's ``judge.config_ref``** -- an LLM YAML (e.g.
   ``configs/gateway.yaml``), with its key from ``judge.api_key_env``
   (or the config's own ``api_key_env``).
3. **Self-judge fallback** -- the actor's own credentials, so a missing
   judge never blocks a run. ALWAYS disclosed as ``self_judge: True``;
   an actor grading itself is not independent evidence.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

ENV_KEY = "SENTINEL_JUDGE_API_KEY"
ENV_URL = "SENTINEL_JUDGE_BASE_URL"
ENV_MODEL = "SENTINEL_JUDGE_MODEL"


class JudgeSpec:
    """A resolved judge: how to build it, and where it came from."""

    def __init__(self, base_url: str, model: str, api_key: str,
                 source: str, self_judge: bool, *,
                 verify_ssl: bool = True, models_url: str = "",
                 fallback_reason: str = "") -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.source = source          # "env" | "project_config" | "self_judge"
        self.self_judge = self_judge
        # I-02: the connection details that were silently dropped before --
        # a judge on the org's self-signed gateway needs verify_ssl=False, and
        # model discovery (/model-judge) needs the gateway's models_url.
        self.verify_ssl = verify_ssl
        self.models_url = models_url
        # I-03: when a CONFIGURED judge could not be used and this spec is
        # the self-judge fallback, why -- so the shell can say "your judge
        # is configured but its key is missing" instead of "no judge".
        self.fallback_reason = fallback_reason

    def meta(self) -> dict[str, Any]:
        """The disclosure block that goes into a validation record."""
        return {"model": self.model, "endpoint": self.base_url,
                "self_judge": self.self_judge, "source": self.source,
                "fallback_reason": self.fallback_reason}

    def build(self):
        from sentinel.llm.client import LLMClient
        return LLMClient(base_url=self.base_url, model=self.model,
                         api_key=self.api_key, verify_ssl=self.verify_ssl,
                         models_url=self.models_url)


# I-03: the reason the most recent configured-judge resolution fell
# through (blank when it didn't). Read by resolve_judge to annotate the
# self-judge fallback it hands back.
_last_fallback_reason = ""


def _from_master_config() -> Optional[JudgeSpec]:
    """main_config.yaml's `judge:` block -- same three-shape scheme as the
    main actor LLM (sentinel.llm.client.LLMConfig.from_master_config),
    independent of the `llm:` block so the judge can run a different
    method/model than the actor. Blank/absent `judge.method` -> None (falls
    through to self-judge), not an error -- an unconfigured judge is a
    normal, expected state. Checked before the older flat
    SENTINEL_JUDGE_API_KEY/_BASE_URL/_MODEL env vars below."""
    global _last_fallback_reason
    try:
        from sentinel.llm.client import LLMConfig, _load_master_config
        if not (_load_master_config().get("judge") or {}).get("method"):
            return None
        cfg = LLMConfig.from_master_config("judge")
        return JudgeSpec(cfg.base_url, cfg.model, cfg.api_key,
                         source="main_config", self_judge=False,
                         verify_ssl=cfg.verify_ssl, models_url=cfg.models_url)
    except Exception as exc:  # noqa: BLE001 -- never block a run on config
        _last_fallback_reason = f"main_config.yaml judge block: {exc}"
        log.warning("main_config.yaml judge.method set but could not "
                    "resolve: %s. Falling through.", exc)
        return None


def _from_env_flat() -> Optional[JudgeSpec]:
    """Older, flat env-var scheme: all three of SENTINEL_JUDGE_API_KEY/
    _BASE_URL/_MODEL set directly. Kept for backward compatibility --
    prefer main_config.yaml's judge: block (see _from_master_config) for
    new setups."""
    key, url, model = (os.environ.get(ENV_KEY), os.environ.get(ENV_URL),
                       os.environ.get(ENV_MODEL))
    if key and url and model:
        return JudgeSpec(url, model, key, source="env", self_judge=False)
    # Partial config is a likely mistake -- all three are required, and
    # silently self-judging because one is missing is exactly the trap
    # this module exists to surface.
    if any((key, url, model)):
        missing = [n for n, v in ((ENV_KEY, key), (ENV_URL, url), (ENV_MODEL, model))
                   if not v]
        log.warning("Judge env vars partially set -- missing %s. "
                    "All three are required; falling through.", ", ".join(missing))
    return None


def _from_env() -> Optional[JudgeSpec]:
    return _from_master_config() or _from_env_flat()


def _from_project(project_or_dotdir) -> Optional[JudgeSpec]:
    """Build a judge from the project's ``judge.config_ref`` LLM YAML."""
    global _last_fallback_reason
    if project_or_dotdir is None:
        return None
    try:
        from sentinel.project.settings import ProjectSettings
        settings = ProjectSettings.load(project_or_dotdir)
        ref = settings.get("judge", "config_ref")
        if not ref:
            return None
        path = Path(ref)
        if not path.is_absolute():
            # Refs are stored repo-relative (that's what the UI writes).
            root = Path(settings.dotdir).parent
            path = (root / ref) if (root / ref).exists() else Path(ref)
        if not path.exists():
            log.warning("judge.config_ref=%r does not exist -- falling through.", ref)
            return None

        # Parse the YAML directly rather than via LLMClient.from_config():
        # the client's __init__ calls resolve_api_key(), which RAISES when
        # api_key_env names an unset var -- that would turn "no judge key"
        # into an exception instead of a clean, logged fall-through here.
        from sentinel.llm.client import LLMConfig
        # section="judge": if this config_ref points at one of the
        # method-based configs (configs/self|gateway|api.yaml), resolve it
        # against main_config.yaml's judge: block, not the actor's llm: block.
        cfg = LLMConfig.from_yaml(path, section="judge")
        # Key: judge.api_key_env wins (explicitly chosen for the judge),
        # else whatever the LLM yaml itself names, else its literal key.
        key = settings.secret("judge", "api_key_env")
        if not key and cfg.api_key_env:
            key = os.environ.get(cfg.api_key_env, "")
        if not key:
            key = cfg.api_key if cfg.api_key != "EMPTY" else ""
        if not key:
            _last_fallback_reason = (f"judge.config_ref={ref!r} has no API "
                                     "key (judge.api_key_env / the config's "
                                     "api_key_env)")
            log.warning("judge.config_ref=%r resolved but no API key was found "
                        "(judge.api_key_env / the config's api_key_env) -- "
                        "falling through.", ref)
            return None
        return JudgeSpec(cfg.base_url, cfg.model, key,
                         source="project_config", self_judge=False,
                         verify_ssl=cfg.verify_ssl, models_url=cfg.models_url)
    except Exception as exc:  # noqa: BLE001 -- never block a run on config
        _last_fallback_reason = f"project judge.config_ref: {exc}"
        log.warning("Could not resolve judge from project config: %s", exc)
        return None


def _default_dotdir() -> Optional[Path]:
    """The project config to consult when a caller names none.

    The eval/benchmark scripts are standalone (no project object), but are
    run from the repo root -- which is exactly where `sentinel shell` /
    the web UI write `.sentinel/config.yaml`. So "set the judge in the UI,
    then run a benchmark" works without passing anything.
    """
    d = Path.cwd() / ".sentinel"
    return d if d.is_dir() else None


def _self_judge(actor_base_url: str, actor_model: str, actor_key: str, *,
                verify_ssl: bool = True, models_url: str = "",
                fallback_reason: str = "") -> JudgeSpec:
    return JudgeSpec(actor_base_url, actor_model, actor_key,
                     source="self_judge", self_judge=True,
                     verify_ssl=verify_ssl, models_url=models_url,
                     fallback_reason=fallback_reason)


def resolve_judge(
    project_or_dotdir=None,
    *,
    actor_base_url: str = "https://api.deepseek.com",
    actor_model: str = "",
    actor_key: str = "",
    actor_verify_ssl: bool = True,
    actor_models_url: str = "",
) -> JudgeSpec:
    """Resolve the judge. See module docstring for precedence.

    Never raises and never returns None: a run always gets a judge, and an
    unconfigured one is a DISCLOSED self-judge rather than a crash. When a
    judge WAS configured but could not be used, the self-judge spec carries
    `fallback_reason` saying why (I-03) -- callers should show it.

    `project_or_dotdir=None` consults ./.sentinel when it exists.
    """
    global _last_fallback_reason
    _last_fallback_reason = ""
    if project_or_dotdir is None:
        project_or_dotdir = _default_dotdir()
    spec = _from_env() or _from_project(project_or_dotdir)
    if spec is None:
        model = actor_model or os.environ.get("SENTINEL_DEEPSEEK_MODEL", "deepseek-chat")
        key = actor_key or os.environ.get("SENTINEL_DEEPSEEK_API_KEY", "")
        # Self-judge talks to the ACTOR's endpoint, so it inherits the
        # actor's connection details (I-02) -- the actor already proved
        # they work.
        spec = _self_judge(actor_base_url, model, key,
                           verify_ssl=actor_verify_ssl,
                           models_url=actor_models_url,
                           fallback_reason=_last_fallback_reason)

    # /model-judge <id> in the shell: a MODEL-only override on top of
    # whatever endpoint the source above resolved (config_ref or
    # self-judge) -- doesn't change source/self_judge, just which model
    # gets asked at that endpoint. Never overrides an env-sourced spec
    # (SENTINEL_JUDGE_MODEL already covers that case with top precedence).
    if spec.source != "env" and project_or_dotdir is not None:
        try:
            from sentinel.project.settings import ProjectSettings
            override = ProjectSettings.load(project_or_dotdir).get("judge", "model")
            if override.strip():
                spec.model = override.strip()
        except Exception:  # noqa: BLE001 -- never let an override block a run
            pass
    return spec
