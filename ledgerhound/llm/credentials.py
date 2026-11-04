"""
ledgerhound.llm.credentials
===========================

Ask-once, store, never-ask-again API keys for main_config.yaml's
method-based ``llm``/``judge`` sections (``self`` needs none; ``gateway``
and ``api`` do). Mirrors ledgerhound.identity's org-login pattern: prompted
only when a key is genuinely missing everywhere else (main_config.yaml,
this store, or the legacy config's own ``api_key_env``), stored at
``~/.ledgerhound/credentials.json``, and reused silently on every later
`ledgerhound shell` start.

Keyed by ``section.method`` (e.g. ``llm.gateway``, ``judge.api``) so the
actor and judge -- or a self-hosted vs. hosted method within the same
section -- never collide.

On a genuine API-key AUTH failure (401/403 from the gateway --
``ledgerhound.llm.client.LLMAuthError``), the CLI clears the stored key here
so the NEXT prompt (or the next `ledgerhound shell` start) asks fresh,
instead of silently retrying with a key already known to be wrong.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

CRED_FILE = Path.home() / ".ledgerhound" / "credentials.json"


def _key_id(section: str, method: str) -> str:
    return f"{section}.{method}"


def _load_all() -> dict:
    try:
        return json.loads(CRED_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 -- no store yet is not an error
        return {}


def _save_all(data: dict) -> None:
    try:
        CRED_FILE.parent.mkdir(parents=True, exist_ok=True)
        CRED_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        try:
            # I-18: plaintext bearer tokens -- owner-only, like
            # .ledgerhound/secrets.env. No-op on Windows (ACLs), real on the
            # Linux deploy where home dirs can be shared mounts.
            import os
            os.chmod(CRED_FILE, 0o600)
        except OSError:
            pass
    except Exception:  # noqa: BLE001 -- a failed write just means we'll ask again
        pass


def load_key(section: str, method: str) -> str:
    data = _load_all()
    key = str(data.get(_key_id(section, method), "") or "")
    if key:
        return key
    return ""


def store_key(section: str, method: str, key: str) -> None:
    data = _load_all()
    data[_key_id(section, method)] = key
    _save_all(data)


def clear_key(section: str, method: str) -> None:
    data = _load_all()
    if data.pop(_key_id(section, method), None) is not None:
        _save_all(data)


def resolve_method(cfg_path, section: str = "llm"):
    """The `method` (self/gateway/api) a config resolves to, or None for a
    legacy literal-style config (its own base_url/model/api_key_env, never
    touches main_config.yaml). Shared by ensure_configured and the CLI's
    LLMAuthError handler -- both need to know which credentials-store slot
    a given cfg_path's key lives in."""
    import yaml
    from ledgerhound.llm.client import resolve_config_ref
    try:
        with open(resolve_config_ref(cfg_path), encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:  # noqa: BLE001
        return None
    llm_section = data.get(section, data) if isinstance(data, dict) else {}
    method = (llm_section or {}).get("method")
    if not method:
        return None
    from ledgerhound.llm.client import normalise_method
    return normalise_method(str(method))


def ensure_configured(cfg_path, section: str = "llm", *,
                      ask: Callable[[str], str] = input,
                      say: Callable[[str], None] = print) -> None:
    """Before constructing an LLMClient from `cfg_path`, make sure an API
    key is available if the resolved method needs one -- prompting (once,
    stored) only when main_config.yaml's own field is blank AND nothing
    was stored here yet. A no-op for a legacy literal-style config or for
    method == "self" (no key needed)."""
    method = resolve_method(cfg_path, section)
    if method is None:
        # An older literal-style config (e.g. configs/gateway.yaml) that
        # names an env var for its key. Those were never asked: startup
        # died later with "env var is empty". Ask once, save, and put it in
        # the environment for this session.
        _ensure_env_key(cfg_path, section, ask=ask, say=say)
        return
    if method not in ("gateway", "api"):
        return
    from ledgerhound.llm.client import _load_master_config
    mc = _load_master_config()
    sect = mc.get(section) or {}
    method_cfg = sect.get(method) or {}
    configured = str(method_cfg.get("api_key") or "").strip()
    ensure_api_key(section, method, configured_key=configured, ask=ask, say=say)


def _ensure_env_key(cfg_path, section: str, *, ask, say) -> None:
    import os
    import yaml
    from ledgerhound.llm.client import resolve_config_ref
    try:
        with open(resolve_config_ref(cfg_path), encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:  # noqa: BLE001
        return
    sect = data.get(section, data) if isinstance(data, dict) else {}
    env = str((sect or {}).get("api_key_env") or "").strip()
    literal = str((sect or {}).get("api_key") or "").strip()
    if not env or os.environ.get(env) or literal not in ("", "$ENV", "EMPTY"):
        return
    key = ensure_api_key(section, f"env:{env}", ask=ask, say=say)
    if key:
        os.environ[env] = key


def ensure_section_configured(section: str, *,
                              ask: Callable[[str], str] = input,
                              say: Callable[[str], None] = print) -> None:
    """ensure_configured for a section driven purely by main_config.yaml's
    own `<section>.method` (the judge, which has no config file of its
    own). No-op when the method is blank/self, or the block already holds
    a key. (I-03)"""
    from ledgerhound.llm.client import _load_master_config, normalise_method
    mc = _load_master_config()
    sect = mc.get(section) or {}
    method = normalise_method(str(sect.get("method") or ""))
    if method not in ("gateway", "api"):
        return
    method_cfg = sect.get(method) or {}
    configured = str(method_cfg.get("api_key") or "").strip()
    ensure_api_key(section, method, configured_key=configured, ask=ask, say=say)


def ensure_api_key(section: str, method: str, *, configured_key: str = "",
                   ask: Callable[[str], str] = input,
                   say: Callable[[str], None] = print) -> str:
    """Return an API key for `section.method` -- from main_config.yaml
    (`configured_key`) if non-blank, else from a prior stored login, else
    prompt once and store. Never call this for method == "self" (no key
    needed); callers decide that, this module has no opinion on methods."""
    if configured_key.strip():
        return configured_key.strip()
    stored = load_key(section, method)
    if stored:
        return stored
    label = method[4:] if method.startswith("env:") else f"{section}.{method}"
    key = (ask(f"LLM API key for {label} is not set in main_config.yaml -- paste it "
               "(hidden; saved to ~/.ledgerhound/credentials.json, asked only once): ")
           or "").strip()
    if key:
        store_key(section, method, key)
        say(f"API key for {section}.{method} stored -- you will not be "
            "asked again unless it stops working.")
    return key
