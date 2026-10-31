"""
sentinel.identity
=================

Tier 1 identity -- declared, not verified.

Resolves "who is running this" so every artifact, eval record, and memory write
can be attributed and namespaced by owner. Tier 1 trusts the ``SENTINEL_USER``
env var (set by the Kubeflow PodDefault); it is spoofable and that is acceptable
for attribution + namespacing. Tier 2 (OIDC device flow + MinIO STS) upgrades
this to verified identity later without changing call sites.

Resolution order:
  1. main_config.yaml's identity.user     (interactive/local use)
  2. SENTINEL_USER env var                (Kubeflow PodDefault -- the
                                            production path; unchanged)
  3. OS login name                        (local dev fallback)
  4. "anonymous"                          (last resort; never blocks a run)

Owner strings are slugified to a safe namespace token (lowercase, [a-z0-9_-]),
so they drop straight into MinIO key prefixes like ``data/{owner}/...``.
"""

from __future__ import annotations

import getpass
import json
import os
import re
from pathlib import Path
from typing import Callable

_SLUG_RE = re.compile(r"[^a-z0-9_-]+")
ANONYMOUS = "anonymous"

# Mandatory one-time login for interactive `sentinel shell` use (never for
# the Kubeflow production path -- that sets $SENTINEL_USER via PodDefault
# and is_declared() short-circuits before any of this runs). Any email
# address is accepted (restricted to $SENTINEL_EMAIL_DOMAIN when set,
# e.g. "example.com"); the part before '@' becomes the
# username. Stored once at _IDENTITY_FILE so it is never asked again on
# this machine.
_EMAIL_DOMAIN = os.environ.get("SENTINEL_EMAIL_DOMAIN", "").lstrip("@").strip()
EMAIL_RE = re.compile(
    r"^[a-z0-9._%+\-]+@"
    + (re.escape(_EMAIL_DOMAIN) if _EMAIL_DOMAIN else r"[a-z0-9.\-]+\.[a-z]{2,}")
    + r"$",
    re.IGNORECASE,
)
_IDENTITY_FILE = Path.home() / ".sentinel" / "identity.json"


def _slugify(name: str) -> str:
    slug = _SLUG_RE.sub("-", (name or "").strip().lower()).strip("-")
    return slug or ANONYMOUS


def current_user() -> str:
    """Return the slugified owner for the current process.

    Never raises and never returns empty -- falls back to the OS user, then
    ``anonymous``, so attribution degrades gracefully instead of blocking work.
    """
    from sentinel.config import get_str
    declared = get_str("identity", "user", env="SENTINEL_USER")
    if declared and declared.strip():
        return _slugify(declared)
    # The work login saved by the first `sentinel shell` on this
    # machine. Only the shell used to read it, so `sentinel doctor` and every
    # other command fell back to the OS login name.
    stored = _load_stored_user()
    if stored:
        return _slugify(stored)
    try:
        login = getpass.getuser()
        if login and login.strip():
            return _slugify(login)
    except Exception:
        pass
    return ANONYMOUS


def identity_info() -> dict:
    """{user, email, source} for status displays (doctor, banner)."""
    from sentinel.config import get_str
    cfg = get_str("identity", "user").strip()
    if cfg:
        return {"user": _slugify(cfg), "email": cfg if "@" in cfg else "",
                "source": "main_config.yaml identity.user"}
    env = os.environ.get("SENTINEL_USER", "").strip()
    stored = _load_stored_record()
    if env:
        email = stored.get("email", "") if stored.get("user") == _slugify(env) else ""
        return {"user": _slugify(env), "email": email, "source": "SENTINEL_USER"}
    if stored.get("user"):
        return {"user": _slugify(stored["user"]), "email": stored.get("email", ""),
                "source": f"login saved in {_IDENTITY_FILE}"}
    return {"user": current_user(), "email": "", "source": "OS login (no org login yet)"}


def _load_stored_record() -> dict:
    try:
        d = json.loads(_IDENTITY_FILE.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def is_declared() -> bool:
    """True when identity.user (main_config.yaml) or SENTINEL_USER is set
    (vs an OS/anonymous fallback)."""
    from sentinel.config import get_str
    return bool(get_str("identity", "user", env="SENTINEL_USER").strip())


def _load_stored_user() -> str:
    try:
        d = json.loads(_IDENTITY_FILE.read_text(encoding="utf-8"))
        return str(d.get("user") or "").strip()
    except Exception:
        return ""


def _store_user(username: str, email: str) -> None:
    try:
        _IDENTITY_FILE.parent.mkdir(parents=True, exist_ok=True)
        _IDENTITY_FILE.write_text(
            json.dumps({"user": username, "email": email}, indent=2),
            encoding="utf-8")
    except Exception:  # noqa: BLE001 -- a failed write just means we'll ask again
        pass


def ensure_login(ask: Callable[[str], str] = input,
                      say: Callable[[str], None] = print) -> str:
    """Mandatory login gate for interactive `sentinel shell` startup.

    Order:
      1. Identity already declared (main_config.yaml's identity.user, or
         $SENTINEL_USER -- the Kubeflow PodDefault production path) --
         never prompted, return it as-is.
      2. A username was stored from a PRIOR login on this machine -- reuse
         it silently. We never ask twice.
      3. Otherwise: loop asking for a work email until one
         validates, take the part before '@' as the username, store it,
         and set $SENTINEL_USER for the rest of this process so every
         existing current_user() call site picks it up unchanged.
    """
    if is_declared():
        return current_user()
    stored = _load_stored_user()
    if stored:
        os.environ["SENTINEL_USER"] = stored
        return stored
    while True:
        email = (ask("Login required -- enter your work email: ")
                 or "").strip()
        if EMAIL_RE.match(email):
            username = _slugify(email.split("@", 1)[0])
            _store_user(username, email.lower())
            os.environ["SENTINEL_USER"] = username
            say(f"Logged in as '{username}'. You will not be asked again on "
                "this machine.")
            return username
        say("Invalid email -- must be a real work address "
            "(e.g. name@example.com). Try again.")


def owner_prefix(stream_path: str, owner: str | None = None) -> str:
    """Build an owner-scoped (private-tier) key prefix.

    >>> owner_prefix("eval")            # -> "<user>/eval"  (private tier)
    """
    o = owner or current_user()
    return f"{o}/{stream_path.lstrip('/')}"
