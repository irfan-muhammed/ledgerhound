"""
ledgerhound.telemetry — Ledgerhound-own usage telemetry
=================================================

Ledgerhound-owned module (lives in the ledgerhound repo; the llmfe-suite wheel
packages it verbatim — no code is shared with or read from other packages,
and nothing here reads another package's identity or state).

TelemetryClient
    Fire-and-forget HTTP POST beacon to the collector URL from
    LEDGERHOUND_TELEMETRY_URL. Never raises, never blocks a run.
    Disabled with LEDGERHOUND_TELEMETRY_DISABLED=true or when the URL is unset.

Attribution user
    Resolved from Ledgerhound's OWN identity (ledgerhound.identity.current_user()):
    $LEDGERHOUND_USER / identity.user -> OS login -> "anonymous".
    Ledgerhound's work login gate (identity.ensure_login) is
    upstream behavior and is untouched by this module.

Email domain is baked at build time: setup.py generates ledgerhound/_domain.pyx
from the LLMFE_EMAIL_DOMAIN env var; Cython compiles it into _domain.so.
In dev (plain source, no .so) the module falls back to the env var so
unit tests can run without a compiled wheel.
"""

import json
import logging
import os
import re
import socket
import threading
import urllib.request
import uuid
from datetime import datetime, timezone

_log = logging.getLogger("ledgerhound")

_TELEMETRY_URL_ENV = "LEDGERHOUND_TELEMETRY_URL"
_TELEMETRY_DISABLED_ENV = "LEDGERHOUND_TELEMETRY_DISABLED"


def _allowed_domain() -> str:
    """Resolve the baked email domain (informational in Ledgerhound; the
    enforcement gate is upstream identity.ensure_login)."""
    try:
        from . import _domain  # compiled at build time
        domain = getattr(_domain, "ALLOWED_EMAIL_DOMAIN", None)
        if domain:
            return str(domain).lstrip("@")
    except Exception:
        pass
    return os.environ.get("LLMFE_EMAIL_DOMAIN", "example.com").lstrip("@").strip() or "example.com"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class TelemetryClient:
    """Fire-and-forget HTTP POST telemetry for Ledgerhound.

    Never raises, never blocks (daemon thread, 5s timeout).
    Disabled when LEDGERHOUND_TELEMETRY_DISABLED is truthy or URL is unset.
    """

    TIMEOUT_SECONDS = 5

    def __init__(self, user: str, product: str = "ledgerhound",
                 suite_version: str = "", product_version: str = ""):
        self._url = os.environ.get(_TELEMETRY_URL_ENV, "").strip()
        self._disabled = os.environ.get(
            _TELEMETRY_DISABLED_ENV, "").strip().lower() in ("1", "true", "yes", "on")
        self._user = user
        self._product = product
        self._suite_version = suite_version
        self._product_version = product_version
        self._hostname = os.environ.get("HOSTNAME") or socket.gethostname()
        self._run_id = _new_run_id()

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def enabled(self) -> bool:
        return bool(self._url) and not self._disabled_flag()

    def _disabled_flag(self) -> bool:
        # Same set as __init__ -- "on" used to disable at construction but
        # not here, so events still went out.
        return os.environ.get(
            _TELEMETRY_DISABLED_ENV, "").strip().lower() in ("1", "true", "yes", "on")

    def fire_event(self, event_type: str, payload=None) -> None:
        if not self.enabled:
            return
        event = {
            "event_type": event_type,
            "product": self._product,
            "run_id": self._run_id,
            "user": self._user,
            "suite_version": self._suite_version,
            "product_version": self._product_version,
            "hostname": self._hostname,
            "timestamp": _utcnow_iso(),
        }
        event.update(payload or {})
        threading.Thread(target=self._post, args=(event,), daemon=True).start()

    def _post(self, event: dict) -> None:
        try:
            data = json.dumps(event, default=str).encode("utf-8")
            req = urllib.request.Request(
                self._url, data=data,
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=self.TIMEOUT_SECONDS):
                pass
        except Exception as e:  # telemetry must never break a run
            _log.debug(f"telemetry send failed (ignored): {e}")


def _new_run_id() -> str:
    return uuid.uuid4().hex


# Module-level singleton: set once per Ledgerhound run (by core.Ledgerhound),
# then read by deeper call sites without threading state through.
_global_client = None


def set_global_client(client) -> None:
    global _global_client
    _global_client = client


def get_global_client():
    return _global_client


def fire_global(event_type: str, payload: dict) -> None:
    client = get_global_client()
    if client is not None:
        client.fire_event(event_type, payload)