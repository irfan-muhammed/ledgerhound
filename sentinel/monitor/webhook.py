"""
sentinel.monitor.webhook
========================

HTTP POST emitter for Drift Monitor alerts.

Design rules
------------
- **Best-effort.**  A failed webhook never crashes the monitor -- we log
  the failure and move on.  The MonitorRunResult records the
  attempt-vs-success counts.
- **Header expansion.**  Values referenced as ``$VARNAME`` inside a
  WebhookConfig's ``headers`` are resolved from the calling process's
  environment at emit time.  This keeps secrets out of YAML.
- **Bounded timeout.**  Default 10s per call; no retries by default
  (the next monitor cycle is itself the retry).
- **Filtering.**  ``WebhookConfig.on_severity`` and ``on_event_types``
  pre-filter so we don't send INFO-level chatter to a paging webhook.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

from sentinel.monitor.config import WebhookConfig
from sentinel.monitor.result import AlertEvent

log = logging.getLogger(__name__)


_ENV_REF = re.compile(r"\$([A-Z_][A-Z0-9_]*)")


def _expand_env(value: str) -> str:
    """Replace ``$VAR`` references with os.environ lookups."""
    if not isinstance(value, str):
        return value
    def _sub(m):
        return os.environ.get(m.group(1), m.group(0))
    return _ENV_REF.sub(_sub, value)


class WebhookEmitter:
    """Best-effort HTTP POST emitter.

    Usage::

        emitter = WebhookEmitter()
        ok = emitter.emit(webhook_config, alert_event)
    """

    def __init__(self, default_timeout: float = 10.0) -> None:
        self.default_timeout = float(default_timeout)

    # -- Filtering -----------------------------------------------------

    @staticmethod
    def matches(webhook: WebhookConfig, alert: AlertEvent) -> bool:
        """Return True if this alert should be delivered to this webhook."""
        if webhook.on_severity and alert.severity not in webhook.on_severity:
            return False
        if webhook.on_event_types and alert.event_type not in webhook.on_event_types:
            return False
        return True

    # -- Emission ------------------------------------------------------

    def emit(
        self,
        webhook: WebhookConfig,
        alert: AlertEvent,
    ) -> bool:
        """POST a single alert to a single webhook. Returns success bool."""
        if not self.matches(webhook, alert):
            return False
        try:
            import httpx
        except Exception as exc:
            log.warning("httpx not available, skipping webhook emit: %s", exc)
            return False

        headers = {k: _expand_env(v) for k, v in (webhook.headers or {}).items()}
        timeout = float(webhook.timeout or self.default_timeout)
        payload = self._payload(alert)

        try:
            resp = httpx.post(
                webhook.url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            ok = 200 <= resp.status_code < 400
            if not ok:
                log.warning(
                    "Webhook %s returned %s: %s",
                    webhook.url, resp.status_code, resp.text[:200],
                )
            return ok
        except Exception as exc:
            log.warning("Webhook %s failed: %s", webhook.url, exc)
            return False

    def emit_many(
        self,
        webhooks: list[WebhookConfig],
        alerts: list[AlertEvent],
    ) -> dict[str, int]:
        """Emit each alert to every matching webhook.

        Returns a dict with keys: ``attempted``, ``succeeded``, ``failed``.
        Also mutates each AlertEvent's ``delivered_to`` list to record
        successful delivery URLs.
        """
        attempted = 0
        succeeded = 0
        failed = 0
        for alert in alerts:
            for wh in webhooks:
                if not self.matches(wh, alert):
                    continue
                attempted += 1
                if self.emit(wh, alert):
                    succeeded += 1
                    alert.delivered_to.append(wh.url)
                else:
                    failed += 1
        return {
            "attempted": attempted,
            "succeeded": succeeded,
            "failed":    failed,
        }

    # -- Payload shape -------------------------------------------------

    @staticmethod
    def _payload(alert: AlertEvent) -> dict[str, Any]:
        """Build the JSON body for a webhook POST.

        We default to a Slack-friendly shape but include the full alert
        dict under ``alert`` so generic webhooks have full context.
        """
        text = (
            f"[{alert.severity}] {alert.event_type} -- {alert.dataset_name}/"
            f"{alert.period}: {alert.claim}"
        )
        return {
            "schema": "sentinel_alert/v1",
            "text":   text,                  # Slack-compatible
            "alert":  alert.to_dict(),       # full record
        }
