"""
sentinel.shell.minio_cmd
==========================

The shell's `/minio` command: see and set the team object store's
credentials without editing files or exporting variables.

    /minio                          status + a live connection check
    /minio --key                    prompt for the secret (hidden input)
    /minio --key <secret>           set it inline (visible in scrollback --
                                    the prompt form is safer)
    /minio --access-key <key>       set the access key
    /minio --clear                  forget what /minio saved

A secret is CHECKED against the store before it is saved: a rejected one is
never stored, so a typo cannot silently replace a working key. If the store
cannot be reached (e.g. off the org's network), the key is saved anyway and
the reply says it is unverified.

Values are never echoed or logged. They are stored by
sentinel.llm.credentials in ~/.sentinel/credentials.json (owner-only), and
a handled slash command never reaches the model, the transcript or
events.jsonl.
"""

from __future__ import annotations

import shlex
from typing import Callable, Optional

PROMPT = "__prompt__"


def parse_minio_args(text: str) -> dict:
    """{"key": str|PROMPT|None, "access_key": str|PROMPT|None, "clear": bool}.
    Raises ValueError with a plain message on anything unrecognised."""
    try:
        tokens = shlex.split(text or "")
    except ValueError:
        tokens = (text or "").split()
    out = {"key": None, "access_key": None, "clear": False}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in ("--key", "--secret", "--secret-key", "--access-key"):
            field = "access_key" if t == "--access-key" else "key"
            nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
            if nxt and not nxt.startswith("--"):
                out[field] = nxt
                i += 2
            else:
                out[field] = PROMPT
                i += 1
        elif t == "--clear":
            out["clear"] = True
            i += 1
        else:
            raise ValueError(
                f"unknown /minio option {t!r}. Use: /minio, /minio --key, "
                "/minio --access-key <key>, /minio --clear")
    return out


def check_store(settings: dict) -> tuple[bool, str, str, str]:
    """(ok, code, message, fix) for one bucket check against `settings`
    (the dict shape sentinel.io.minio.shared_settings returns). Uses the
    same error classifier the tokenizer loader uses, so the two can never
    describe the same failure differently."""
    from sentinel.llm import tokenizer as T
    if not settings.get("endpoint"):
        return (False, T.NOT_CONFIGURED, "no object store endpoint is configured.",
                "set minio.endpoint in main_config.yaml (e.g. "
                "objectstore.example.com:9000).")
    if not settings.get("access_key"):
        return (False, T.AUTH_FAILED, "the access key is not set.",
                "set minio.access_key in main_config.yaml, or run "
                "/minio --access-key <key>.")
    if not settings.get("secret_key"):
        return (False, T.AUTH_FAILED, "the secret key is not set.",
                "run /minio --key and paste it at the prompt.")
    s = dict(settings)
    s.setdefault("prefix", "tokenizers")
    try:
        client = T._client(s)
        exists = client.bucket_exists(s["bucket"])
    except T._StoreError as e:
        return False, e.code, e.message, e.fix
    except Exception as e:  # noqa: BLE001
        err = T._classify_store_error(e, s, f"bucket {s['bucket']}")
        return False, err.code, err.message, err.fix
    if not exists:
        return (False, T.STORE_UNREACHABLE,
                f"bucket '{s['bucket']}' does not exist at {s['endpoint']} or "
                "is not visible to this access key.",
                "check minio.bucket in main_config.yaml.")
    return True, T.OK, (f"{s['endpoint']} reachable, bucket '{s['bucket']}' "
                        "visible with these credentials."), ""


def _masked(value: str) -> str:
    return "set" if value else "NOT set"


def run_minio_command(args: str, *,
                      say: Callable[[str], None],
                      warn: Callable[[str], None],
                      ask_secret: Callable[[str], str],
                      settings_fn: Optional[Callable[[], dict]] = None,
                      check_fn: Optional[Callable[[dict], tuple]] = None,
                      on_changed: Callable[[], None] = lambda: None) -> None:
    """Run `/minio <args>`. Never raises for ordinary misuse; prints
    through `say` (normal) / `warn` (needs attention). `settings_fn` and
    `check_fn` are injectable for tests."""
    from sentinel.io import minio as M
    settings_fn = settings_fn or M.shared_settings
    check_fn = check_fn or check_store
    try:
        opts = parse_minio_args(args)
    except ValueError as e:
        warn(str(e))
        return

    if opts["clear"]:
        M.clear_saved_credentials()
        say("forgot the access key and secret saved with /minio. Environment "
            "variables and main_config.yaml values, if any, still apply.")
        on_changed()
        return

    if opts["access_key"] is not None:
        value = opts["access_key"]
        if value == PROMPT:
            value = (ask_secret("MinIO access key: ") or "").strip()
        if not value:
            warn("no access key entered -- nothing changed.")
            return
        M.save_access_key(value)
        say("access key saved (~/.sentinel/credentials.json, owner-only).")
        if opts["key"] is None:
            on_changed()

    if opts["key"] is not None:
        secret = opts["key"]
        if secret == PROMPT:
            secret = (ask_secret("MinIO secret key (input hidden): ") or "").strip()
        if not secret:
            warn("no secret entered -- nothing changed.")
            return
        trial = dict(settings_fn())
        trial["secret_key"] = secret
        ok, code, message, fix = check_fn(trial)
        if ok:
            M.save_secret(secret)
            say(f"secret verified -- {message} Saved to "
                "~/.sentinel/credentials.json (owner-only); it is not shown "
                "or logged anywhere.")
            if settings_fn().get("secret_source") == "main_config.yaml":
                warn("note: main_config.yaml also sets minio.secret_key, and that "
                     "value is the one used. Blank it there to use the key saved now.")
            on_changed()
            return
        if code == "auth_failed" and trial.get("endpoint") and trial.get("access_key"):
            warn(f"the store REJECTED this secret -- NOT saved. {message} "
                 "Check it (and the access key) and run /minio --key again.")
            return
        if code == "store_unreachable":
            M.save_secret(secret)
            warn(f"saved, but NOT verified: {message} It will be used as soon "
                 "as the store is reachable -- run /minio to re-check.")
            on_changed()
            return
        warn(f"not saved: {message} Fix: {fix}")
        return

    if opts["access_key"] is not None:
        return

    # plain `/minio`: status + live check
    s = settings_fn()
    lines = [
        "object store (main_config.yaml `minio:`):",
        f"  endpoint   : {s.get('endpoint') or '(not set)'}",
        f"  bucket     : {s.get('bucket')}",
        f"  access key : {_masked(s.get('access_key'))}"
        + (f" ({s.get('access_source')})" if s.get("access_key") else ""),
        f"  secret key : {_masked(s.get('secret_key'))}"
        + (f" ({s.get('secret_source')})" if s.get("secret_key") else ""),
    ]
    ok, code, message, fix = check_fn(s)
    if ok:
        say("\n".join(lines + [f"  connection : OK -- {message}"]))
    else:
        warn("\n".join(lines + [f"  connection : FAILED [{code}] -- {message}",
                                f"  fix        : {fix}"]))
