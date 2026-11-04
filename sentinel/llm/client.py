"""
sentinel.llm.client
===================

A flexible LLM client that talks to **any OpenAI-compatible API**.

Supported endpoints
-------------------
* vLLM / SGLang on local GPUs  (``http://gpu:8000/v1``)
* OpenAI                       (``https://api.openai.com/v1``)
* Together.ai, Groq, Fireworks, etc.

Features
--------
* Automatic retry with exponential back-off.
* Thinking-mode extraction (``<think>`` blocks, ``reasoning_content``).
* Streaming support.
* Full call logging for audit trail.

Usage::

    # Local vLLM
    client = LLMClient(base_url="http://gpu:8000/v1",
                        model="Qwen3-235B-A22B-Thinking-2507")

    # From YAML config
    client = LLMClient.from_config("configs/llm_config.yaml")

    resp = client.generate([{"role": "user", "content": "Hello"}])
    print(resp.content)
"""

from __future__ import annotations

import json, logging, os, re, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

import httpx, yaml

log = logging.getLogger(__name__)


# Recover tool calls that a gateway emitted as raw text instead of parsing them
# into OpenAI `tool_calls`. This happens when the inference server (vLLM /
# TensorRT-LLM / etc.) is NOT started with a model-specific tool-call parser
# (e.g. Kimi-K2 needs `--tool-call-parser kimi_k2`): the model's native
# tool-call tokens land in `content` verbatim. The proper fix is server-side,
# but recovering here keeps the agent working instead of looking silent.
_KIMI_TOOLCALL_RE = re.compile(
    r"<\|tool_call_begin\|>\s*(?:functions\.)?([A-Za-z0-9_\-]+)\s*(?::\s*\d+)?\s*"
    r"<\|tool_call_argument_begin\|>\s*(\{.*?\})\s*<\|tool_call_end\|>",
    re.DOTALL,
)
_KIMI_SECTION_RE = re.compile(
    r"<\|tool_calls_section_begin\|>.*?<\|tool_calls_section_end\|>", re.DOTALL)
_HERMES_TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# Perfectly identify an API-key problem so it can be handled distinctly
# from a transient error: 401/403 are unambiguous; a few gateways return
# 400 with a phrase instead of the "correct" status code, so the text is
# checked too. Deliberately NOT matched: "rate limit" (a real key, just
# throttled -- retrying is the right response, not re-prompting for a key).
_AUTH_ERROR_PHRASES = (
    "invalid api key", "incorrect api key", "invalid_api_key",
    "unauthorized", "authentication failed", "invalid bearer token",
    "api key not found", "missing api key", "invalid x-api-key",
)


def normalise_method(method: str) -> str:
    """`llm.method` as compared everywhere: trimmed, lower-case."""
    return (method or "").strip().lower()


def resolve_config_ref(ref: str | Path) -> str:
    """Find an LLM config file named by a repo-relative reference.

    `configs/gateway.yaml` (the default, and what projects save) is relative
    to the Sentinel checkout. Launched from any other folder, a plain
    `Path(ref)` missed it and the shell refused to start. Order: the path
    as given (absolute, or relative to the current folder), then relative
    to the checkout Sentinel is installed from. Returns the first that
    exists, else `ref` unchanged so the caller's not-found message names
    what was asked for."""
    p = Path(ref).expanduser()
    if p.exists() or p.is_absolute():
        return str(p)
    install_root = Path(__file__).resolve().parents[2]   # <checkout>/sentinel/llm/client.py
    cand = install_root / p
    if cand.exists():
        return str(cand)
    return str(ref)


def merge_system_messages(messages: list[dict]) -> list[dict]:
    """One leading system message, as every chat template accepts.

    The shell keeps more than one `system` message in its history: the
    prompt, the ACTIVE DATA schema pin, and after compaction a history
    summary -- which can sit mid-conversation. Several chat templates
    (Gemma, some Qwen/Llama builds) reject or mangle a system message that
    is not first, or more than one. So the REQUEST gets them merged, in
    order, into one leading system message; the shell's own history is
    untouched. Messages with no system role pass through unchanged.
    """
    systems = [m for m in messages if m.get("role") == "system"]
    if len(systems) <= 1 and (not systems or messages[0] is systems[0]):
        return messages
    parts = [m.get("content") if isinstance(m.get("content"), str) else
             json.dumps(m.get("content")) for m in systems]
    merged = {"role": "system", "content": "\n\n".join(p for p in parts if p)}
    return [merged] + [m for m in messages if m.get("role") != "system"]


class LLMAuthError(Exception):
    """The gateway rejected the request because the API key is wrong,
    expired, or missing -- never worth retrying with the same key. Raised
    immediately (no retry loop) so a caller can re-prompt for a fresh key
    instead of burning `retry_attempts` on something that can't succeed."""


def _is_auth_error(status_code: int, text: str) -> bool:
    if status_code in (401, 403):
        return True
    low = (text or "").lower()
    return any(p in low for p in _AUTH_ERROR_PHRASES)


def _recover_tool_calls_from_text(content: str) -> tuple[list[dict], str]:
    """Parse tool calls out of raw text. Returns (tool_calls, cleaned_content).

    Conservative: only fires on explicit tool-call markers, so plain prose is
    never misread as a call. Handles the Kimi-K2 native token format and the
    Hermes/Qwen ``<tool_call>{...}</tool_call>`` JSON format.
    """
    pairs: list[tuple[str, str]] = []
    cleaned = content

    for m in _KIMI_TOOLCALL_RE.finditer(content):
        pairs.append((m.group(1), m.group(2)))
    if pairs:
        cleaned = _KIMI_SECTION_RE.sub("", content)
        cleaned = _KIMI_TOOLCALL_RE.sub("", cleaned)
    else:
        for m in _HERMES_TOOLCALL_RE.finditer(content):
            try:
                obj = json.loads(m.group(1))
            except Exception:  # noqa: BLE001
                continue
            name = obj.get("name")
            if name:
                args = obj.get("arguments", {})
                pairs.append((name, args if isinstance(args, str)
                              else json.dumps(args)))
        if pairs:
            cleaned = _HERMES_TOOLCALL_RE.sub("", content)

    calls = [{"id": f"call_{i}", "type": "function",
              "function": {"name": name, "arguments": args}}
             for i, (name, args) in enumerate(pairs)]
    return calls, cleaned.strip()


# Back-compat aliases: sentinel.eval.judge_config, sentinel.cli.main, and
# webui.session import `_load_master_config` / `_MASTER_CONFIG_CACHE` from
# THIS module by these names -- the real implementation now lives in
# sentinel.config (the one place EVERY main_config.yaml reader uses, not
# just LLM config), kept here as aliases so nothing else needed to change.
from sentinel.config import _CACHE as _MASTER_CONFIG_CACHE  # noqa: E402
from sentinel.config import load_master_config as _load_master_config  # noqa: E402


@dataclass
class LLMConfig:
    """All settings needed to reach an LLM endpoint.

    SSL verification
    ----------------
    ``verify_ssl=False`` corresponds to ``curl -k``. Use this for internal
    endpoints with self-signed certs (e.g. the org's gateway gateway). Never
    set False against a public internet endpoint.

    API-key resolution
    ------------------
    ``api_key`` may be:
      * a literal string (``"sk-..."``)
      * the special value ``"$ENV"`` plus ``api_key_env`` to load from env
      * left as ``"EMPTY"`` for vLLM-style auth-less endpoints
    """

    base_url: str = "http://localhost:8000/v1"
    # Model DISCOVERY (GET .../models -- sentinel.llm.catalog) often lives
    # behind a different URL than chat completions on gateways that split
    # the two (a separate model-registry endpoint, a different API
    # version path, etc.). Blank (the common case) -- list_models() just
    # hits "/models" on `base_url` as before. Set only when the gateway
    # genuinely needs a different one.
    models_url: str = ""
    model: str = "default"
    api_key: str = "EMPTY"          # vLLM doesn't need a real key
    api_key_env: str | None = None  # if set, resolve api_key from this env var
    verify_ssl: bool = True         # set False for self-signed internal certs
    temperature: float = 0.6
    max_tokens: int = 16384          # OUTPUT cap per reply (not the window)
    # The model's total context window. NOT sent in the request -- the
    # chat-completions API has no such field, and the server enforces its
    # own limit. It exists so the CLIENT can decide what to put in
    # `messages` before sending (see sentinel/shell/compressor.py). Without
    # it, compaction is driven by message count and is blind to how full
    # the window actually is.
    context_window: int = 256_000
    # True when main_config.yaml set context_window for this method --
    # the operator's number then wins over discovery and the table.
    context_window_explicit: bool = False
    top_p: float = 0.95
    timeout: int = 300              # seconds per request
    retry_attempts: int = 3
    retry_delay: float = 2.0
    enable_thinking: bool = True
    thinking_budget: int = 8192
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)

    def resolve_api_key(self) -> str:
        """Return the effective API key, reading env if configured."""
        if self.api_key_env:
            val = os.environ.get(self.api_key_env)
            if not val:
                raise ValueError(
                    f"api_key_env={self.api_key_env!r} but env var is empty. "
                    f"Set it before constructing LLMClient."
                )
            return val
        return self.api_key

    @classmethod
    def from_dict(cls, d: dict) -> LLMConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_master_config(cls, section: str = "llm", *,
                           overrides: dict | None = None,
                           path: str | Path | None = None) -> LLMConfig:
        """Build a config ENTIRELY from ``main_config.yaml`` -- no
        environment variables, no export-before-every-session.

        Replaces an earlier env-var-driven scheme (V3-058): a shell with
        many variables to set before every session is exactly the friction
        this exists to remove. ``main_config.yaml`` is gitignored (never
        pushed, holds real keys) with a committed ``main_config.example.yaml``
        template alongside it.

        ``section`` is ``"llm"`` for the main actor or ``"judge"`` for the
        evaluation judge -- each is independent, so the judge can run
        against a different method/model than the actor.

        Shape (see main_config.example.yaml for the full template)::

            llm:
              method: gateway        # self | gateway | api
              self:   {base_url, models_url, model, model_fallback}
              gateway: {base_url, models_url, model, model_fallback, api_key}
              api:    {base_url, models_url, model, model_fallback, api_key}
            # models_url is OPTIONAL on each -- blank means "GET /models on
            # base_url", which is right for almost every OpenAI-compatible
            # gateway; set it only when model DISCOVERY genuinely lives at a
            # different URL than chat completions.
            judge:
              method: ""             # blank -> self-judge (see judge_config.py)
              self / gateway / api: (same shape as above)

        ``self`` never needs ``api_key`` -- literal-key fields for it are
        ignored. ``overrides`` lets a YAML config (e.g. configs/gateway.yaml)
        that names a method still set its own STATIC fields (temperature,
        timeout, ...) on top -- see ``from_yaml``.
        """
        cfg = _load_master_config(path)
        sect = cfg.get(section) or {}
        method = normalise_method(str(sect.get("method") or "api"))
        if method not in ("self", "gateway", "api"):
            raise ValueError(
                f"main_config.yaml: {section}.method={method!r} is not one "
                f"of self|gateway|api (for an OpenAI-compatible gateway set "
                f"{section}.method: gateway and name its block `gateway:`)")
        method_cfg = sect.get(method) or {}
        fields: dict[str, Any] = {}
        base_url = str(method_cfg.get("base_url") or "").strip()
        if base_url:
            fields["base_url"] = base_url
        models_url = str(method_cfg.get("models_url") or "").strip()
        if models_url:
            fields["models_url"] = models_url
        model = (str(method_cfg.get("model") or "").strip()
                or str(method_cfg.get("model_fallback") or "").strip())
        if model:
            fields["model"] = model
        try:
            cw = int(method_cfg.get("context_window") or 0)
        except (TypeError, ValueError):
            cw = 0
        if cw > 0:
            fields["context_window"] = cw
            fields["context_window_explicit"] = True
        if method in ("gateway", "api"):
            key = str(method_cfg.get("api_key") or "").strip()
            if not key:
                # Ask-once-and-store fallback (sentinel.llm.credentials):
                # this classmethod has no I/O hooks to prompt with, so it
                # only READS whatever the CLI already collected and stored
                # (see sentinel.cli.main's ensure_llm_api_key, called before
                # LLMClient construction). Still raises if truly nothing is
                # on file anywhere -- a non-interactive caller (tests, a
                # script) must not silently hang waiting for a key.
                from sentinel.llm.credentials import load_key
                key = load_key(section, method)
            if not key:
                raise ValueError(
                    f"main_config.yaml: {section}.{method}.api_key is empty. "
                    f"Fill it in main_config.yaml (see main_config.example.yaml), "
                    f"or run `sentinel shell` interactively once to be prompted.")
            fields["api_key"] = key
        fields.update(overrides or {})
        return cls.from_dict(fields)

    @classmethod
    def from_yaml(cls, path: str | Path, section: str = "llm") -> LLMConfig:
        """Load static settings from `path`, then resolve base_url/model/
        api_key against ``main_config.yaml``'s ``{section}`` block (see
        `from_master_config`) -- the YAML never hardcodes a live
        endpoint/model/key, only the method it wants
        (`llm.method: self|gateway|api`) and static tuning (temperature,
        timeout, verify_ssl override, ...).

        Backward compatible: a YAML with NO `method` key and its own literal
        `base_url`/`model`/`api_key`/`api_key_env` (the old per-provider
        config style) is used exactly as written -- master-config
        resolution only kicks in when `method` is present.
        """
        with open(resolve_config_ref(path), encoding="utf-8") as f:
            data = yaml.safe_load(f)
        llm_section = data.get("llm", data)
        method = llm_section.get("method")
        if not method:
            return cls.from_dict(llm_section)
        static = {k: v for k, v in llm_section.items()
                 if k in cls.__dataclass_fields__ and k != "method"}
        return cls.from_master_config(section, overrides=static)


@dataclass
class LLMResponse:
    """Parsed response from the LLM."""

    content: str = ""
    thinking: str = ""          # internal reasoning chain
    model: str = ""
    usage: dict = field(default_factory=dict)
    finish_reason: str = ""
    latency_ms: float = 0.0
    raw: dict = field(default_factory=dict)
    # Native function-calling tool calls (OpenAI ``tool_calls`` format).
    # Each entry: {"id": "...", "type": "function",
    #              "function": {"name": "...", "arguments": "{...}"}}
    tool_calls: list[dict] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)

    @property
    def has_thinking(self) -> bool:
        return bool(self.thinking)

    @property
    def has_tool_calls(self) -> bool:
        """True when the model returned native function-call requests."""
        return bool(self.tool_calls)


# --- context-length overflow handling ---------------------------------------
# Gateways word this differently (OpenAI: "context_length_exceeded"; vLLM:
# "maximum context length"; others: "too long"/"reduce the length"). Match on
# any of them rather than one vendor's exact string.
_CTX_ERR_MARKERS = (
    "context_length_exceeded", "maximum context length", "context length",
    "too many tokens", "reduce the length", "input is too long",
    "prompt is too long", "exceeds the model",
)


def _is_context_length_error(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in _CTX_ERR_MARKERS)


def _shrink_messages(messages: list[dict]) -> list[dict] | None:
    """Drop the middle of the history, keeping the system prompt and the most
    recent turns. Returns None when there is nothing left to safely drop.

    A `tool` message must immediately follow the assistant message carrying
    its tool_calls, so the kept tail is snapped back over any leading tool
    messages -- otherwise the retry fails with a DIFFERENT 400 (orphaned tool
    message) and looks like the shrink did not help.
    """
    if len(messages) <= 4:
        return None
    head = messages[:1]                       # system prompt
    keep = max(2, (len(messages) - 1) // 2)   # halve the remaining history
    start = len(messages) - keep
    while start < len(messages) and messages[start].get("role") == "tool":
        start += 1
    tail = messages[start:]
    if not tail or len(head) + len(tail) >= len(messages):
        return None
    note = {"role": "system",
            "content": ("[history truncated to fit the model's context "
                        "window -- earlier turns are archived in "
                        "compacted_history.jsonl and retrievable with "
                        "recall_history]")}
    return head + [note] + tail


class LLMClient:
    """OpenAI-compatible chat-completion client.

    Parameters
    ----------
    base_url   : API base (e.g. ``http://localhost:8000/v1``).
    model      : Model name served at that endpoint.
    api_key    : Bearer token (use ``"EMPTY"`` for vLLM).
    config     : Pass an ``LLMConfig`` directly (overrides other args).
    """

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None, config: LLMConfig | None = None,
                 **kw):
        if config:
            self.config = config
        else:
            self.config = LLMConfig(
                base_url=base_url or "http://localhost:8000/v1",
                model=model or "default",
                api_key=api_key or "EMPTY",
                **{k: v for k, v in kw.items() if k in LLMConfig.__dataclass_fields__},
            )
        effective_key = self.config.resolve_api_key()
        if not self.config.verify_ssl:
            log.warning(
                "verify_ssl=False -- accepting self-signed certs at %s. "
                "OK for internal endpoints; NEVER for public ones.",
                self.config.base_url,
            )
        self._client = httpx.Client(
            base_url=self.config.base_url,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {effective_key}",
                     **self.config.extra_headers},
            timeout=httpx.Timeout(self.config.timeout, connect=10.0),
            verify=self.config.verify_ssl,
        )
        self._call_log: list[dict] = []

    @classmethod
    def from_config(cls, path: str | Path, section: str = "llm") -> LLMClient:
        """Construct from a YAML config file. `section` matters only for a
        method-based config (`llm.method: self|gateway|api`) -- pass "judge"
        to resolve a judge config against main_config.yaml's judge: block
        instead of the main actor's llm: block."""
        return cls(config=LLMConfig.from_yaml(path, section=section))

    # -- core API -------------------------------------------

    def generate(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        **kw,
    ) -> LLMResponse:
        """Send a chat-completion request with retry.

        Parameters
        ----------
        messages:
            Conversation history (OpenAI format).
        temperature, max_tokens, stop:
            Standard generation controls; override the config defaults.
        tools:
            OpenAI-style tool specs (``[{"type": "function", "function": {...}}]``).
            Passed through as ``tools=`` in the request body.  Use
            ``DICTIONARY_TOOL_SPECS`` from ``sentinel.dictionary.tool`` to
            inject Schema KB tools into native function-calling models.
        tool_choice:
            ``"auto"`` (default when tools are provided), ``"none"``, or a
            specific tool dict ``{"type": "function", "function": {"name": "..."}}``.
        """
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": merge_system_messages(messages),
            # Explicitly non-streaming: some gateways (e.g. the gateway) default
            # to SSE streaming when `stream` is omitted, which returns a
            # `data: {...}` event stream instead of a JSON object and makes
            # r.json() fail with "Expecting value: line 1 column 1 (char 0)".
            # The working curl sets stream:false -- we match it. (extra_body /
            # kw can still override if a caller really wants streaming here.)
            "stream": False,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "top_p": self.config.top_p,
            **self.config.extra_body, **kw,
        }
        if stop:
            body["stop"] = stop
        if tools:
            body["tools"] = tools
            if tool_choice is not None:
                body["tool_choice"] = tool_choice

        last_err = None
        for attempt in range(self.config.retry_attempts):
            try:
                t0 = time.perf_counter()
                r = self._client.post("/chat/completions", json=body)
                latency = (time.perf_counter() - t0) * 1000

                if r.status_code != 200:
                    if _is_auth_error(r.status_code, r.text):
                        # Never retry a bad key -- it will fail identically
                        # `retry_attempts` times, burning retry_delay each
                        # time, before the caller finds out. Raise now so
                        # the CLI can re-prompt immediately instead.
                        raise LLMAuthError(
                            f"HTTP {r.status_code}: {r.text[:300]}")
                    last_err = Exception(f"HTTP {r.status_code}: {r.text[:300]}")
                    log.warning("LLM error (attempt %d): %s", attempt + 1, last_err)
                    # A context-length rejection can NEVER succeed on retry
                    # with the same body -- the old code resent it verbatim
                    # three times and then failed. Shrink the middle of the
                    # message list (keeping the system prompt and the most
                    # recent turns) and retry smaller instead.
                    if _is_context_length_error(r.text):
                        shrunk = _shrink_messages(body.get("messages") or [])
                        if shrunk is not None:
                            log.warning(
                                "context-length rejection: retrying with %d "
                                "messages instead of %d",
                                len(shrunk), len(body.get("messages") or []))
                            body["messages"] = shrunk
                            continue    # retry immediately, smaller
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue

                resp = self._parse(self._response_json(r), latency)
                self._log(messages, resp)
                return resp

            except LLMAuthError:
                raise   # never caught by the catch-all below -- no retry
            except httpx.TimeoutException as e:
                last_err = e
                log.warning("LLM timeout (attempt %d)", attempt + 1)
                time.sleep(self.config.retry_delay * (attempt + 1))
            except Exception as e:
                last_err = e
                log.error("LLM error (attempt %d): %s", attempt + 1, e)
                time.sleep(self.config.retry_delay * (attempt + 1))

        raise ConnectionError(f"Failed after {self.config.retry_attempts} attempts: {last_err}")

    def generate_stream(self, messages: list[dict], **kw) -> Generator[str, None, None]:
        """Yield tokens as they arrive (SSE streaming)."""
        body = {"model": self.config.model, "messages": merge_system_messages(messages),
                "stream": True,
                "temperature": kw.get("temperature", self.config.temperature),
                "max_tokens": kw.get("max_tokens", self.config.max_tokens),
                **self.config.extra_body}
        with self._client.stream("POST", "/chat/completions", json=body) as r:
            for line in r.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    token = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if token:
                        yield token
                except json.JSONDecodeError:
                    continue

    # -- helpers --------------------------------------------

    @staticmethod
    def _response_json(r) -> dict:
        """Parse a chat-completion response robustly.

        Handles three cases seen against real gateways:
        1. a normal JSON object -> returned as-is;
        2. an SSE stream (``data: {...}`` lines) even though we asked for
           stream:false -> reassemble the choices from the chunks;
        3. an empty/garbage body -> raise a CLEAR error with the raw text
           and content-type, instead of the cryptic
           "Expecting value: line 1 column 1 (char 0)".
        """
        try:
            return r.json()
        except Exception:
            pass
        text = (r.text or "").strip()
        # SSE reassembly
        if "data:" in text:
            content_parts: list[str] = []
            last_obj: dict = {}
            usage: dict = {}
            finish_reason = ""
            # Streamed native tool calls arrive as fragments keyed by
            # `index`: the first carries id/name, later ones append to
            # `arguments`. They used to be dropped here, so a gateway that
            # streamed despite stream:false produced an apparently empty turn.
            tool_calls: dict[int, dict] = {}
            for line in text.splitlines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload in ("", "[DONE]"):
                    continue
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                last_obj = obj
                if obj.get("usage"):
                    usage = obj["usage"]
                choice = (obj.get("choices") or [{}])[0]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or choice.get("message") or {}
                if delta.get("content"):
                    content_parts.append(delta["content"])
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", len(tool_calls))
                    slot = tool_calls.setdefault(idx, {
                        "id": "", "type": "function",
                        "function": {"name": "", "arguments": ""}})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["function"]["arguments"] += fn["arguments"]
            if content_parts or last_obj:
                message: dict = {"role": "assistant", "content": "".join(content_parts)}
                if tool_calls:
                    message["tool_calls"] = [
                        {**tc, "id": tc["id"] or f"call_{i}"}
                        for i, tc in (sorted(tool_calls.items()))]
                return {"choices": [{"message": message, "finish_reason": finish_reason}],
                        "usage": usage}
        ctype = r.headers.get("content-type", "?")
        raise ValueError(
            f"Non-JSON response from LLM endpoint (content-type={ctype}). "
            f"First 200 chars: {text[:200]!r}. If this is an SSE stream, the "
            "gateway ignored stream:false; if empty, check the model name and "
            "endpoint path.")

    def _parse(self, data: dict, latency: float) -> LLMResponse:
        # `or [{}]` guards an empty choices list (key present but []), which
        # would otherwise IndexError before we can read the content.
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content") or ""
        thinking = ""

        # Extract <think>...</think> blocks (Qwen3 / DeepSeek / Kimi style)
        if "<think>" in content and "</think>" in content:
            i = content.index("<think>") + 7
            j = content.index("</think>")
            thinking = content[i:j].strip()
            content = content[j + 8:].strip()

        # Some APIs put reasoning in a separate field. Only use it when a
        # <think> block did not already populate `thinking`, so we never
        # clobber the inline reasoning we just extracted.
        if not thinking and msg.get("reasoning_content"):
            thinking = msg["reasoning_content"]

        # Native function-calling tool calls (OpenAI format).
        tool_calls: list[dict] = list(msg.get("tool_calls") or [])
        # Legacy single `function_call` (older OpenAI schema; still emitted by
        # some gateways). If present and no `tool_calls`, normalise it so the
        # agent loop can dispatch it -- otherwise the model looks like it said
        # nothing at all (empty content + an unread function_call = a silent
        # turn, exactly the symptom we hunt for).
        if not tool_calls and msg.get("function_call"):
            fc = msg["function_call"]
            tool_calls = [{
                "id": fc.get("name") or "call_0",
                "type": "function",
                "function": {
                    "name": fc.get("name", ""),
                    "arguments": fc.get("arguments") or "{}",
                },
            }]
        # Last resort: the server didn't parse the model's native tool-call
        # tokens, so they arrived as text in `content`. Recover them and strip
        # the markup from the visible content.
        if not tool_calls and content:
            recovered, cleaned = _recover_tool_calls_from_text(content)
            if recovered:
                tool_calls = recovered
                content = cleaned

        return LLMResponse(
            content=content,
            thinking=thinking,
            model=data.get("model", self.config.model),
            usage=data.get("usage", {}),
            finish_reason=choice.get("finish_reason", ""),
            latency_ms=round(latency, 1),
            raw=data,
            tool_calls=tool_calls,
        )

    def _log(self, messages: list, resp: LLMResponse):
        self._call_log.append({
            "ts": time.time(), "model": resp.model,
            "prompt_tokens": resp.usage.get("prompt_tokens", 0),
            "completion_tokens": resp.usage.get("completion_tokens", 0),
            "latency_ms": resp.latency_ms,
            "has_thinking": resp.has_thinking,
        })

    @property
    def call_log(self) -> list[dict]:
        return self._call_log

    def health_check(self) -> dict:
        """Ping ``/models`` to verify the endpoint is alive."""
        try:
            r = self._client.get("/models")
            if r.status_code == 200:
                models = [m.get("id", "") for m in r.json().get("data", [])]
                return {"status": "healthy", "models": models}
            return {"status": "error", "code": r.status_code}
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}

    def list_models_detailed(self, timeout: float = 8.0) -> list[dict]:
        """The raw `data` entries from the models listing -- id plus
        whatever the gateway adds (vLLM-style servers include
        `max_model_len`, the real context window; see
        sentinel.llm.catalog.context_window_from_listing, I-11). Empty
        list on any failure, same contract as list_models()."""
        try:
            url = self.config.models_url.strip() or "/models"
            r = self._client.get(url, timeout=timeout)
            if r.status_code == 200:
                return [m for m in r.json().get("data", [])
                        if isinstance(m, dict) and m.get("id")]
        except Exception:  # noqa: BLE001
            pass
        return []

    def list_models(self, timeout: float = 8.0) -> list[str]:
        """Model ids this endpoint actually serves right now (see
        sentinel.llm.catalog for ranking/selection). A short, INDEPENDENT
        timeout -- this runs on every shell startup, so it must not inherit
        the full generation timeout (up to `config.timeout`, often minutes)
        and turn a slow/unreachable endpoint into a multi-minute hang before
        the session even starts. Empty list on any failure -- discovery
        must never be what blocks a session; callers fall back to the
        configured model. Uses `config.models_url` instead of `/models`
        on `base_url` when the gateway needs a separate discovery
        endpoint (an absolute URL bypasses the client's base_url)."""
        try:
            url = self.config.models_url.strip() or "/models"
            r = self._client.get(url, timeout=timeout)
            if r.status_code == 200:
                return [m.get("id", "") for m in r.json().get("data", [])
                       if m.get("id")]
        except Exception:  # noqa: BLE001
            pass
        return []

    def set_api_key(self, key: str) -> None:
        """Live-swap the API key -- updates both `config.api_key` (so
        anything reading it back sees the new one) and the already-open
        httpx client's Authorization header (updating `config.api_key`
        alone would NOT affect requests, since the header is fixed at
        construction time). Used after an LLMAuthError: re-prompt, then
        call this instead of rebuilding the whole client mid-session."""
        self.config.api_key = key
        self.config.api_key_env = None
        self._client.headers["Authorization"] = f"Bearer {key}"

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
