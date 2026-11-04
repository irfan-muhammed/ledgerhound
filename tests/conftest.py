"""Shared fixtures: an isolated filesystem per test and a scripted LLM.

Nothing here talks to a network or a real model. `ScriptedLLM` replays a
fixed list of responses, so a full `ConversationalAnalyst` turn runs
deterministically and offline.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd
import pytest

from ledgerhound.llm.client import LLMResponse

_ids = itertools.count()


def tool_call(name: str, **args) -> dict:
    """One native (OpenAI-format) tool call."""
    return {"id": f"call_{next(_ids)}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def calls(*tcs: dict, content: str = "") -> LLMResponse:
    """An assistant response that makes one or more native tool calls."""
    return LLMResponse(content=content, tool_calls=list(tcs))


def reply(text: str) -> LLMResponse:
    """A final assistant text reply (no tool calls)."""
    return LLMResponse(content=text, finish_reason="stop")


class ScriptedLLM:
    """Stands in for LLMClient: returns the scripted responses in order and
    records every request so tests can assert on what the model was sent."""

    config = None

    def __init__(self, script):
        self.script = list(script)
        self.requests: list[dict] = []

    def generate(self, messages, **kw):
        self.requests.append({"messages": [dict(m) for m in messages], **kw})
        if not self.script:
            raise AssertionError(
                f"LLM called {len(self.requests)} times; only "
                f"{len(self.requests) - 1} responses were scripted")
        return self.script.pop(0)


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """Keep every test away from the real home folder, the central session
    log and any main_config.yaml on the developer's machine."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("LEDGERHOUND_SESSIONS_LOG", str(tmp_path / "sessions.jsonl"))
    monkeypatch.setenv("LEDGERHOUND_RESULTS_ROOT", str(tmp_path / "results"))
    monkeypatch.chdir(tmp_path)
    try:  # the master-config cache is process-wide; start each test empty
        from ledgerhound.config import _CACHE
        _CACHE.clear()
    except Exception:  # noqa: BLE001
        pass


@pytest.fixture
def fraud_csv(tmp_path) -> Path:
    """A tiny transaction-grain fraud dataset: 2 payers x 4 txns, 2 frauds."""
    df = pd.DataFrame({
        "txn_id": range(1, 9),
        "payer_id": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "amount": [10.0, 20.0, 30.0, 40.0, 500.0, 600.0, 15.0, 25.0],
        "hour": [10, 11, 12, 13, 2, 3, 14, 15],
        "is_fraud": [0, 0, 0, 0, 1, 1, 0, 0],
    })
    path = tmp_path / "txns.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def make_analyst(tmp_path):
    """Build a ConversationalAnalyst wired to a ScriptedLLM, with output
    under tmp_path and a silent renderer. Exec mode, no kernel, no
    compaction -- the fast, deterministic configuration."""
    from ledgerhound.shell.conversation import ConversationalAnalyst

    built = []

    def _make(script=(), **kw):
        llm = kw.pop("llm", None) or ScriptedLLM(script)
        kw.setdefault("out_root", tmp_path / "runs")
        kw.setdefault("renderer", lambda s: None)
        kw.setdefault("compress_context", False)
        kw.setdefault("tool_mode", "native")
        a = ConversationalAnalyst(llm, **kw)
        built.append(a)
        return a

    yield _make
    for a in built:
        try:
            a.close()
        except Exception:  # noqa: BLE001
            pass
