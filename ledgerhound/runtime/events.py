"""
ledgerhound.runtime.events
========================

V3-012: a typed, hash-chained, append-only event stream -- the canonical
record of what an agent session did. Each ``Event`` is sequenced and
chained to the previous one (``prev_hash``), so tampering with any single
stored line is detectable and localizable: ``EventStream.verify()`` reports
the exact ``seq`` where the chain breaks.

Design notes (an architecture review preceded implementation; see
docs/v3.0/V3_GAP_ANALYSIS_AND_ROADMAP.md V3-012 for the close-out evidence):

  - Hashing covers the canonical form -- ``json.dumps(..., sort_keys=True,
    separators=(",", ":"))`` over ``{seq, ts, type, payload}`` -- computed
    over the FULL payload, never a truncated/preview form. Anything that
    needs truncation for display (long code, long output) stores a
    ``..._preview`` field alongside a ``..._hash`` of the full untruncated
    text, so verification is unaffected by what's convenient to print.
  - ``session_code.py`` and ``findings.jsonl`` (the two views a shell
    session already produced) become genuinely single-sourced FROM this
    stream via ``ledgerhound.runtime.views``: the same projection functions
    are called both live (so the file exists immediately -- unchanged
    behaviour) and in bulk by ``ledgerhound run-views <run_dir>``.
  - ``transcript.jsonl`` (the older flat per-line log) is NOT derived from
    this stream. It remains a parallel, independently-written log for
    backward compatibility with existing consumers (tests,
    ``ledgerhound.eval.fluidity``) that parse its flat shape. This is a
    deliberate, documented scope decision: collapsing it in is future work.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

GENESIS_HASH = "0" * 64


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(seq: int, ts: str, type_: str, payload: dict) -> bytes:
    return json.dumps(
        {"seq": seq, "ts": ts, "type": type_, "payload": payload},
        sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")


def content_hash(prev_hash: str, seq: int, ts: str, type_: str, payload: dict) -> str:
    h = hashlib.sha256()
    h.update(prev_hash.encode("utf-8"))
    h.update(_canonical(seq, ts, type_, payload))
    return h.hexdigest()


@dataclass
class Event:
    seq: int
    ts: str
    type: str
    payload: dict
    prev_hash: str
    hash: str

    def to_dict(self) -> dict:
        return {
            "seq": self.seq, "ts": self.ts, "type": self.type,
            "payload": self.payload, "prev_hash": self.prev_hash, "hash": self.hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return cls(
            seq=d["seq"], ts=d["ts"], type=d["type"], payload=d.get("payload", {}),
            prev_hash=d["prev_hash"], hash=d["hash"],
        )


@dataclass
class VerifyResult:
    ok: bool
    n_events: int
    first_bad_seq: Optional[int] = None
    reason: str = ""


class EventStream:
    """Append-only, hash-chained event log at ``path`` (JSONL, one Event
    per line). Not thread-safe -- matches the rest of the shell's
    single-threaded design."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._last_hash = GENESIS_HASH
        self._next_seq = 0
        # Called with each Event right after it is written -- lets a
        # consumer (the session scorecard) keep running counts instead of
        # re-reading this file. A failing listener never affects the write.
        self.listeners: list = []
        if self.path.exists():
            for ev in self.load():
                self._last_hash = ev.hash
                self._next_seq = ev.seq + 1

    def append(self, type_: str, **payload: Any) -> Event:
        ts = _ts()
        seq = self._next_seq
        h = content_hash(self._last_hash, seq, ts, type_, payload)
        ev = Event(seq=seq, ts=ts, type=type_, payload=payload,
                   prev_hash=self._last_hash, hash=h)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev.to_dict(), default=str) + "\n")
        self._last_hash = h
        self._next_seq = seq + 1
        for listener in self.listeners:
            try:
                listener(ev)
            except Exception:  # noqa: BLE001 -- observers never break the log
                pass
        return ev

    def load(self) -> list[Event]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            out.append(Event.from_dict(json.loads(line)))
        return out

    def verify(self) -> VerifyResult:
        events = self.load()
        prev = GENESIS_HASH
        for ev in events:
            if ev.prev_hash != prev:
                return VerifyResult(False, len(events), ev.seq,
                                    f"prev_hash mismatch at seq {ev.seq}")
            expect = content_hash(prev, ev.seq, ev.ts, ev.type, ev.payload)
            if expect != ev.hash:
                return VerifyResult(False, len(events), ev.seq,
                                    f"content hash mismatch at seq {ev.seq}")
            prev = ev.hash
        return VerifyResult(True, len(events))


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file -- used to fingerprint the data a
    session loaded, without holding the whole file in memory twice."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
