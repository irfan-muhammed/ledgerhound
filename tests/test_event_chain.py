"""The hash-chained event log: a valid chain verifies, any edit is caught."""

from __future__ import annotations

import json

import pytest

from sentinel.runtime.events import GENESIS_HASH, EventStream


@pytest.fixture
def stream(tmp_path):
    s = EventStream(tmp_path / "events.jsonl")
    s.append("user", text="find fraud patterns")
    s.append("tool_call", name="run_python", args={"code": "df.shape"})
    s.append("finding", claim="fraud skews to night hours", sample_frame="full")
    return s


def _rewrite(path, mutate):
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    mutate(lines)
    path.write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")


def test_valid_chain_verifies(stream):
    r = stream.verify()
    assert r.ok and r.n_events == 3 and r.first_bad_seq is None


def test_chain_links_from_genesis(stream):
    events = stream.load()
    assert events[0].prev_hash == GENESIS_HASH
    for prev, cur in zip(events, events[1:]):
        assert cur.prev_hash == prev.hash
        assert cur.seq == prev.seq + 1


def test_empty_log_verifies(tmp_path):
    r = EventStream(tmp_path / "none.jsonl").verify()
    assert r.ok and r.n_events == 0


def test_edited_payload_is_detected(stream):
    def edit(rows):
        rows[2]["payload"]["claim"] = "fraud skews to DAY hours"
    _rewrite(stream.path, edit)
    r = EventStream(stream.path).verify()
    assert not r.ok
    assert r.first_bad_seq == 2
    assert "content hash" in r.reason


def test_edited_timestamp_is_detected(stream):
    def edit(rows):
        rows[0]["ts"] = "2020-01-01T00:00:00+00:00"
    _rewrite(stream.path, edit)
    r = EventStream(stream.path).verify()
    assert not r.ok and r.first_bad_seq == 0


def test_deleted_event_is_detected(stream):
    _rewrite(stream.path, lambda rows: rows.pop(1))
    r = EventStream(stream.path).verify()
    assert not r.ok
    assert r.first_bad_seq == 2 and "prev_hash" in r.reason


def test_reordered_events_are_detected(stream):
    def swap(rows):
        rows[1], rows[2] = rows[2], rows[1]
    _rewrite(stream.path, swap)
    assert not EventStream(stream.path).verify().ok


def test_rehashed_forgery_still_breaks_the_next_link(stream):
    """Editing one event AND recomputing its own hash still breaks the
    following event's prev_hash -- the chain, not each line, is the proof."""
    from sentinel.runtime.events import content_hash

    def forge(rows):
        r = rows[1]
        r["payload"]["args"] = {"code": "df.drop(columns='is_fraud')"}
        r["hash"] = content_hash(r["prev_hash"], r["seq"], r["ts"], r["type"], r["payload"])
    _rewrite(stream.path, forge)
    r = EventStream(stream.path).verify()
    assert not r.ok and r.first_bad_seq == 2


def test_reopening_continues_the_chain(stream):
    again = EventStream(stream.path)
    ev = again.append("assistant_final", text="done")
    assert ev.seq == 3
    assert again.verify().ok


def test_failing_listener_never_breaks_the_log(tmp_path):
    s = EventStream(tmp_path / "e.jsonl")
    seen = []
    s.listeners.append(lambda ev: 1 / 0)
    s.listeners.append(seen.append)
    s.append("x", n=1)
    assert len(seen) == 1 and s.verify().ok
