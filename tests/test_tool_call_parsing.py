"""Tool calls arrive in many shapes: native JSON, legacy function_call,
text markup (Kimi / Hermes / our <tool> protocol / bare JSON), and SSE
streams from gateways that ignore stream:false. Each must parse, and
prose must never be mistaken for a call."""

from __future__ import annotations

import json

import httpx
import pytest

from sentinel.llm.client import LLMClient, LLMConfig, _recover_tool_calls_from_text

# -- LLM client: text recovery -----------------------------------------------


def test_recover_hermes_tool_call():
    text = 'Let me look.\n<tool_call>{"name": "run_python", "arguments": {"code": "df.shape"}}</tool_call>'
    tcs, cleaned = _recover_tool_calls_from_text(text)
    assert [t["function"]["name"] for t in tcs] == ["run_python"]
    assert json.loads(tcs[0]["function"]["arguments"]) == {"code": "df.shape"}
    assert cleaned == "Let me look."


def test_recover_kimi_native_tokens():
    text = ("<|tool_calls_section_begin|><|tool_call_begin|>functions.load_data:0"
            '<|tool_call_argument_begin|>{"path": "txns.csv"}<|tool_call_end|>'
            "<|tool_calls_section_end|>")
    tcs, cleaned = _recover_tool_calls_from_text(text)
    assert tcs[0]["function"]["name"] == "load_data"
    assert json.loads(tcs[0]["function"]["arguments"]) == {"path": "txns.csv"}
    assert cleaned == ""


@pytest.mark.parametrize("prose", [
    "The fraud rate is 2%.",
    'Config looks like {"name": "run_python"} in the docs.',
    "<tool_call>not json</tool_call>",
])
def test_prose_is_not_a_tool_call(prose):
    tcs, _ = _recover_tool_calls_from_text(prose)
    assert tcs == []


# -- LLM client: full generate() against a mocked gateway --------------------

def _client(handler) -> LLMClient:
    c = LLMClient(config=LLMConfig(base_url="http://gw.test/v1", model="m", api_key="k",
                                   retry_attempts=1, retry_delay=0))
    c._client = httpx.Client(base_url="http://gw.test/v1", transport=httpx.MockTransport(handler))
    return c


def _sse(*chunks: dict) -> httpx.Response:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


def test_native_tool_calls_parse():
    tc = {"id": "c1", "type": "function",
          "function": {"name": "profile_data", "arguments": "{}"}}
    c = _client(lambda req: httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [tc]},
                     "finish_reason": "tool_calls"}]}))
    r = c.generate([{"role": "user", "content": "hi"}])
    assert r.tool_calls == [tc] and r.content == ""


def test_request_asks_for_non_streaming():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})
    _client(handler).generate([{"role": "user", "content": "hi"}])
    assert seen["stream"] is False


def test_legacy_function_call_is_normalised():
    c = _client(lambda req: httpx.Response(200, json={"choices": [{"message": {
        "content": "", "function_call": {"name": "load_data", "arguments": '{"path": "x.csv"}'}}}]}))
    r = c.generate([{"role": "user", "content": "hi"}])
    assert r.tool_calls[0]["function"] == {"name": "load_data", "arguments": '{"path": "x.csv"}'}


def test_text_tool_call_in_content_is_recovered():
    content = '<tool_call>{"name": "run_python", "arguments": {"code": "1+1"}}</tool_call>'
    c = _client(lambda req: httpx.Response(200, json={"choices": [{"message": {"content": content}}]}))
    r = c.generate([{"role": "user", "content": "hi"}])
    assert r.tool_calls[0]["function"]["name"] == "run_python"
    assert r.content == ""


def test_think_block_is_split_from_content():
    c = _client(lambda req: httpx.Response(200, json={"choices": [{"message": {
        "content": "<think>check grain first</think>Fraud is 2%."}}]}))
    r = c.generate([{"role": "user", "content": "hi"}])
    assert r.thinking == "check grain first" and r.content == "Fraud is 2%."


def test_sse_text_is_reassembled():
    c = _client(lambda req: _sse(
        {"choices": [{"delta": {"role": "assistant", "content": "Fraud "}}]},
        {"choices": [{"delta": {"content": "is 2%."}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 5, "completion_tokens": 3}}))
    r = c.generate([{"role": "user", "content": "hi"}])
    assert r.content == "Fraud is 2%."
    assert r.usage["completion_tokens"] == 3
    assert r.finish_reason == "stop"


def test_sse_streamed_tool_calls_are_reassembled():
    """A gateway that streams despite stream:false sends native tool calls
    as argument fragments keyed by index. They must be stitched back
    together, not dropped (which made the turn look empty)."""
    c = _client(lambda req: _sse(
        {"choices": [{"delta": {"role": "assistant", "tool_calls": [
            {"index": 0, "id": "c1", "type": "function",
             "function": {"name": "run_python", "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '{"code": '}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '"df.shape"}'}},
            {"index": 1, "id": "c2", "type": "function",
             "function": {"name": "profile_data", "arguments": "{}"}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}))
    r = c.generate([{"role": "user", "content": "hi"}])
    assert [t["function"]["name"] for t in r.tool_calls] == ["run_python", "profile_data"]
    assert json.loads(r.tool_calls[0]["function"]["arguments"]) == {"code": "df.shape"}
    assert r.tool_calls[0]["id"] == "c1"
    assert r.finish_reason == "tool_calls"


def test_sse_text_tool_call_is_recovered():
    c = _client(lambda req: _sse(
        {"choices": [{"delta": {"content": '<tool_call>{"name": "load_data", '}}]},
        {"choices": [{"delta": {"content": '"arguments": {"path": "a.csv"}}</tool_call>'}}]}))
    r = c.generate([{"role": "user", "content": "hi"}])
    assert r.tool_calls[0]["function"]["name"] == "load_data"


def test_garbage_body_raises_clear_error():
    c = _client(lambda req: httpx.Response(200, text="", headers={"content-type": "text/html"}))
    with pytest.raises(ConnectionError, match="Non-JSON response"):
        c.generate([{"role": "user", "content": "hi"}])


# -- shell text protocol (ConversationalAnalyst._parse_text_tool_calls) -------

def test_wrapped_tool_block(make_analyst):
    a = make_analyst(tool_mode="auto")
    calls, cleaned = a._parse_text_tool_calls(
        'Checking.\n<tool>{"name": "run_python", "arguments": {"code": "df.head()"}}</tool>')
    assert [(c["name"], c["arguments"]) for c in calls] == [("run_python", {"code": "df.head()"})]
    assert cleaned == "Checking."


def test_bare_json_call_at_end_of_message_runs(make_analyst):
    a = make_analyst(tool_mode="auto")
    calls, _ = a._parse_text_tool_calls(
        'Sure, computing that.\n{"name": "run_python", "arguments": {"code": "df.shape"}}')
    assert [c["name"] for c in calls] == ["run_python"]


def test_bare_json_quoted_in_prose_is_ignored_in_auto_mode(make_analyst):
    a = make_analyst(tool_mode="auto")
    calls, _ = a._parse_text_tool_calls(
        'I would call {"name": "run_python", "arguments": {"code": "df.shape"}} -- shall I?')
    assert calls == []


def test_bare_json_quoted_in_prose_runs_in_prompt_mode(make_analyst):
    a = make_analyst(tool_mode="prompt")
    calls, _ = a._parse_text_tool_calls(
        'I would call {"name": "run_python", "arguments": {"code": "df.shape"}} -- shall I?')
    assert [c["name"] for c in calls] == ["run_python"]


def test_unknown_tool_name_is_not_a_call(make_analyst):
    a = make_analyst(tool_mode="auto")
    calls, _ = a._parse_text_tool_calls('<tool>{"name": "rm_rf", "arguments": {}}</tool>')
    assert calls == []


def test_nested_braces_in_arguments(make_analyst):
    a = make_analyst(tool_mode="auto")
    code = "d = {'a': {'b': 1}}"
    calls, _ = a._parse_text_tool_calls(json.dumps({"name": "run_python", "arguments": {"code": code}}))
    assert calls[0]["arguments"] == {"code": code}


def test_unreadable_arguments_are_reported_not_run_with_empty_args(make_analyst):
    a = make_analyst(tool_mode="auto")
    calls, _ = a._parse_text_tool_calls(
        '<tool>{"name": "run_python", "arguments": "{code: df.shape"}</tool>')
    assert calls[0]["arguments"] is None and calls[0]["arg_error"]
