from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aibecomeshuman.config import ModelProfile
from aibecomeshuman.core.events import EventLog
from aibecomeshuman.llm.anthropic_provider import FALLBACK_BETA, AnthropicProvider
from aibecomeshuman.llm.fake import FakeProvider, example_from_schema
from aibecomeshuman.llm.recording import CassetteProvider, RecordingProvider
from aibecomeshuman.llm.types import ImagePart, LLMError, LLMRefusal, LLMRequest, Message

SCHEMA = {
    "type": "object",
    "properties": {
        "choice": {"type": "string", "enum": ["calm", "pressure"]},
        "confidence": {"type": "number"},
        "evidence": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    },
    "required": ["choice", "confidence", "evidence"],
    "additionalProperties": False,
}


def fake_profile(**options) -> ModelProfile:
    return ModelProfile(name="fake", provider="fake", model="fake-1", options=options)


def req(text: str = "hello", **kw) -> LLMRequest:
    return LLMRequest(purpose="test", messages=[Message.user(text)], **kw)


# -- fake provider -------------------------------------------------------------


def test_fake_scripted_then_echo() -> None:
    provider = FakeProvider(fake_profile(responses=["one"]))
    assert provider.complete(req()).text == "one"
    assert provider.complete(req("echo")).text.endswith("echo")
    assert len(provider.calls) == 2


def test_fake_generates_schema_conforming_json() -> None:
    response = FakeProvider(fake_profile()).complete(req(json_schema=SCHEMA))
    assert response.parsed == {"choice": "calm", "confidence": 0.0, "evidence": ["fake"]}


def test_fake_scripted_invalid_json_raises() -> None:
    with pytest.raises(LLMError):
        FakeProvider(fake_profile(responses=["not json"])).complete(req(json_schema=SCHEMA))


def test_example_from_schema_handles_nullable() -> None:
    assert example_from_schema({"type": ["null", "integer"]}) == 0


# -- fingerprint and redaction -------------------------------------------------


def test_fingerprint_is_stable_and_content_sensitive() -> None:
    assert req("a").fingerprint() == req("a").fingerprint()
    assert req("a").fingerprint() != req("b").fingerprint()


def test_redacted_replaces_image_data_with_hash() -> None:
    request = LLMRequest(
        purpose="obs",
        messages=[Message(role="user", content=[ImagePart(data_b64="QUJD")])],
    )
    part = request.redacted()["messages"][0]["content"][0]
    assert "data_b64" not in part and len(part["sha256"]) == 64


# -- recording and cassette ----------------------------------------------------


def test_recording_then_cassette_replays_identically(tmp_path: Path) -> None:
    calls = tmp_path / "llm_calls.jsonl"
    events = EventLog(tmp_path / "events.jsonl")
    recorder = RecordingProvider(
        FakeProvider(fake_profile(responses=["recorded"])), calls, role="decider", events=events
    )
    original = recorder.complete(req("situation"))

    record = json.loads(calls.read_text(encoding="utf-8").strip())
    assert record["role"] == "decider" and record["response"]["text"] == "recorded"

    llm_events = [e for e in events if e.type == "llm.call"]
    assert llm_events[0].payload["ok"] is True
    assert "text" not in llm_events[0].payload  # the event log stores no content

    cassette = CassetteProvider(fake_profile(), calls)
    assert cassette.complete(req("situation")) == original
    with pytest.raises(LLMError, match="cassette"):
        cassette.complete(req("another situation"))


def test_recording_logs_errors_and_reraises(tmp_path: Path) -> None:
    calls = tmp_path / "llm_calls.jsonl"
    events = EventLog(tmp_path / "events.jsonl")
    recorder = RecordingProvider(
        FakeProvider(fake_profile(responses=["not json"])), calls, role="judge", events=events
    )
    with pytest.raises(LLMError):
        recorder.complete(req(json_schema=SCHEMA))
    assert "error" in json.loads(calls.read_text(encoding="utf-8"))
    assert [e.payload["ok"] for e in events if e.type == "llm.call"] == [False]


# -- Anthropic adapter (no network: stubbed client) ----------------------------


class StubMessages:
    def __init__(self, response):
        self.response = response
        self.kwargs: dict | None = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def stub_response(text: str, stop_reason: str = "end_turn"):
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        stop_reason=stop_reason,
        stop_details=None,
        model="claude-test",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=None),
        _request_id="req_123",
    )


def stub_client(response):
    plain, beta = StubMessages(response), StubMessages(response)
    return SimpleNamespace(messages=plain, beta=SimpleNamespace(messages=beta)), plain, beta


def anthropic_profile(**kw) -> ModelProfile:
    return ModelProfile(name="c", provider="anthropic", model="claude-test", **kw)


def test_anthropic_builds_params_and_parses_structured_output() -> None:
    client, plain, beta = stub_client(stub_response('{"choice": "calm"}'))
    provider = AnthropicProvider(anthropic_profile(effort="low"), client=client)
    request = LLMRequest(
        purpose="decide",
        system="sys",
        messages=[Message(role="user", content=[ImagePart(data_b64="QUJD")])],
        json_schema=SCHEMA,
    )
    response = provider.complete(request)

    assert response.parsed == {"choice": "calm"}
    assert response.usage.input_tokens == 10 and response.usage.cache_read_tokens == 0
    assert response.request_id == "req_123"
    assert beta.kwargs is None
    sent = plain.kwargs
    assert sent["system"] == "sys"
    assert sent["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": SCHEMA},
    }
    assert sent["messages"][0]["content"][0]["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": "QUJD",
    }


def test_anthropic_uses_server_fallback_when_enabled() -> None:
    client, plain, beta = stub_client(stub_response("hello"))
    AnthropicProvider(anthropic_profile(refusal_fallback=True), client=client).complete(req())
    assert plain.kwargs is None
    assert beta.kwargs["betas"] == [FALLBACK_BETA] and beta.kwargs["fallbacks"] == "default"


def test_anthropic_refusal_raises() -> None:
    client, _, _ = stub_client(stub_response("", stop_reason="refusal"))
    with pytest.raises(LLMRefusal):
        AnthropicProvider(anthropic_profile(), client=client).complete(req())


def test_anthropic_truncated_structured_output_raises() -> None:
    client, _, _ = stub_client(stub_response('{"choice": "ca', stop_reason="max_tokens"))
    with pytest.raises(LLMError, match="max_tokens"):
        AnthropicProvider(anthropic_profile(), client=client).complete(req(json_schema=SCHEMA))
