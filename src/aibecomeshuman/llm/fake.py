"""Deterministic fake provider for tests and simulation (no network, no cost)."""

from __future__ import annotations

import json
from typing import Any

from aibecomeshuman.config import ModelProfile
from aibecomeshuman.core.clock import Clock, SystemClock
from aibecomeshuman.llm.types import LLMError, LLMRequest, LLMResponse, Usage


class FakeProvider:
    """Answers, in order of preference:

    1. With the next scripted response (`responses`, or the profile's `options.responses`).
    2. If there is a `json_schema`, with a minimal object that satisfies it.
    3. With an echo of the last user message.
    """

    def __init__(
        self,
        profile: ModelProfile,
        responses: list[str] | None = None,
        clock: Clock | None = None,
        simulated_latency_ms: float | None = None,
    ) -> None:
        self.profile = profile
        if responses is None:
            responses = profile.options.get("responses", [])
        self._responses = list(responses)
        self._cursor = 0
        self._clock = clock or SystemClock()
        self._latency_ms = float(
            simulated_latency_ms
            if simulated_latency_ms is not None
            else profile.options.get("latency_ms", 0.0)
        )
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)

        if self._cursor < len(self._responses):
            text = self._responses[self._cursor]
            self._cursor += 1
        elif request.json_schema is not None:
            text = json.dumps(example_from_schema(request.json_schema), ensure_ascii=False)
        else:
            text = f"[fake:{self.profile.model}] {request.last_user_text()}"

        parsed = None
        if request.json_schema is not None:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as e:
                raise LLMError(f"scripted response is not valid JSON: {e}") from e

        # Rough estimate (~4 characters per token); good enough to test accounting.
        prompt_chars = len(request.model_dump_json())
        return LLMResponse(
            text=text,
            parsed=parsed,
            stop_reason="end_turn",
            usage=Usage(input_tokens=prompt_chars // 4, output_tokens=len(text) // 4),
            provider="fake",
            model=self.profile.model,
            latency_ms=self._latency_ms,
        )


def example_from_schema(schema: dict[str, Any]) -> Any:
    """Build the minimal value that satisfies a simple JSON Schema."""
    if "enum" in schema:
        return schema["enum"][0]
    if "const" in schema:
        return schema["const"]
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), "null")
    match kind:
        case "object":
            props = schema.get("properties", {})
            required = schema.get("required", list(props))
            return {key: example_from_schema(props[key]) for key in required if key in props}
        case "array":
            min_items = schema.get("minItems", 0)
            item = example_from_schema(schema.get("items", {"type": "string"}))
            return [item] * min_items
        case "string":
            return "fake"
        case "integer":
            return int(schema.get("minimum", 0))
        case "number":
            return float(schema.get("minimum", 0))
        case "boolean":
            return False
        case _:
            return None
