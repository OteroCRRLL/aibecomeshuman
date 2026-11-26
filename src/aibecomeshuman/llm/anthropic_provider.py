"""Adaptador para la API de Claude (SDK oficial `anthropic`)."""

from __future__ import annotations

import json
from typing import Any

import anthropic

from aibecomeshuman.config import ModelProfile
from aibecomeshuman.core.clock import Clock, SystemClock
from aibecomeshuman.llm.types import (
    ImagePart,
    LLMError,
    LLMRefusal,
    LLMRequest,
    LLMResponse,
    TextPart,
    Usage,
)

# Reintento en servidor si el modelo rechaza la petición; el API elige el modelo de respaldo.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider:
    def __init__(
        self,
        profile: ModelProfile,
        client: Any | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.profile = profile
        # Sin clave explícita: el SDK la resuelve (ANTHROPIC_API_KEY o perfil de `ant auth login`).
        self._client = client or anthropic.Anthropic(timeout=profile.timeout_s)
        self._clock = clock or SystemClock()

    def build_params(self, request: LLMRequest) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.profile.model,
            "max_tokens": request.max_tokens or self.profile.max_tokens,
            "messages": [_to_api_message(m.role, m.content) for m in request.messages],
        }
        if request.system:
            params["system"] = request.system

        output_config: dict[str, Any] = {}
        if self.profile.effort:
            output_config["effort"] = self.profile.effort
        if request.json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": request.json_schema}
        if output_config:
            params["output_config"] = output_config
        return params

    def complete(self, request: LLMRequest) -> LLMResponse:
        params = self.build_params(request)
        started = self._clock.monotonic()
        try:
            if self.profile.refusal_fallback:
                response = self._client.beta.messages.create(
                    betas=[FALLBACK_BETA], fallbacks="default", **params
                )
            else:
                response = self._client.messages.create(**params)
        except anthropic.APIConnectionError as e:
            raise LLMError(f"sin conexión con la API de Anthropic: {e}") from e
        except anthropic.RateLimitError as e:
            raise LLMError(f"límite de peticiones alcanzado: {e.message}") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"error {e.status_code} de la API de Anthropic: {e.message}") from e
        latency_ms = (self._clock.monotonic() - started) * 1000

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise LLMRefusal(f"el modelo rechazó la petición (categoría: {category})")

        text = "".join(block.text for block in response.content if block.type == "text")

        parsed = None
        if request.json_schema is not None:
            if response.stop_reason == "max_tokens":
                raise LLMError("respuesta estructurada cortada por max_tokens")
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as e:
                raise LLMError(f"respuesta estructurada no es JSON válido: {e}") from e

        usage = response.usage
        return LLMResponse(
            text=text,
            parsed=parsed,
            stop_reason=response.stop_reason,
            usage=Usage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=getattr(usage, "cache_read_input_tokens", None) or 0,
            ),
            provider="anthropic",
            model=response.model,
            latency_ms=latency_ms,
            request_id=getattr(response, "_request_id", None),
        )


def _to_api_message(role: str, parts: list[TextPart | ImagePart]) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, TextPart):
            content.append({"type": "text", "text": part.text})
        else:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": part.media_type,
                        "data": part.data_b64,
                    },
                }
            )
    return {"role": role, "content": content}
