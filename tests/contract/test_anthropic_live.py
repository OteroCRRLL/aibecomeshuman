"""Prueba de contrato contra la API real. Excluida por defecto (marcador `live`).

    uv run pytest -m live
"""

from __future__ import annotations

import pytest

from aibecomeshuman.config import load_project
from aibecomeshuman.llm.registry import build_provider
from aibecomeshuman.llm.types import LLMRequest, Message

from ..conftest import REPO_ROOT

pytestmark = pytest.mark.live

SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}, "note": {"type": "string"}},
    "required": ["ok", "note"],
    "additionalProperties": False,
}


@pytest.mark.parametrize("model_name", ["claude-haiku", "claude-opus"])
def test_structured_output_roundtrip(model_name: str) -> None:
    profile = load_project(REPO_ROOT).models[model_name]
    response = build_provider(profile).complete(
        LLMRequest(
            purpose="contract",
            messages=[Message.user("Devuelve ok=true y una nota de tres palabras.")],
            json_schema=SCHEMA,
            max_tokens=2000,
        )
    )
    assert response.parsed is not None and response.parsed["ok"] is True
    assert response.usage.input_tokens > 0
