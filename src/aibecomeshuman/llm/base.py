"""Model port: the only thing the rest of the system knows about an LLM."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from aibecomeshuman.config import ModelProfile
from aibecomeshuman.llm.types import LLMRequest, LLMResponse


@runtime_checkable
class LLMProvider(Protocol):
    profile: ModelProfile

    def complete(self, request: LLMRequest) -> LLMResponse: ...
