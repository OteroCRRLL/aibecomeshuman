"""Builds the right provider from a model profile.

Adding a new provider = writing its adapter and registering it here.
"""

from __future__ import annotations

from collections.abc import Callable

from aibecomeshuman.config import ModelProfile
from aibecomeshuman.core.clock import Clock
from aibecomeshuman.llm.base import LLMProvider

ProviderFactory = Callable[[ModelProfile, Clock | None], LLMProvider]


def _fake(profile: ModelProfile, clock: Clock | None) -> LLMProvider:
    from aibecomeshuman.llm.fake import FakeProvider

    return FakeProvider(profile, clock=clock)


def _anthropic(profile: ModelProfile, clock: Clock | None) -> LLMProvider:
    from aibecomeshuman.llm.anthropic_provider import AnthropicProvider

    return AnthropicProvider(profile, clock=clock)


PROVIDERS: dict[str, ProviderFactory] = {
    "fake": _fake,
    "anthropic": _anthropic,
}


def build_provider(profile: ModelProfile, clock: Clock | None = None) -> LLMProvider:
    try:
        factory = PROVIDERS[profile.provider]
    except KeyError:
        raise ValueError(f"provider '{profile.provider}' is not registered") from None
    return factory(profile, clock)
