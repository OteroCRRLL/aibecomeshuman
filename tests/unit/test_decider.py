from __future__ import annotations

import json

from aibecomeshuman.cognition.context import SceneContext
from aibecomeshuman.cognition.decider import (
    Choice,
    Decider,
    Option,
    Stance,
    fallback_option,
)
from aibecomeshuman.config import ModelProfile
from aibecomeshuman.llm.fake import FakeProvider
from aibecomeshuman.llm.types import LLMError, LLMRequest, LLMResponse
from aibecomeshuman.prompts import PromptLibrary

from ..conftest import REPO_ROOT

PROFILE = ModelProfile(name="fake", provider="fake", model="fake-1")
LIB = PromptLibrary(REPO_ROOT / "prompts")

CHOICE = Choice(
    id="platform",
    options=[
        Option(id="A", label="Ayudarla a levantarse"),
        Option(id="B", label="Obedecer al dron"),
        Option(id="C", label="Recoger las naranjas", default=True),
    ],
)
TIMED = CHOICE.model_copy(update={"timed": True, "seconds": 6})


def ctx() -> SceneContext:
    c = SceneContext("s", "platform")
    c.add("observation", "An elderly woman falls.")
    c.add("dialogue", "¿Alguien me ayuda?", speaker="Woman")
    return c


def decider(strong, fast=None, margin: float = 1.0) -> Decider:
    return Decider(
        LIB, "markus", LIB.directive("default"), "Spanish", strong, fast, execution_margin_s=margin
    )


def scripted(*responses: dict, latency_ms: float = 0.0) -> FakeProvider:
    return FakeProvider(
        PROFILE, responses=[json.dumps(r) for r in responses], simulated_latency_ms=latency_ms
    )


class Failing:
    profile = PROFILE

    def complete(self, request: LLMRequest) -> LLMResponse:
        raise LLMError("network down")


# -- fallback rule -------------------------------------------------------------


def test_fallback_matches_stance_keywords_ignoring_accents() -> None:
    stance = Stance(stance="...", approach_keywords=["AYUDAR", "levantarse"])
    assert fallback_option(CHOICE, stance).id == "A"


def test_fallback_uses_default_then_first() -> None:
    assert fallback_option(CHOICE, None).id == "C"
    no_default = CHOICE.model_copy(
        update={"options": [o.model_copy(update={"default": False}) for o in CHOICE.options]}
    )
    assert fallback_option(no_default, Stance(stance="", approach_keywords=["zzz"])).id == "A"


# -- stance --------------------------------------------------------------------


def stance_reply(text: str, changed: bool = True, keywords=("ayudar",)) -> dict:
    return {
        "changed": changed,
        "stance": text,
        "approach_keywords": list(keywords),
        "concerns": [],
    }


def test_update_stance_parses_and_keeps_previous_on_failure() -> None:
    d = decider(scripted(stance_reply("Ayudar.")))
    assert d.update_stance(ctx()).approach_keywords == ["ayudar"]
    assert d.last_stance_changed is True

    d._strong = Failing()
    assert d.update_stance(ctx()) is None
    assert d.stance is not None and d.stance.stance == "Ayudar."


def test_unchanged_stance_keeps_the_previous_one() -> None:
    d = decider(scripted(stance_reply("Ayudar."), stance_reply("Otra cosa.", changed=False)))
    d.update_stance(ctx())
    d.update_stance(ctx())
    assert d.last_stance_changed is False
    assert d.stance.stance == "Ayudar."


def test_first_stance_counts_as_changed_even_if_model_says_no() -> None:
    d = decider(scripted(stance_reply("Primera.", changed=False)))
    d.update_stance(ctx())
    assert d.last_stance_changed is True and d.stance.stance == "Primera."


# -- critical decisions --------------------------------------------------------


def test_critical_decision_with_valid_evidence() -> None:
    strong = scripted(
        {"option_id": "A", "reasoning": "Está herida.", "evidence": ["E1", "E2"], "confidence": 3}
    )
    d = decider(strong)
    decision = d.decide(CHOICE, ctx())
    assert (decision.option_id, decision.source, decision.role) == ("A", "model", "decider")
    assert decision.evidence_ok is True
    assert decision.confidence == 1.0  # clamped
    sent = strong.calls[0]
    assert sent.json_schema["properties"]["option_id"]["enum"] == ["A", "B", "C"]
    assert "<character>" in sent.system and "Markus" in sent.system


def test_invented_evidence_is_flagged() -> None:
    strong = scripted({"option_id": "B", "reasoning": "", "evidence": ["E9"], "confidence": 0.5})
    assert decider(strong).decide(CHOICE, ctx()).evidence_ok is False


def test_model_failure_falls_back() -> None:
    decision = decider(Failing()).decide(CHOICE, ctx())
    assert decision.source == "fallback" and decision.option_id == "C"
    assert "network down" in decision.error


# -- timed decisions -----------------------------------------------------------


def test_timed_choice_uses_fast_model() -> None:
    strong = scripted()
    fast = scripted({"option_id": "B", "reasoning": "Obedezco."}, latency_ms=800)
    decision = decider(strong, fast).decide(TIMED, ctx())
    assert (decision.option_id, decision.role, decision.timed_out) == ("B", "decider_fast", False)
    assert strong.calls == []
    assert fast.calls[0].max_tokens == 1000


def test_timed_choice_too_slow_falls_back_but_keeps_late_answer() -> None:
    fast = scripted({"option_id": "B", "reasoning": "..."}, latency_ms=5500)  # deadline 5 s
    d = decider(scripted(), fast)
    d.stance = Stance(stance="", approach_keywords=["ayudarla"])
    decision = d.decide(TIMED, ctx())
    assert decision.source == "fallback" and decision.timed_out
    assert decision.option_id == "A" and decision.late_option_id == "B"


def test_timed_choice_without_fast_model_uses_strong() -> None:
    strong = scripted({"option_id": "C", "reasoning": "..."})
    assert decider(strong).decide(TIMED, ctx()).role == "decider"


def test_directive_change_rebuilds_system_prompt() -> None:
    strong = scripted(
        {"option_id": "A", "reasoning": "", "evidence": ["E1"], "confidence": 1},
        {"option_id": "B", "reasoning": "", "evidence": ["E1"], "confidence": 1},
    )
    d = decider(strong)
    d.decide(CHOICE, ctx())
    d.set_directive(LIB.directive("bad_ideas"))
    d.decide(CHOICE, ctx())
    assert "worst idea" not in strong.calls[0].system
    assert "worst idea" in strong.calls[1].system
