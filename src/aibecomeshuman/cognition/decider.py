"""Decider: keeps the scene stance and chooses options.

- `update_stance`: the strong model (`decider`) summarizes how the character wants to
  act right now. Runs off the clock, at scene start and after important events.
- `decide`, untimed choice: the strong model decides with the full scene context.
- `decide`, timed choice: the fast model (`decider_fast`) only matches the options
  against the stance. If it misses the deadline, or anything fails, a local rule picks
  the option closest to the stance, without calling any model.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Literal

from pydantic import BaseModel, Field

from aibecomeshuman.cognition.context import SceneContext
from aibecomeshuman.llm.base import LLMProvider
from aibecomeshuman.llm.types import LLMError, LLMRequest, Message
from aibecomeshuman.prompts import Directive, PromptLibrary

# What the fast path sees of the scene. Too little (it was 8) and it reasons from the
# stance alone: in the first mission it mixed up who it was talking to.
FAST_CONTEXT_ENTRIES = 30


class Option(BaseModel):
    id: str
    label: str
    default: bool = False


class Choice(BaseModel):
    id: str
    options: list[Option] = Field(min_length=2)
    timed: bool = False
    seconds: float | None = None
    prompt: str = ""

    @property
    def option_ids(self) -> list[str]:
        return [o.id for o in self.options]


class Stance(BaseModel):
    stance: str
    approach_keywords: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)


class Decision(BaseModel):
    choice_id: str
    option_id: str
    source: Literal["model", "fallback"]
    role: str | None = None
    reasoning: str = ""
    evidence: list[str] = Field(default_factory=list)
    # False when the model cited no evidence, or cited ids that are not in its context.
    evidence_ok: bool = False
    confidence: float | None = None
    latency_ms: float = 0.0
    timed_out: bool = False
    # What the model chose when it answered too late (useful for analysis).
    late_option_id: str | None = None
    error: str | None = None


STANCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "changed": {"type": "boolean"},
        "stance": {"type": "string"},
        "approach_keywords": {"type": "array", "items": {"type": "string"}},
        "concerns": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["changed", "stance", "approach_keywords", "concerns"],
    "additionalProperties": False,
}


def decision_schema(choice: Choice) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "option_id": {"type": "string", "enum": choice.option_ids},
            "reasoning": {"type": "string"},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
        },
        "required": ["option_id", "reasoning", "evidence", "confidence"],
        "additionalProperties": False,
    }


def fast_decision_schema(choice: Choice) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "option_id": {"type": "string", "enum": choice.option_ids},
            "reasoning": {"type": "string"},
        },
        "required": ["option_id", "reasoning"],
        "additionalProperties": False,
    }


STANCE_TASK = (
    "Update your stance for this scene. If nothing important has happened since your "
    "previous stance, set `changed` to false and copy the previous stance as it is. "
    "A stance is about the situation you are in NOW (who you face, what you are trying to "
    "achieve with them): drop goals from small incidents that are already over, and never "
    "carry a previous stance forward just because it was there. "
    "Otherwise set `changed` to true and, in `stance`, write what you intend to do next and "
    "why, as your own intention in this situation (never as a rule you were given), in ONE "
    "short sentence of at most 20 words. If your directive overrides your "
    "personality, the stance must describe what you will actually do under it, as a "
    "concrete intention, not what you would do without it. Do not retell the scene or "
    "repeat your previous stance. In `approach_keywords`, give 3 to 6 short words, in the "
    "same language as the on-screen text, describing the kind of option you will pick next "
    "(e.g. calma, ayudar, obedecer). In `concerns`, list at most two worries of a few "
    "words each."
)

DECIDE_TASK = (
    "Choose one option. In `reasoning`, explain your choice in character in ONE short "
    "sentence of at most 25 words. In `evidence`, list the ids that justify it: events "
    "from this scene "
    "(E1, E2...) and, if they matter, recollections or promises from your memory "
    "(M1, C1...). In `confidence`, give a number from 0 to 1."
)

FAST_TASK = (
    "Time is running out. Pick the option that best fits your current stance. "
    "The options act on what the choice's prompt says they are about; the last thing you "
    "heard may be about something else. "
    "In `reasoning`, give one short sentence of at most 15 words."
)


def _normalize(text: str) -> set[str]:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return {w for w in re.split(r"[^a-z0-9]+", plain) if w}


def fallback_option(choice: Choice, stance: Stance | None) -> Option:
    """Option closest to the stance keywords; else the choice's default; else the first."""
    keywords: set[str] = set()
    if stance is not None:
        for keyword in stance.approach_keywords:
            keywords |= _normalize(keyword)
    if keywords:
        scored = [(len(_normalize(o.label) & keywords), i, o) for i, o in enumerate(choice.options)]
        best_score, _, best = max(scored, key=lambda t: (t[0], -t[1]))
        if best_score > 0:
            return best
    return next((o for o in choice.options if o.default), choice.options[0])


def render_choice(choice: Choice) -> str:
    timer = f' timed="true" seconds="{choice.seconds:g}"' if choice.timed and choice.seconds else ""
    lines = [f'<choice id="{choice.id}"{timer}>']
    if choice.prompt:
        lines.append(f"<prompt>{choice.prompt}</prompt>")
    lines += [f'<option id="{o.id}">{o.label}</option>' for o in choice.options]
    lines.append("</choice>")
    return "\n".join(lines)


def render_stance(stance: Stance | None) -> str:
    if stance is None:
        return "<stance>No stance yet.</stance>"
    keywords = ", ".join(stance.approach_keywords)
    return f"<stance keywords=\"{keywords}\">{stance.stance}</stance>"


class Decider:
    def __init__(
        self,
        prompts: PromptLibrary,
        character: str,
        directive: Directive,
        output_language: str,
        strong: LLMProvider,
        fast: LLMProvider | None = None,
        execution_margin_s: float = 1.0,
    ) -> None:
        self._prompts = prompts
        self.character = character
        self.directive = directive
        self.output_language = output_language
        self._strong = strong
        self._fast = fast
        self._margin = execution_margin_s
        self._system_cache: dict[str, str] = {}
        self.stance: Stance | None = None
        # Whether the last successful update produced a new stance (False = kept the old one).
        self.last_stance_changed = False
        # Long-term memory, set by the runner: personality goes into the system prompt,
        # recollections/known entities/promises into the stance and critical decisions.
        self.personality_text: str | None = None
        self.memory_text: str = ""
        self.memory_ids: set[str] = set()

    def set_directive(self, directive: Directive) -> None:
        self.directive = directive
        self._system_cache.clear()

    def set_memory(self, personality_text: str | None, memory_text: str, ids: set[str]) -> None:
        if personality_text != self.personality_text:
            self._system_cache.clear()
        self.personality_text = personality_text
        self.memory_text = memory_text
        self.memory_ids = set(ids)

    def system(self, role: str) -> str:
        if role not in self._system_cache:
            self._system_cache[role] = self._prompts.system_prompt(
                role, self.character, self.directive, self.output_language, self.personality_text
            )
        return self._system_cache[role]

    def _with_memory(self, parts: list[str]) -> str:
        return "\n\n".join([self.memory_text, *parts] if self.memory_text else parts)

    # -- stance ----------------------------------------------------------------

    def update_stance(self, ctx: SceneContext) -> Stance | None:
        """Ask the strong model for a new stance. On failure, keep the previous one."""
        request = LLMRequest(
            purpose="stance.update",
            system=self.system("decider"),
            messages=[
                Message.user(
                    self._with_memory(
                        [ctx.render(), render_stance(self.stance), f"<task>{STANCE_TASK}</task>"]
                    )
                )
            ],
            json_schema=STANCE_SCHEMA,
        )
        try:
            response = self._strong.complete(request)
        except LLMError:
            return None
        assert response.parsed is not None
        changed = bool(response.parsed.get("changed", True)) or self.stance is None
        if changed:
            self.stance = Stance.model_validate(response.parsed)
        self.last_stance_changed = changed
        return self.stance

    # -- decisions -------------------------------------------------------------

    def decide(self, choice: Choice, ctx: SceneContext) -> Decision:
        if choice.timed and choice.seconds:
            return self._decide_timed(choice, ctx)
        return self._decide_critical(choice, ctx)

    def _decide_critical(self, choice: Choice, ctx: SceneContext) -> Decision:
        request = LLMRequest(
            purpose="decide.critical",
            system=self.system("decider"),
            messages=[
                Message.user(
                    self._with_memory(
                        [
                            ctx.render(),
                            render_stance(self.stance),
                            render_choice(choice),
                            f"<task>{DECIDE_TASK}</task>",
                        ]
                    )
                )
            ],
            json_schema=decision_schema(choice),
        )
        try:
            response = self._strong.complete(request)
        except LLMError as e:
            return self._fallback(choice, error=str(e))
        parsed = response.parsed or {}
        evidence = [str(x) for x in parsed.get("evidence", [])]
        valid = ctx.ids | self.memory_ids
        confidence = parsed.get("confidence")
        return Decision(
            choice_id=choice.id,
            option_id=parsed["option_id"],
            source="model",
            role="decider",
            reasoning=parsed.get("reasoning", ""),
            evidence=evidence,
            evidence_ok=bool(evidence) and all(e in valid for e in evidence),
            confidence=None if confidence is None else max(0.0, min(1.0, float(confidence))),
            latency_ms=response.latency_ms,
        )

    def _decide_timed(self, choice: Choice, ctx: SceneContext) -> Decision:
        provider, role = (self._fast, "decider_fast") if self._fast else (self._strong, "decider")
        assert choice.seconds is not None
        deadline_ms = max(0.0, choice.seconds - self._margin) * 1000
        request = LLMRequest(
            purpose="decide.timed",
            system=self.system("decider_fast"),
            messages=[
                Message.user(
                    "\n\n".join(
                        [
                            ctx.render(last_n=FAST_CONTEXT_ENTRIES),
                            render_stance(self.stance),
                            render_choice(choice),
                            f"<task>{FAST_TASK}</task>",
                        ]
                    )
                )
            ],
            json_schema=fast_decision_schema(choice),
            max_tokens=1000,
        )
        try:
            response = provider.complete(request)
        except LLMError as e:
            return self._fallback(choice, error=str(e))
        parsed = response.parsed or {}
        if response.latency_ms > deadline_ms:
            decision = self._fallback(choice, error=None)
            decision.timed_out = True
            decision.late_option_id = parsed.get("option_id")
            decision.latency_ms = response.latency_ms
            return decision
        return Decision(
            choice_id=choice.id,
            option_id=parsed["option_id"],
            source="model",
            role=role,
            reasoning=parsed.get("reasoning", ""),
            latency_ms=response.latency_ms,
        )

    def _fallback(self, choice: Choice, error: str | None) -> Decision:
        option = fallback_option(choice, self.stance)
        return Decision(
            choice_id=choice.id,
            option_id=option.id,
            source="fallback",
            reasoning="",
            error=error,
        )
