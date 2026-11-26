"""Scenario runner: plays a scenario through the decider inside a run session.

Every step becomes an event in the run's log, so the same run can later feed memory,
the overlay and the editing markers.
"""

from __future__ import annotations

from collections.abc import Callable

from aibecomeshuman.cognition.context import SceneContext
from aibecomeshuman.cognition.decider import Decider, Decision
from aibecomeshuman.config import ConfigError
from aibecomeshuman.env.scenario import ChoiceStep, DialogueStep, ObservationStep, Scenario
from aibecomeshuman.prompts import Directive, PromptLibrary
from aibecomeshuman.run.session import RunSession

Echo = Callable[[str], None]


class ScenarioRunner:
    def __init__(
        self,
        session: RunSession,
        scenario: Scenario,
        directive: str | None = None,
        echo: Echo | None = None,
    ) -> None:
        self.session = session
        self.scenario = scenario
        self._echo = echo or (lambda _line: None)
        profile = session.profile
        if scenario.character not in profile.characters:
            raise ConfigError(
                f"scenario character '{scenario.character}' is not in run "
                f"'{profile.name}' characters {profile.characters}"
            )
        self.prompts = PromptLibrary(session.project.prompts_dir)
        self.directive: Directive = self.prompts.directive(directive or profile.directive)
        self.ctx = SceneContext(scenario.id, scenario.location)
        self.decider = Decider(
            prompts=self.prompts,
            character=scenario.character,
            directive=self.directive,
            output_language=profile.output_language,
            strong=session.llm("decider"),
            fast=session.llm("decider_fast") if session.has_role("decider_fast") else None,
            execution_margin_s=session.project.app.timing.execution_margin_s,
        )

    @property
    def _events(self):
        assert self.session.events is not None
        return self.session.events

    def change_directive(self, name: str) -> None:
        """Swap the run directive mid-scene (operator action)."""
        old = self.directive.name
        self.directive = self.prompts.directive(name)
        self.decider.set_directive(self.directive)
        self._events.emit(
            "directive.changed", source="operator", payload={"from": old, "to": name}
        )
        self._echo(f"!! Directive changed: {old} -> {name}")

    def run(self) -> list[Decision]:
        s = self.scenario
        self._events.emit(
            "scene.started",
            source="env",
            payload={
                "scenario": s.id,
                "title": s.title,
                "character": s.character,
                "location": s.location,
                "directive": self.directive.name,
                "synthetic": s.synthetic,
            },
        )
        self._echo(f"== {s.title} ({s.character}) · directive: {self.directive.name} ==")

        decisions: list[Decision] = []
        for step in s.steps:
            if isinstance(step, ObservationStep):
                entry = self.ctx.add("observation", step.text)
                self._perceived(entry.id, "observation", step.text)
                self._echo(f"  [{entry.id}] {step.text}")
                if step.significant:
                    self._update_stance()
            elif isinstance(step, DialogueStep):
                entry = self.ctx.add("dialogue", step.text, speaker=step.speaker)
                self._perceived(entry.id, "dialogue", step.text, speaker=step.speaker)
                self._echo(f'  [{entry.id}] {step.speaker}: "{step.text}"')
                if step.significant:
                    self._update_stance()
            elif isinstance(step, ChoiceStep):
                decisions.append(self._play_choice(step))

        self._events.emit("scene.finished", source="env", payload={"scenario": s.id})
        return decisions

    # -- internals -------------------------------------------------------------

    def _perceived(self, entry_id: str, kind: str, text: str, speaker: str | None = None) -> None:
        payload = {"entry_id": entry_id, "kind": kind, "text": text}
        if speaker:
            payload["speaker"] = speaker
        self._events.emit("scene.perceived", source="env", payload=payload)

    def _update_stance(self) -> None:
        stance = self.decider.update_stance(self.ctx)
        if stance is None:
            self._events.emit("stance.failed", source="cognition", payload={})
            return
        if not self.decider.last_stance_changed:
            self._events.emit("stance.kept", source="cognition", payload={})
            self._echo("  >> (stance unchanged)")
            return
        self._events.emit("stance.updated", source="cognition", payload=stance.model_dump())
        self._echo(f"  >> Stance: {stance.stance}")

    def _play_choice(self, step: ChoiceStep) -> Decision:
        choice = step.to_choice()
        # A timed choice needs a stance to lean on; build one off the clock if missing.
        if choice.timed and self.decider.stance is None:
            self._update_stance()

        self._events.emit("choice.presented", source="env", payload=choice.model_dump())
        timer = f" ({choice.seconds:g} s)" if choice.timed else ""
        labels = "  ".join(f"{o.id}) {o.label}" for o in choice.options)
        self._echo(f"  ?? Choice{timer}: {labels}")

        decision = self.decider.decide(choice, self.ctx)
        label = next(o.label for o in choice.options if o.id == decision.option_id)
        self._events.emit(
            "decision.made",
            source="cognition",
            payload=decision.model_dump() | {"label": label, "directive": self.directive.name},
        )
        self._echo(f"  => {decision.option_id}) {label}  [{_describe(decision)}]")
        if decision.reasoning:
            self._echo(f"     {decision.reasoning}")

        self.ctx.add("decision", f"I chose: {label}")
        outcome = step.outcome_of(decision.option_id)
        entry = self.ctx.add("outcome", outcome)
        self._perceived(entry.id, "outcome", outcome)
        self._echo(f"  [{entry.id}] {outcome}")
        self._update_stance()
        return decision


def _describe(decision: Decision) -> str:
    if decision.source == "fallback":
        if decision.timed_out:
            return f"fallback: model too slow ({decision.latency_ms:.0f} ms)"
        return f"fallback: {decision.error or 'no model answer'}"
    parts = [decision.role or "model", f"{decision.latency_ms:.0f} ms"]
    if decision.role == "decider":
        parts.append("evidence ok" if decision.evidence_ok else "evidence MISSING")
    return ", ".join(parts)
