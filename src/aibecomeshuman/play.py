"""Scenario runner: plays scenes through the decider inside a run session, with memory.

Every step becomes an event in the run's log, so the same run can later feed the overlay
and the editing markers. At the end of each scene the character's memory is consolidated
and its personality reflected on; both are saved to the campaign (or to the run folder
when no campaign is given).
"""

from __future__ import annotations

from collections.abc import Callable

from aibecomeshuman.cognition.context import SceneContext
from aibecomeshuman.cognition.decider import Choice, Decider, Decision
from aibecomeshuman.cognition.guard import Judge, Lexicon, Statement
from aibecomeshuman.cognition.reflection import SceneReport, consolidate, reflect
from aibecomeshuman.config import ConfigError
from aibecomeshuman.core.events import EventLog
from aibecomeshuman.env.scenario import (
    ChoiceStep,
    Condition,
    DialogueStep,
    ObservationStep,
    Scenario,
)
from aibecomeshuman.llm.types import LLMError
from aibecomeshuman.memory.models import CharacterMemory, Personality, Trait
from aibecomeshuman.memory.render import render_memory, render_personality
from aibecomeshuman.memory.store import CampaignState, MemoryStore, check_campaign_name
from aibecomeshuman.prompts import Directive, PromptLibrary
from aibecomeshuman.run.session import RunSession

Echo = Callable[[str], None]


class Campaign:
    """Memory and choices for one playthrough, shared by every scene of a run."""

    def __init__(
        self, session: RunSession, name: str | None = None, folder: str = "memory"
    ) -> None:
        self.name = check_campaign_name(name) if name else None
        project = session.project
        assert session.dir is not None
        # Without a campaign, memory lives in the run folder (`folder` keeps cases apart).
        directory = project.campaigns_dir / self.name if self.name else session.dir / folder
        self.store = MemoryStore(directory)
        self.prompts = PromptLibrary(project.prompts_dir)
        self.state: CampaignState = self.store.load_state()
        self._memories: dict[str, CharacterMemory] = {}

    def memory_for(self, character: str) -> CharacterMemory:
        if character not in self._memories:

            def initial() -> Personality:
                traits = self.prompts.character_traits(character)
                return Personality.from_traits(
                    {k: Trait(value=v, meaning=m) for k, (v, m) in traits.items()}
                )

            self._memories[character] = self.store.load(character, initial)
        return self._memories[character]

    def save(self, character: str) -> None:
        self.store.save(self.memory_for(character))
        self.store.save_state(self.state)


class ScenarioRunner:
    def __init__(
        self,
        session: RunSession,
        scenario: Scenario,
        directive: str | None = None,
        echo: Echo | None = None,
        campaign: Campaign | None = None,
        briefing: str = "",
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
        self.campaign = campaign or Campaign(session)
        self.memory = self.campaign.memory_for(scenario.character)
        self.ctx = SceneContext(scenario.id, scenario.location, briefing=briefing)
        self.decider = Decider(
            prompts=self.prompts,
            character=scenario.character,
            directive=self.directive,
            output_language=profile.output_language,
            strong=session.llm("decider"),
            fast=session.llm("decider_fast") if session.has_role("decider_fast") else None,
            execution_margin_s=session.project.app.timing.execution_margin_s,
        )
        self._decisions: list[tuple[Decision, str]] = []
        # Anti-metagaming guard: free word check always; the judge when the run has one.
        self.lexicon = Lexicon.load(session.project.root / "config" / "guard.toml")
        self.judge = (
            Judge(session.llm("judge"), self.prompts.role("judge"))
            if session.has_role("judge")
            else None
        )
        self._statements: list[Statement] = []
        self._choices_seen: list[str] = []
        # Shadow mode: the operator picks, so the personality does not learn from the picks.
        self.operator_controlled = False

    @property
    def _events(self) -> EventLog:
        assert self.session.events is not None
        return self.session.events

    @property
    def _memory_settings(self):
        return self.session.project.app.memory

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
        self.begin()
        for step in self.scenario.steps:
            if step.when is not None and not self._condition_met(step.when):
                continue
            if isinstance(step, ObservationStep):
                self.perceive("observation", step.text, significant=step.significant)
            elif isinstance(step, DialogueStep):
                self.perceive("dialogue", step.text, step.speaker, step.significant)
            elif isinstance(step, ChoiceStep):
                self._play_choice(step)
        self.finish()
        return [d for d, _ in self._decisions]

    # -- live API (also used by run) ---------------------------------------------

    def perceive(
        self, kind: str, text: str, speaker: str | None = None, significant: bool = False
    ) -> None:
        entry = self.ctx.add(kind, text, speaker=speaker)  # type: ignore[arg-type]
        self._perceived(entry.id, kind, text, speaker=speaker)
        self._echo(f'  [{entry.id}] {speaker}: "{text}"' if speaker else f"  [{entry.id}] {text}")
        if significant:
            self._update_stance()

    def suggest(self, choice: Choice) -> tuple[Decision, str]:
        """Shadow mode: decide what the character would pick, without it becoming a fact
        (the operator plays; `record_pick` stores what was really picked)."""
        if choice.timed and self.decider.stance is None:
            self._update_stance()
        self._events.emit(
            "choice.presented", source="env", payload=choice.model_dump() | {"mode": "shadow"}
        )
        self._choices_seen.append(
            f"{choice.prompt} options: {'  '.join(o.label for o in choice.options)}".strip()
        )
        self._refresh_memory()
        decision = self.decider.decide(choice, self.ctx)
        label = next(o.label for o in choice.options if o.id == decision.option_id)
        self._events.emit(
            "decision.made",
            source="cognition",
            payload=decision.model_dump()
            | {"label": label, "directive": self.directive.name, "mode": "shadow"},
        )
        self._echo(f"  => SUGGESTION: {label}  [{_describe(decision)}]")
        if decision.reasoning:
            self._echo(f"     {decision.reasoning}")
            self._check("decision", f"decision-{choice.id}", decision.reasoning)
        return decision, label

    def record_pick(self, label: str) -> None:
        """What the operator actually picked: this is what the character did."""
        self.ctx.add("decision", f"I chose: {label}")
        self._update_stance()

    def finish(self) -> None:
        self._events.emit("scene.finished", source="env", payload={"scenario": self.scenario.id})
        self._audit_scene()
        self._end_of_scene()

    def begin(self) -> None:
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
                "campaign": self.campaign.name,
                "memories": len(self.memory.items),
                "personality_version": self.memory.personality.version,
                "briefing": self.ctx.briefing,
            },
        )
        self._echo(f"== {s.title} ({s.character}) · directive: {self.directive.name} ==")
        if self.ctx.briefing:
            self._echo(f"   briefing: {self.ctx.briefing}")
        if self.memory.items:
            self._echo(f"   (remembers {len(self.memory.items)} earlier scene(s))")

    # -- memory ----------------------------------------------------------------

    def _condition_met(self, condition: Condition) -> bool:
        key = self.scenario.qualify(condition.choice)
        chosen = self.campaign.state.choices.get(key)
        if chosen is None:
            self._events.emit(
                "scene.condition_unknown", source="env", payload={"choice": key}
            )
            return False
        return chosen in condition.any_of

    def _refresh_memory(self) -> None:
        self.decider.set_memory(
            render_personality(self.memory),
            render_memory(self.memory, self.ctx.render(), self._memory_settings),
            self.memory.ids,
        )

    def _end_of_scene(self) -> None:
        character = self.scenario.character
        report: SceneReport | None = None
        if self.session.has_role("summarizer"):
            try:
                report = consolidate(
                    self.session.llm("summarizer"),
                    self.decider.system("summarizer"),
                    self.memory,
                    self.ctx,
                    self.scenario.title,
                    self._memory_settings,
                    self.session.dir.name if self.session.dir else None,
                )
            except LLMError as e:
                self._events.emit("memory.failed", source="memory", payload={"error": str(e)})
                self._echo(f"  ## Memory not saved: {e}")
        if report is not None:
            self._events.emit(
                "memory.consolidated",
                source="memory",
                payload={
                    "character": character,
                    "item": report.item.model_dump(),
                    "new_promises": [c.model_dump() for c in report.new_commitments],
                    "resolved_promises": [c.model_dump() for c in report.resolved_commitments],
                    "entities": report.entities_touched,
                },
            )
            self._echo(f"  ## Memory {report.item.id} ({report.item.importance}/5): "
                       f"{report.item.text}")
            for c in report.new_commitments:
                self._echo(f"  ## New promise {c.id} to {c.to or '?'}: {c.text}")
            for c in report.resolved_commitments:
                self._echo(f"  ## Promise {c.id} {c.status}: {c.note}")
            self._check("memory", report.item.id, report.item.text, audit=False)
            self._reflect(report)
        self.campaign.save(character)

    def _reflect(self, report: SceneReport) -> None:
        if self.operator_controlled:
            self._events.emit(
                "personality.skipped",
                source="memory",
                payload={"reason": "the operator made the choices (shadow mode)"},
            )
            self._echo("  ## Personality unchanged: the operator made the choices")
            return
        if not self.directive.shapes_personality:
            self._events.emit(
                "personality.skipped",
                source="memory",
                payload={"reason": f"directive '{self.directive.name}' does not shape personality"},
            )
            self._echo("  ## Personality unchanged: this directive's choices were imposed")
            return
        if not self.session.has_role("reflector"):
            return
        try:
            changes = reflect(
                self.session.llm("reflector"),
                self.decider.system("reflector"),
                self.memory,
                report,
                self._decisions,
                self._memory_settings,
                review=self._review_reflection,
            )
        except LLMError as e:
            self._events.emit("personality.failed", source="memory", payload={"error": str(e)})
            return
        p = self.memory.personality
        self._events.emit(
            "personality.updated" if changes else "personality.unchanged",
            source="memory",
            payload={
                "character": self.scenario.character,
                "version": p.version,
                "changes": [c.model_dump() for c in changes],
                "traits": {k: t.value for k, t in p.traits.items()},
                "self_description": p.self_description,
            },
        )
        for c in changes:
            self._echo(f"  ## {c.trait}: {c.old:g} -> {c.new:g} · {c.reason}")
        if not changes:
            self._echo("  ## Personality unchanged")

    # -- anti-metagaming guard ---------------------------------------------------

    def _record(self) -> str:
        """Everything the character may legitimately know right now."""
        return "\n\n".join(
            [
                f"<character_sheet>\n{self.prompts.character(self.scenario.character)}\n"
                "</character_sheet>",
                f"<memory>\n{self.memory.model_dump_json(exclude={'personality'})}\n</memory>",
                f"<scene>\n{self.ctx.render()}\n</scene>",
                "<choices_presented>\n" + "\n".join(self._choices_seen) + "\n</choices_presented>",
            ]
        )

    def _check(self, kind: str, statement_id: str, text: str, audit: bool = True) -> list[str]:
        """Free word check now; queue the statement for the end-of-scene judge audit."""
        if not text.strip():
            return []
        terms = self.lexicon.unexplained(text, self._record() + self.directive.text)
        if terms:
            self._events.emit(
                "guard.flag",
                source="guard",
                payload={"kind": kind, "id": statement_id, "terms": terms, "text": text},
            )
            self._echo(f"  !! Guard: {kind} uses {', '.join(terms)} without having met it")
        if audit:
            self._statements.append(Statement(id=statement_id, kind=kind, text=text))
        return terms

    def _audit_scene(self) -> None:
        """The judge reads the whole scene and every stance and reason the agent wrote."""
        if self.judge is None or not self._statements:
            return
        try:
            verdicts = self.judge.review(self._record(), self._statements)
        except LLMError as e:
            self._events.emit("guard.failed", source="guard", payload={"error": str(e)})
            return
        flagged = [v for v in verdicts if not v.grounded]
        self._events.emit(
            "guard.audit",
            source="guard",
            payload={
                "checked": len(self._statements),
                "flagged": [v.model_dump() for v in flagged],
                "texts": {s.id: s.text for s in self._statements if s.id in
                          {v.id for v in flagged}},
            },
        )
        for v in flagged:
            self._echo(f"  !! Judge: {v.id} not grounded: {'; '.join(v.unsupported)}")
        self._statements.clear()

    def _review_reflection(self, proposal: dict) -> dict:
        """Drop personality changes whose reasons did not happen (or use outside words)."""
        changes = [c for c in proposal.get("trait_changes", []) if c.get("reason")]
        rejected: dict[str, str] = {}
        for c in changes:
            if self._check("reflection", f"trait-{c.get('trait')}", c["reason"], audit=False):
                rejected[c.get("trait", "?")] = "uses words it never met"
        if self.judge is not None and changes:
            statements = [
                Statement(id=f"trait-{c.get('trait')}", kind="reflection", text=c["reason"])
                for c in changes
            ]
            try:
                for v in self.judge.review(self._record(), statements):
                    if not v.grounded:
                        rejected[v.id.removeprefix("trait-")] = "; ".join(v.unsupported)
            except LLMError as e:
                self._events.emit("guard.failed", source="guard", payload={"error": str(e)})
        description = str(proposal.get("self_description", ""))
        if description and self._check("reflection", "self-description", description, False):
            proposal = proposal | {"self_description": ""}
        if rejected:
            self._events.emit(
                "personality.rejected",
                source="guard",
                payload={
                    "changes": [
                        {"trait": c.get("trait"), "reason": c["reason"],
                         "why": rejected[c.get("trait", "?")]}
                        for c in changes if c.get("trait") in rejected
                    ]
                },
            )
            for trait, why in rejected.items():
                self._echo(f"  !! Change to {trait} rejected: {why}")
            proposal = proposal | {
                "trait_changes": [
                    c for c in proposal.get("trait_changes", []) if c.get("trait") not in rejected
                ]
            }
        return proposal

    # -- steps -----------------------------------------------------------------

    def _perceived(self, entry_id: str, kind: str, text: str, speaker: str | None = None) -> None:
        payload = {"entry_id": entry_id, "kind": kind, "text": text}
        if speaker:
            payload["speaker"] = speaker
        self._events.emit("scene.perceived", source="env", payload=payload)

    def _update_stance(self) -> None:
        self._refresh_memory()
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
        concerns = " ".join(getattr(stance, "concerns", []) or [])
        self._check("stance", f"stance-{len(self._statements) + 1}", f"{stance.stance} {concerns}")

    def _play_choice(self, step: ChoiceStep) -> None:
        choice = step.to_choice()
        # A timed choice needs a stance to lean on; build one off the clock if missing.
        if choice.timed and self.decider.stance is None:
            self._update_stance()

        self._events.emit("choice.presented", source="env", payload=choice.model_dump())
        timer = f" ({choice.seconds:g} s)" if choice.timed else ""
        labels = "  ".join(f"{o.id}) {o.label}" for o in choice.options)
        self._echo(f"  ?? Choice{timer}: {labels}")
        self._choices_seen.append(f"{choice.prompt}{timer} options: {labels}".strip())

        self._refresh_memory()
        decision = self.decider.decide(choice, self.ctx)
        label = next(o.label for o in choice.options if o.id == decision.option_id)
        self._decisions.append((decision, label))
        self.campaign.state.choices[self.scenario.qualify(step.id)] = decision.option_id
        self._events.emit(
            "decision.made",
            source="cognition",
            payload=decision.model_dump() | {"label": label, "directive": self.directive.name},
        )
        self._echo(f"  => {decision.option_id}) {label}  [{_describe(decision)}]")
        if decision.reasoning:
            self._echo(f"     {decision.reasoning}")
            self._check("decision", f"decision-{step.id}", decision.reasoning)

        self.ctx.add("decision", f"I chose: {label}")
        outcome = step.outcome_of(decision.option_id)
        # Scenes taken from recordings have no separate outcome: the next steps show it.
        if outcome:
            entry = self.ctx.add("outcome", outcome)
            self._perceived(entry.id, "outcome", outcome)
            self._echo(f"  [{entry.id}] {outcome}")
        self._update_stance()


def play_scenarios(
    session: RunSession,
    scenarios: list[Scenario],
    directive: str | None = None,
    echo: Echo | None = None,
    campaign: str | None = None,
) -> list[Decision]:
    """Play scenes in order in one run; memory carries over from one to the next."""
    shared = Campaign(session, campaign)
    decisions: list[Decision] = []
    for scenario in scenarios:
        runner = ScenarioRunner(session, scenario, directive=directive, echo=echo, campaign=shared)
        decisions += runner.run()
    return decisions


def _describe(decision: Decision) -> str:
    if decision.source == "fallback":
        if decision.timed_out:
            return f"fallback: model too slow ({decision.latency_ms:.0f} ms)"
        return f"fallback: {decision.error or 'no model answer'}"
    parts = [decision.role or "model", f"{decision.latency_ms:.0f} ms"]
    if decision.role == "decider":
        parts.append("evidence ok" if decision.evidence_ok else "evidence MISSING")
    return ", ".join(parts)
