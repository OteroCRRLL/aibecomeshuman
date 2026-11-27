"""Shadow mode: the operator plays, the agent watches live and suggests every choice.

    source frames --(local signals)--> keyframes --(observer workers)--> observations
        --(in order: redaction, transcript)--> the character (ScenarioRunner live API)
    menus skip the line: a dedicated worker reads them at once and the character is asked
    for a suggestion right away, while the countdown is still running.

What the operator really picks is recorded as what the character did (memory is kept);
the personality does not learn from those picks. At the end-of-chapter flowchart the
scene is closed (memory saved) and a new one starts.
"""

from __future__ import annotations

import itertools
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from aibecomeshuman.cognition.decider import Choice, Option
from aibecomeshuman.env.scenario import ObservationStep, Scenario
from aibecomeshuman.live.sources import FrameSource, SourceLost
from aibecomeshuman.llm.types import LLMError, LLMRequest, Message
from aibecomeshuman.perception.escalate import Escalator
from aibecomeshuman.perception.observer import Observation, Observer, ObserverContext
from aibecomeshuman.perception.pipeline import CONTEXT_FRAME_WIDTH, CONTEXT_FRAMES
from aibecomeshuman.perception.redact import Redactor
from aibecomeshuman.perception.transcript import (
    TranscriptBuilder,
    TranscriptItem,
    infer_picks,
    similar,
)
from aibecomeshuman.perception.video import (
    encode_jpeg,
    fmt_time,
    select_keyframes,
    shrink_jpeg,
)
from aibecomeshuman.play import Campaign, ScenarioRunner
from aibecomeshuman.prompts import PromptLibrary
from aibecomeshuman.run.session import RunSession

Echo = Callable[[str], None]

MENU, ITEM, CHAPTER_END = 0, 1, 1  # queue priorities: menus first
# Lines heard after a menu before deciding what the operator picked.
PICK_LINES = 3

PICK_SCHEMA = {
    "type": "object",
    "properties": {"picked": {"type": "string"}},
    "required": ["picked"],
    "additionalProperties": False,
}


def guess_pick(provider, choice_item: TranscriptItem, after: list[TranscriptItem]) -> str:
    """Which option was picked, judging from the lines that followed (cheap text call)."""
    options = [o.label for o in choice_item.options if not o.locked]
    lines = "\n".join(f'{i.speaker}: "{i.text}"' for i in after)
    request = LLMRequest(
        purpose="shadow.pick",
        system="You read a short exchange right after a dialogue menu and say which option "
        "the protagonist picked: their next line usually puts it into words. Answer with "
        "the option exactly as written, or an empty string if it is not clear.",
        messages=[Message.user(f"options: {' | '.join(options)}\nwhat followed:\n{lines}")],
        json_schema=PICK_SCHEMA,
        max_tokens=200,
    )
    try:
        answer = str((provider.complete(request).parsed or {}).get("picked", ""))
    except LLMError:
        return ""
    return next((o for o in options if similar(o, answer) >= 0.85), "")


@dataclass
class ShadowReport:
    keyframes: int = 0
    observed: int = 0
    failed: int = 0
    suggestions: int = 0
    compared: int = 0
    agreed: int = 0
    scenes: int = 0
    comparisons: list[dict] = field(default_factory=list)


@dataclass
class _Suggestion:
    labels: list[str]
    suggested: str
    buttons: dict[str, str]


class ShadowSession:
    def __init__(
        self,
        session: RunSession,
        source: FrameSource,
        character: str | None = None,
        campaign: str | None = None,
        echo: Echo | None = None,
        stop: threading.Event | None = None,
        chapter: int | None = None,
    ) -> None:
        """`character` None = the protagonist of the chapter's briefing (connor if none).
        `chapter` None = continue where the campaign left off."""
        assert session.events is not None and session.dir is not None
        self.session = session
        self.source = source
        self.fixed_character = character is not None
        self.character = character or "connor"
        self.say = echo or (lambda _: None)
        self.stop = stop or threading.Event()
        self.settings = session.project.app.perception
        self.live = session.project.app.live
        self.report = ShadowReport()
        self.campaign = Campaign(session, campaign)
        self.events = session.events
        self.frames_dir = session.dir / "frames"
        self.frames_dir.mkdir(exist_ok=True)

        library = PromptLibrary(session.project.prompts_dir)
        self.library = library
        if chapter is not None:
            self.campaign.state.chapter = chapter
        briefing = library.briefing(self.campaign.state.chapter)
        if briefing is not None and not self.fixed_character:
            self.character = briefing.character
        self.observer = Observer(session.llm("observer"), library.role("observer"))
        self.escalator: Escalator | None = None
        if session.has_role("observer_strong"):

            def strong_spent() -> float:
                usage = session.manifest.usage_by_role.get("observer_strong")  # type: ignore[union-attr]
                return usage.cost_usd if usage else 0.0

            self.escalator = Escalator(
                Observer(session.llm("observer_strong"), library.role("observer")),
                self.live.strong_cap_per_chapter_usd,
                strong_spent,
            )
        self.redactor = Redactor(self.settings.hidden_patterns)
        self.ctx = ObserverContext(protagonist=self.character.capitalize())
        self.builder = TranscriptBuilder(
            self.character.capitalize(), 1.0 / self.live.frames_per_s
        )

        self._lock = threading.Lock()
        self._results: dict[int, Observation | None] = {}
        self._next_seq = 0
        self._recent: list[tuple[float, bytes]] = []
        self._agent_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._order = itertools.count()
        self._items_sent = 0
        self._suggested: list[_Suggestion] = []
        self._in_flowchart = False
        self._last_menu_t = -1e9
        self._last_no_menu_t = -1e9

    # -- main loop ----------------------------------------------------------------

    def run(self) -> ShadowReport:
        self.events.emit(
            "shadow.started",
            source="live",
            payload={"source": self.source.name, "character": self.character,
                     "campaign": self.campaign.name},
        )
        agent = threading.Thread(target=self._agent_loop, daemon=True)
        agent.start()
        pool = ThreadPoolExecutor(max_workers=self.live.workers)
        menu_lane = ThreadPoolExecutor(max_workers=1)
        seqs = itertools.count()
        try:
            frames = self.source.frames(1.0 / self.live.frames_per_s)
            for kf in select_keyframes(self._until_stopped(frames), self.settings):
                seq = next(seqs)
                self.report.keyframes += 1
                name = f"t{int(kf.t):05d}_{kf.reason}.jpg"
                jpeg = encode_jpeg(kf.image)
                (self.frames_dir / name).write_bytes(jpeg)
                with self._lock:
                    ctx = self.ctx.model_copy(deep=True)
                    earlier = [(fmt_time(t), small) for t, small in self._recent if kf.t - t <= 6]
                    self._recent = [*self._recent, (kf.t, shrink_jpeg(jpeg, CONTEXT_FRAME_WIDTH))][
                        -CONTEXT_FRAMES:
                    ]
                in_burst = bool(self.escalator and self.escalator.note_keyframe(kf.t, kf.reason))
                lane = menu_lane if kf.reason == "menu" else pool
                lane.submit(
                    self._observe, seq, kf.t, name, jpeg, ctx, earlier, time.monotonic(), in_burst
                )
        except SourceLost as e:
            self.events.emit("shadow.source_lost", source="live", payload={"reason": str(e)})
            self.say("  -- the game is no longer visible: ending and saving memory --")
        finally:
            self.source.close()
            pool.shutdown(wait=True)
            menu_lane.shutdown(wait=True)
            self._agent_queue.put((ITEM + 1, next(self._order), ("end", None)))
            agent.join()
            self.events.emit("shadow.finished", source="live", payload=self.report.__dict__)
        return self.report

    def _until_stopped(self, frames):
        for item in frames:
            if self.stop.is_set():
                return
            yield item

    # -- perception (worker threads) ----------------------------------------------

    def _observe(self, seq, t, name, jpeg, ctx, earlier, captured_at, in_burst=False) -> None:
        obs: Observation | None
        try:
            obs = self.observer.observe(jpeg, t, fmt_time(t), ctx, earlier)
            obs.frame = name
        except LLMError as e:
            self.events.emit("perception.failed", "perception", {"t": t, "error": str(e)})
            obs = None
        if obs is not None and self.escalator is not None and self.escalator.wanted(obs, in_burst):
            try:
                strong = self.escalator.strong.observe(jpeg, t, fmt_time(t), ctx, earlier)
                strong.frame = name
                self.events.emit(
                    "perception.escalated",
                    source="perception",
                    payload={"t": t, "frame": name, "cheap": obs.description,
                             "strong": strong.description,
                             "chapter_spend_usd": round(self.escalator.chapter_spend(), 4)},
                )
                obs = strong
            except LLMError as e:  # keep the cheap reading
                self.events.emit("perception.failed", "perception", {"t": t, "error": str(e)})
        visible = None
        if obs is not None:
            visible, hidden = self.redactor.apply(obs)
            self.events.emit(
                "perception.observed",
                source="perception",
                payload={"t": t, "frame": name, "visible": visible is not None, "hidden": hidden,
                         "observation": obs.model_dump(mode="json", exclude={"t", "frame"})},
            )
            if visible is not None and visible.choice.present:
                self._menu_seen(visible, captured_at)  # right away, not in order
        with self._lock:
            if obs is None:
                self.report.failed += 1
            else:
                self.report.observed += 1
            self._results[seq] = obs
            self._feed_in_order()

    def _feed_in_order(self) -> None:
        """Called with the lock held: pass finished observations on, in frame order."""
        while self._next_seq in self._results:
            obs = self._results.pop(self._next_seq)
            self._next_seq += 1
            if obs is None:
                continue
            if obs.screen == "flowchart":
                if not self._in_flowchart:
                    self._in_flowchart = True
                    self._agent_queue.put((CHAPTER_END, next(self._order), ("chapter_end", None)))
                continue
            visible, _ = self.redactor.apply(obs)
            if visible is None:
                continue  # pause menu etc.: says nothing about the game's own menu
            if not visible.choice.present:
                self._last_no_menu_t = max(self._last_no_menu_t, visible.t)
            self._in_flowchart = False
            self._update_context(visible)
            self.builder.add(visible)
            for item in self.builder.items[self._items_sent :]:
                self._agent_queue.put((ITEM, next(self._order), ("item", item)))
            self._items_sent = len(self.builder.items)

    def _update_context(self, visible: Observation) -> None:
        if visible.description:
            self.ctx.previous_description = visible.description
        for line in visible.subtitles:
            self.ctx.recent_lines = [*self.ctx.recent_lines, f"{line.speaker}: {line.text}"][-6:]
        for n in visible.names_seen:
            if n.name.strip():
                self.ctx.known_people.setdefault(n.name.strip(), n.who.strip())

    def _menu_seen(self, visible: Observation, captured_at: float) -> None:
        labels = [o.label.strip() for o in visible.choice.options if not o.locked and o.label]
        if len(labels) < 2:
            return
        with self._lock:
            # A menu is new only if a frame WITHOUT a menu was seen since the last sighting
            # (the operator may take their time; the same menu must not be answered again).
            closed_since = self._last_no_menu_t > self._last_menu_t
            self._last_menu_t = max(self._last_menu_t, visible.t)
            same = self._suggested and self._overlaps(self._suggested[-1].labels, labels)
            if same and not closed_since:
                return
            self._suggested.append(_Suggestion(labels, "", {}))
        buttons = {o.label.strip(): o.button for o in visible.choice.options}
        # The countdown started when the frame was captured, not when it was read.
        payload = (labels, buttons, visible.choice.timed, captured_at, visible.choice.about)
        self._agent_queue.put((MENU, next(self._order), ("menu", payload)))

    @staticmethod
    def _overlaps(a: list[str], b: list[str]) -> bool:
        return any(similar(x, y) >= 0.85 for x in a for y in b)

    # -- the character (agent thread) ----------------------------------------------

    def _new_runner(self) -> ScenarioRunner:
        """A scene for the current chapter: its briefing, and its protagonist if known."""
        self.report.scenes += 1
        n = self.report.scenes
        chapter = self.campaign.state.chapter
        briefing = self.library.briefing(chapter)
        if briefing is not None and not self.fixed_character:
            self.character = briefing.character
        # The observer and the transcript follow the chapter's protagonist.
        self.ctx.protagonist = self.character.capitalize()
        self.ctx.appearance = self.library.appearance(self.character)
        self.builder.protagonist = self.character.capitalize()
        title = f"Chapter {chapter}" + (f": {briefing.title}" if briefing else "")
        scenario = Scenario(
            id=f"live-{self.session.dir.name}-{n}",  # type: ignore[union-attr]
            title=title,
            character=self.character,
            synthetic=False,
            steps=[ObservationStep(kind="observation", text="(live)")],
        )
        self.events.emit(
            "shadow.chapter",
            source="live",
            payload={"chapter": chapter, "title": briefing.title if briefing else "",
                     "character": self.character, "briefing": bool(briefing)},
        )
        runner = ScenarioRunner(
            self.session, scenario, echo=self.say, campaign=self.campaign,
            briefing=briefing.text if briefing else "",
        )
        runner.operator_controlled = True
        runner.begin()
        return runner

    def _agent_loop(self) -> None:
        runner = self._new_runner()
        since_stance = 0
        pending: TranscriptItem | None = None
        after_pick: list[TranscriptItem] = []
        choice_n = 0
        while True:
            _, _, (kind, payload) = self._agent_queue.get()
            try:
                if kind == "end":
                    if pending is not None:
                        self._resolve_pick(runner, pending, after_pick)
                    runner.finish()
                    return
                if kind == "chapter_end":
                    self.say("  -- end of chapter: saving memory --")
                    runner.finish()
                    self.campaign.state.chapter += 1  # the game's chapters come in order
                    self.campaign.save(self.character)
                    if self.escalator is not None:
                        self.escalator.new_chapter()
                    runner = self._new_runner()
                    continue
                if kind == "menu":
                    choice_n += 1
                    self._suggest(runner, choice_n, *payload)
                    continue
                item: TranscriptItem = payload
                if item.kind == "choice":
                    if pending is not None:
                        self._resolve_pick(runner, pending, after_pick)
                    pending, after_pick = item, []  # resolved once the next lines arrive
                    continue
                if item.kind == "dialogue":
                    runner.perceive("dialogue", item.text, item.speaker)
                    if pending is not None:
                        after_pick.append(item)
                        if len(after_pick) >= PICK_LINES:
                            self._resolve_pick(runner, pending, after_pick)
                            pending = None
                elif item.kind == "observation":
                    runner.perceive("observation", item.text)
                else:
                    runner.perceive("observation", f"My display shows: {item.text}")
                since_stance += 1
                if since_stance >= self.live.stance_every:
                    since_stance = 0
                    runner._update_stance()
            except Exception as e:  # keep watching even if one step fails
                self.events.emit("shadow.error", source="live", payload={"error": repr(e)})

    def _suggest(self, runner, n, labels, buttons, timed, seen_at, about="") -> None:
        elapsed = time.monotonic() - seen_at
        seconds = max(1.5, self.live.timed_choice_s - elapsed) if timed else None
        choice = Choice(
            id=f"live{n}",
            prompt=f"The options are about: {about}" if about else "",
            timed=timed,
            seconds=seconds,
            options=[Option(id=f"o{i}", label=label) for i, label in enumerate(labels, 1)],
        )
        decision, label = runner.suggest(choice)
        with self._lock:
            match = next((s for s in self._suggested if s.labels == labels), None)
            if match is not None:
                match.suggested, match.buttons = label, buttons
        self.report.suggestions += 1
        button = buttons.get(label, "")
        self.say(f"\n>>> {button} {label}  ({decision.reasoning or 'no reason'})\n")
        self.events.emit(
            "shadow.suggestion",
            source="live",
            payload={"options": labels, "suggested": label, "button": button,
                     "reasoning": decision.reasoning, "source": decision.source,
                     "seconds_after_menu": round(time.monotonic() - seen_at, 1)},
        )

    def _resolve_pick(self, runner, choice_item: TranscriptItem, after: list) -> None:
        """Work out what the operator picked: the words of the protagonist's next line
        first (most reliable), then a highlighted option, then a quick text-only guess."""
        highlighted = choice_item.chosen
        probe = choice_item.model_copy(update={"chosen": ""})
        infer_picks([probe, *after])
        picked, basis = probe.chosen, "words"
        if not picked and highlighted:
            picked, basis = highlighted, "highlight"
        if not picked and after:
            picked, basis = guess_pick(self.session.llm("observer"), choice_item, after), "guess"
        choice_item.chosen = picked
        labels = [o.label for o in choice_item.options]
        with self._lock:
            suggestion = next(
                (s for s in reversed(self._suggested) if self._overlaps(s.labels, labels)), None
            )
        suggested = suggestion.suggested if suggestion else ""
        if picked:
            runner.record_pick(picked, choice_item.about)
            self.ctx.just_chosen = picked
        comparison = {
            "t": choice_item.t,
            "options": labels,
            "suggested": suggested,
            "picked": picked,
            "picked_by": basis if picked else "",
            "agreed": bool(picked and suggested and similar(picked, suggested) >= 0.85),
        }
        if picked and suggested:
            self.report.compared += 1
            self.report.agreed += comparison["agreed"]
        self.report.comparisons.append(comparison)
        self.events.emit("shadow.compare", source="live", payload=comparison)
        verdict = "same" if comparison["agreed"] else "different"
        self.say(f"  -- you picked {picked or '(not seen)'}; suggested {suggested or '-'} "
                 f"({verdict})")
