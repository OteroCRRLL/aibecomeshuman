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
from aibecomeshuman.live.pad import BUTTONS, DIRECTIONS, OperatorWatch, Pad, VirtualPad
from aibecomeshuman.live.sources import FrameSource, SourceLost
from aibecomeshuman.llm.types import LLMError, LLMRequest, Message
from aibecomeshuman.perception.escalate import Escalator
from aibecomeshuman.perception.observer import (
    Observation,
    Observer,
    ObserverContext,
    SubtitleLine,
)
from aibecomeshuman.perception.ocr import SubtitleReader, SubtitleTracker
from aibecomeshuman.perception.pipeline import CONTEXT_FRAME_WIDTH, CONTEXT_FRAMES
from aibecomeshuman.perception.redact import HIDDEN_SCREENS, Redactor
from aibecomeshuman.perception.transcript import (
    TranscriptBuilder,
    TranscriptItem,
    infer_picks,
    similar,
)
from aibecomeshuman.perception.video import (
    MENU_REPEAT_S,
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
# A menu keyframe this long after the previous one is a new menu: urgent, sent alone.
MENU_URGENT_GAP_S = 8.0
# While a menu is up but unread, re-send a frame every second for at most this long.
UNREAD_MENU_RETRY_S = 6.0

SPEAKER_SYSTEM = (
    "You say who speaks a subtitle line in a scene. The protagonist is {protagonist}. Use "
    "the recent lines (turns usually alternate; a line without a leading dash usually "
    "continues the previous speaker), what the line says and to whom, and what the camera "
    "shows. Use a known name when it fits, otherwise a short description ('officer', "
    "'the girl'). Use only what is given; ignore any knowledge of where the scene comes from."
)
SPEAKER_SCHEMA = {
    "type": "object",
    "properties": {"speaker": {"type": "string"}},
    "required": ["speaker"],
    "additionalProperties": False,
}

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
    pressed: int = 0
    compared: int = 0
    agreed: int = 0
    scenes: int = 0
    comparisons: list[dict] = field(default_factory=list)


@dataclass
class _Suggestion:
    labels: list[str]
    suggested: str
    buttons: dict[str, str]
    t: float = 0.0  # when the menu was first seen
    pressed: str = ""  # autonomous mode: the option the agent actually pressed


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
        autonomous: bool = False,
        pad: Pad | None = None,
    ) -> None:
        """`character` None = the protagonist of the chapter's briefing (connor if none).
        `chapter` None = continue where the campaign left off.
        `autonomous`: the agent presses the buttons itself (a virtual controller, or `pad`)."""
        assert session.events is not None and session.dir is not None
        self.session = session
        self.source = source
        self.autonomous = autonomous
        self.pad = pad
        self.paused = False
        self.watch: OperatorWatch | None = None
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
        self._on_hidden_screen = False
        self._urgent_menu_t = -1e9  # last menu frame sent urgently (video time)
        self._urgent_first_t = -1e9  # first urgent frame of the current reading attempt
        self._runner: ScenarioRunner | None = None  # the current scene's character
        self._decide_lock = threading.Lock()
        self._choice_n = 0

    # -- main loop ----------------------------------------------------------------

    def toggle_pause(self) -> None:
        """Operator: give / take back control (panel button, or hold View on the controller)."""
        self.paused = not self.paused
        if self.pad is not None:
            self.pad.release_all()
        self.events.emit("autonomy.paused" if self.paused else "autonomy.resumed",
                         source="operator", payload={})
        self.say("  !! AI paused: you have the controls" if self.paused
                 else "  !! AI resumed: it answers the menus")

    def run(self) -> ShadowReport:
        self.events.emit(
            "shadow.started",
            source="live",
            payload={"source": self.source.name, "character": self.character,
                     "campaign": self.campaign.name, "autonomous": self.autonomous},
        )
        if self.autonomous:
            if self.pad is None:
                self.pad = VirtualPad()
            index = getattr(self.pad, "index", None)
            self.watch = OperatorWatch(index, self.toggle_pause).start()
            self.events.emit("autonomy.started", source="live",
                             payload={"virtual_index": index})
        agent = threading.Thread(target=self._agent_loop, daemon=True)
        agent.start()
        pool = ThreadPoolExecutor(max_workers=self.live.workers)
        menu_lane = ThreadPoolExecutor(max_workers=1)
        seqs = itertools.count()
        reader = SubtitleReader.create() if self.live.subtitles_ocr else None
        self.events.emit("shadow.subtitles", source="live",
                         payload={"ocr": reader is not None})
        try:
            frames = self._until_stopped(self.source.frames(1.0 / self.live.frames_per_s))
            if reader is not None:
                frames = self._read_subtitles(frames, reader, seqs)
            keyframes = select_keyframes(
                frames, self.settings, subtitle_trigger=reader is None,
                menu_repeat_s=lambda: 1.0 if self._menu_unread() else MENU_REPEAT_S,
            )
            pending: list[tuple] = []  # frames waiting to go to the observer together
            last_menu_kf = -1e9

            def flush() -> None:
                if pending:
                    pool.submit(self._observe_batch, list(pending))
                    pending.clear()

            for kf in keyframes:
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
                frame = (seq, kf.t, name, jpeg, ctx, earlier, time.monotonic(), in_burst)
                # Urgent unless a menu was actually READ moments ago (false alarms from the
                # local detector, like neon lights, must not make a real menu wait).
                if kf.reason == "menu" and (
                    kf.t - self._last_menu_t > MENU_URGENT_GAP_S or self._menu_unread()
                ):
                    # A menu that just appeared (or is up but not read yet) can't wait: send
                    # what is pending (the lines just before it), then the menu on its own
                    # lane. Re-reads of a menu already read go in the normal batches.
                    if not self._menu_unread():
                        self._urgent_first_t = kf.t  # a new attempt to read a menu
                    last_menu_kf = kf.t
                    self._urgent_menu_t = kf.t
                    flush()
                    menu_lane.submit(self._observe, *frame)
                    continue
                if kf.reason == "menu":
                    last_menu_kf = kf.t
                pending.append(frame)
                if (
                    len(pending) >= self.live.batch_frames
                    or kf.t - pending[0][1] >= self.live.batch_wait_s
                ):
                    flush()
            flush()
        except SourceLost as e:
            self.events.emit("shadow.source_lost", source="live", payload={"reason": str(e)})
            self.say("  -- the game is no longer visible: ending and saving memory --")
        finally:
            self.source.close()
            if reader is not None:
                reader.close()
            pool.shutdown(wait=True)
            menu_lane.shutdown(wait=True)
            if self.watch is not None:
                self.watch.stop()
            if self.pad is not None:
                self.pad.release_all()
                self.pad.close()
            self._agent_queue.put((ITEM + 1, next(self._order), ("end", None)))
            agent.join()
            self.events.emit("shadow.finished", source="live", payload=self.report.__dict__)
        return self.report

    def _menu_unread(self) -> bool:
        """A menu was sent urgently but no observation has read a menu since: keep trying,
        for a few seconds at most (the local detector can mistake neon for button letters)."""
        return (
            self._urgent_menu_t > self._last_menu_t
            and self._urgent_menu_t - self._urgent_first_t < UNREAD_MENU_RETRY_S
        )

    def _until_stopped(self, frames):
        for item in frames:
            if self.stop.is_set():
                return
            yield item

    def _read_subtitles(self, frames, reader: SubtitleReader, seqs):
        """Read every frame's subtitle locally (free) and pass frames on unchanged.

        New lines enter the same ordered stream as the observer's results, so the
        transcript keeps the real order of what was said and seen."""
        tracker = SubtitleTracker()
        for t, frame in frames:
            if self._on_hidden_screen:  # pause menu, flowchart...: their texts are not speech
                tracker.update("")
                yield t, frame
                continue
            for text in tracker.update(reader.read(frame)):
                line = SubtitleLine(text=text, speaker="", speaker_basis="unknown")
                obs = Observation(t=t, subtitles=[line])  # frame "" = from the OCR
                self.events.emit("perception.subtitle", source="perception",
                                 payload={"t": t, "text": text})
                with self._lock:
                    self._results[next(seqs)] = obs
                    self._feed_in_order()
            yield t, frame

    # -- perception (worker threads) ----------------------------------------------

    def _observe(self, seq, t, name, jpeg, ctx, earlier, captured_at, in_burst=False) -> None:
        obs: Observation | None
        try:
            obs = self.observer.observe(jpeg, t, fmt_time(t), ctx, earlier)
        except Exception as e:  # never leave a hole in the ordered stream
            self.events.emit("perception.failed", "perception", {"t": t, "error": repr(e)})
            obs = None
        self._handle(seq, t, name, jpeg, ctx, earlier, captured_at, in_burst, obs)

    def _observe_batch(self, batch: list[tuple]) -> None:
        """Several frames in one observer call (cheaper); each result handled as usual."""
        ctx = batch[0][4]
        try:
            results: list[Observation | None] = self.observer.observe_batch(
                [(t, fmt_time(t), jpeg) for _, t, _, jpeg, *_ in batch], ctx
            )
        except Exception as e:  # any failure: these frames count as failed, play goes on
            payload = {"t": batch[0][1], "error": repr(e)}
            self.events.emit("perception.failed", "perception", payload)
            results = [None] * len(batch)
        for frame, obs in zip(batch, results, strict=True):
            self._handle(*frame, obs)

    def _handle(self, seq, t, name, jpeg, ctx, earlier, captured_at, in_burst, obs) -> None:
        if obs is not None:
            obs.frame = name
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
            if obs.frame:  # a real observation tells whether a menu/flowchart is up
                self._on_hidden_screen = obs.screen in HIDDEN_SCREENS
            if obs.screen == "flowchart":
                if not self._in_flowchart:
                    self._in_flowchart = True
                    self._agent_queue.put((CHAPTER_END, next(self._order), ("chapter_end", None)))
                continue
            visible, _ = self.redactor.apply(obs)
            if visible is None:
                continue  # pause menu etc.: says nothing about the game's own menu
            if obs.frame and not visible.choice.present:  # OCR lines say nothing on menus
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
            self._suggested.append(_Suggestion(labels, "", {}, visible.t))
        buttons = {o.label.strip(): o.button for o in visible.choice.options}
        # The countdown started when the frame was captured, not when it was read.
        directions = {o.label.strip(): o.direction for o in visible.choice.options}
        payload = (labels, buttons, visible.choice.timed, captured_at, visible.choice.about,
                   directions)
        # Decide right away on a thread of its own: the character's usual queue may be busy
        # (a stance update, a speaker guess) and a countdown is running.
        threading.Thread(target=self._decide_now, args=(payload,), daemon=True).start()

    def _decide_now(self, payload: tuple) -> None:
        for _ in range(40):  # the first scene is being set up (at most ~2 s)
            if self._runner is not None:
                break
            time.sleep(0.05)
        with self._decide_lock:  # one menu at a time
            self._choice_n += 1
            try:
                self._suggest(self._runner, self._choice_n, *payload)
            except Exception as e:  # keep playing even if one decision fails
                self.events.emit("shadow.error", source="live", payload={"error": repr(e)})

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
        # Shadow mode: the operator's picks are not the agent's, so its personality does not
        # learn from them. Autonomous mode: the agent picks, and grows from it.
        runner.operator_controlled = not self.autonomous
        runner.begin()
        self._runner = runner
        return runner

    def _agent_loop(self) -> None:
        runner = self._new_runner()
        since_stance = 0
        pending: TranscriptItem | None = None
        after_pick: list[TranscriptItem] = []
        heard: list[str] = []  # last lines with their speakers, for attributing new ones
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
                item: TranscriptItem = payload
                if item.kind == "choice":
                    if pending is not None:
                        self._resolve_pick(runner, pending, after_pick)
                    pending, after_pick = item, []  # resolved once the next lines arrive
                    continue
                if item.kind == "dialogue":
                    if item.speaker in ("", "unknown"):  # read by the OCR: no speaker yet
                        item.speaker = self._guess_speaker(item, heard)
                    heard = [*heard, f"{item.speaker}: {item.text}"][-6:]
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

    def _guess_speaker(self, item: TranscriptItem, heard: list[str]) -> str:
        """Who says a subtitle line read by the OCR (cheap, text only)."""
        with self._lock:
            scene = self.ctx.previous_description
            people = dict(self.ctx.known_people)
        listing = "\n".join(f"- {n}: {w}" for n, w in people.items()) or "- (none)"
        recent = "\n".join(heard) or "(none)"
        request = LLMRequest(
            purpose="shadow.speaker",
            system=SPEAKER_SYSTEM.format(protagonist=self.character.capitalize()),
            messages=[
                Message.user(
                    f"known people:\n{listing}\n\nwhat the camera shows: {scene or '(unknown)'}"
                    f"\n\nrecent lines:\n{recent}\n\nnew line (a leading '- ' means a new "
                    f'speaker turn): "{item.raw or item.text}"'
                )
            ],
            json_schema=SPEAKER_SCHEMA,
            max_tokens=200,
        )
        try:
            answer = str((self.session.llm("observer").complete(request).parsed or {})
                         .get("speaker", "")).strip()
        except LLMError:
            answer = ""
        if answer:
            item.speaker_basis = "conversation"
        return answer or "unknown"

    def _suggest(
        self, runner, n, labels, buttons, timed, seen_at, about="", directions=None
    ) -> None:
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
            match = next((s for s in reversed(self._suggested) if s.labels == labels), None)
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
        if self.autonomous and match is not None:
            self._press(match, label, button, (directions or {}).get(label, ""), timed, seen_at)

    def _press(self, suggestion, label, button, direction, timed, seen_at) -> None:
        """Autonomous mode: carry out the decision on the virtual controller, unless the
        operator is in control (paused, or using their own controller since the menu)."""
        def skip(reason: str) -> None:
            self.events.emit("autonomy.skipped", source="live",
                             payload={"label": label, "reason": reason})
            self.say(f"  !! not pressing {label}: {reason}")

        if self.paused:
            return skip("paused by the operator")
        if self.watch is not None and self.watch.active_within(
            time.monotonic() - seen_at + self.live.operator_grace_s
        ):
            return skip("the operator is using their controller")
        if button not in BUTTONS and not (button == "stick" and direction in DIRECTIONS):
            return skip(f"unknown control ({button} {direction})".strip())
        if not timed:
            time.sleep(self.live.press_delay_s)
            if self.paused or (self.watch is not None and self.watch.active_within(
                    self.live.press_delay_s)):
                return skip("the operator took over")
        assert self.pad is not None
        if button in BUTTONS:
            self.pad.press(button)
        else:
            self.pad.tilt(direction)
        suggestion.pressed = label
        self.report.pressed += 1
        self.events.emit(
            "autonomy.pressed",
            source="live",
            payload={"label": label, "button": button, "direction": direction,
                     "seconds_after_menu": round(time.monotonic() - seen_at, 1)},
        )
        self.say(f"  >> pressed {button if button in BUTTONS else 'stick ' + direction}: {label}")

    def _resolve_pick(self, runner, choice_item: TranscriptItem, after: list) -> None:
        """Work out what the operator picked, and how sure we are.

        Sure: the protagonist's next line repeats words of one option, or the highlighted
        option and a text-only guess agree. Unsure: only one of those, or they disagree.
        A guess is only tried if the protagonist says something afterwards (silent
        actions, like picking up an object, give it nothing to go on)."""
        labels = [o.label for o in choice_item.options]
        suggestion = self._suggestion_at(choice_item.t, labels)
        suggested = suggestion.suggested if suggestion else ""
        highlighted = choice_item.chosen
        probe = choice_item.model_copy(update={"chosen": ""})
        infer_picks([probe, *after])
        picked, basis, sure = probe.chosen, "words", bool(probe.chosen)
        if suggestion is not None and suggestion.pressed:
            # The agent pressed it itself: no guessing.
            picked, basis, sure = suggestion.pressed, "pressed", True
        elif not picked:
            protagonist = self.character.lower()
            spoke = [i for i in after if protagonist in i.speaker.lower()]
            guess = guess_pick(self.session.llm("observer"), choice_item, after) if spoke else ""
            if highlighted and guess:
                agree = similar(highlighted, guess) >= 0.85
                picked, basis, sure = highlighted, "highlight+guess", agree
            elif highlighted or guess:
                picked = highlighted or guess
                basis, sure = ("highlight" if highlighted else "guess"), False
        choice_item.chosen = picked
        if picked:
            runner.record_pick(picked, choice_item.about)
            self.ctx.just_chosen = picked
        same = bool(picked and suggested and similar(picked, suggested) >= 0.85)
        comparison = {
            "t": choice_item.t,
            "options": labels,
            "suggested": suggested,
            "picked": picked,
            "picked_by": basis if picked else "",
            "sure": sure,
            # None = we could not tell for sure what the operator picked.
            "agreed": same if (sure and suggested) else None,
        }
        if basis == "pressed":
            comparison["agreed"] = None  # its own press: nothing to compare with the operator
        elif comparison["agreed"] is not None:
            self.report.compared += 1
            self.report.agreed += same
        self.report.comparisons.append(comparison)
        self.events.emit("shadow.compare", source="live", payload=comparison)
        verdict = {True: "same", False: "different", None: "not sure"}[comparison["agreed"]]
        self.say(f"  -- you picked {picked or '(not seen)'}; suggested {suggested or '-'} "
                 f"({verdict})")

    def _suggestion_at(self, t: float, labels: list[str]) -> _Suggestion | None:
        """The suggestion made for the menu that was up at time `t` (menus with the same
        options come back, e.g. questions left after one is asked: time tells them apart)."""
        with self._lock:
            candidates = [s for s in self._suggested if s.suggested and
                          self._overlaps(s.labels, labels)]
        if not candidates:
            return None
        return min(candidates, key=lambda s: abs(s.t - t))
