"""Offline perception: a recording in, a transcript the agent could play from out.

    recording -> keyframes (local, free) -> observer (vision model) -> redaction
              -> transcript (+ draft scenario, + evaluation against labels if present)

Outputs, inside the run folder:
    frames/tMMSS_x.jpg     the keyframes shown to the observer
    observations.jsonl     every observation, with what was hidden from the agent
    transcript.md / .json  what the agent would know, in order
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from aibecomeshuman.config import ConfigError
from aibecomeshuman.llm.types import BudgetExceeded, LLMError
from aibecomeshuman.perception.evaluate import Evaluation, evaluate, labels_path, load_labels
from aibecomeshuman.perception.export import export_scenario
from aibecomeshuman.perception.observer import Observation, Observer, ObserverContext
from aibecomeshuman.perception.redact import Redactor
from aibecomeshuman.perception.speakers import resolve_speakers
from aibecomeshuman.perception.transcript import (
    TranscriptBuilder,
    TranscriptItem,
    infer_picks,
    render_markdown,
)
from aibecomeshuman.perception.video import (
    encode_jpeg,
    fmt_time,
    sample_frames,
    select_keyframes,
    shrink_jpeg,
    video_info,
)
from aibecomeshuman.prompts import PromptLibrary
from aibecomeshuman.run.session import RunSession

Echo = Callable[[str], None]

# Rough per-frame token use, for the up-front estimate (image + prompt, and the answer).
TEXT_TOKENS_PER_FRAME = 1100
# Previous frames shown (small) with each frame, so fast action reads as movement.
CONTEXT_FRAMES = 2
CONTEXT_FRAME_WIDTH = 640
MOTION_WINDOW_S = 6.0
CONTEXT_TOKENS_PER_FRAME = CONTEXT_FRAMES * 640 * 360 / 750
OUTPUT_TOKENS_PER_FRAME = 350


@dataclass
class StoredKeyframe:
    t: float
    file: str
    reason: str
    width: int
    height: int


@dataclass
class PerceptionResult:
    keyframes: int
    observed: int = 0
    failed: int = 0
    hidden_frames: int = 0
    speakers_changed: int = 0
    items: list[TranscriptItem] = field(default_factory=list)
    estimate_usd: float = 0.0


def estimate_cost(session: RunSession, frames: list[StoredKeyframe]) -> float:
    model = session.project.models[session.profile.roles["observer"]]
    tokens_in = sum(
        f.width * f.height / 750 + TEXT_TOKENS_PER_FRAME + CONTEXT_TOKENS_PER_FRAME
        for f in frames
    )
    tokens_out = OUTPUT_TOKENS_PER_FRAME * len(frames)
    return (
        tokens_in * model.price_input_per_mtok + tokens_out * model.price_output_per_mtok
    ) / 1e6


def extract_keyframes(
    session: RunSession, video: Path, start_s: float, end_s: float | None
) -> list[StoredKeyframe]:
    assert session.dir is not None
    settings = session.project.app.perception
    frames_dir = session.dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    stored = []
    frames = sample_frames(video, settings.sample_every_s, start_s, end_s)
    for kf in select_keyframes(frames, settings):
        name = f"t{int(kf.t):05d}_{kf.reason}.jpg"
        (frames_dir / name).write_bytes(encode_jpeg(kf.image))
        h, w = kf.image.shape[:2]
        stored.append(StoredKeyframe(t=kf.t, file=name, reason=kf.reason, width=w, height=h))
    return stored


def perceive(
    session: RunSession,
    video: Path,
    character: str,
    start_s: float = 0.0,
    end_s: float | None = None,
    dry_run: bool = False,
    echo: Echo | None = None,
) -> PerceptionResult:
    say = echo or (lambda _: None)
    assert session.events is not None and session.dir is not None
    events = session.events
    if not session.has_role("observer"):
        raise ConfigError(f"run '{session.profile.name}' assigns no model to the 'observer' role")
    settings = session.project.app.perception
    info = video_info(video)

    say(f"Reading {video.name} ({fmt_time(info.duration_s)}, {info.width}x{info.height})...")
    frames = extract_keyframes(session, video, start_s, end_s)
    result = PerceptionResult(keyframes=len(frames), estimate_usd=estimate_cost(session, frames))
    events.emit(
        "perception.started",
        source="perception",
        payload={
            "video": video.name,
            "duration_s": round(info.duration_s, 1),
            "from_s": start_s,
            "to_s": end_s,
            "keyframes": len(frames),
            "estimate_usd": round(result.estimate_usd, 4),
            "dry_run": dry_run,
            "character": character,
        },
    )
    say(f"{len(frames)} keyframes selected (estimated cost ${result.estimate_usd:.3f}).")
    if dry_run or not frames:
        return result

    library = PromptLibrary(session.project.prompts_dir)
    observer = Observer(session.llm("observer"), library.role("observer"))
    redactor = Redactor(settings.hidden_patterns)
    known_people: dict[str, str] = {}
    lock = threading.Lock()
    stop = threading.Event()
    observations: list[tuple[Observation, Observation | None]] = []
    progress = {"done": 0}

    def work(chunk: list[StoredKeyframe]) -> None:
        ctx = ObserverContext(
            protagonist=character.capitalize(), appearance=library.appearance(character)
        )
        last_chosen = ""
        recent: list[tuple[float, bytes]] = []  # small copies of this worker's last frames
        for kf in chunk:
            if stop.is_set():
                return
            with lock:
                ctx.known_people = dict(known_people)
            jpeg = (session.dir / "frames" / kf.file).read_bytes()  # type: ignore[operator]
            earlier = [
                (fmt_time(t), small) for t, small in recent if kf.t - t <= MOTION_WINDOW_S
            ]
            recent = [*recent, (kf.t, shrink_jpeg(jpeg, CONTEXT_FRAME_WIDTH))][-CONTEXT_FRAMES:]
            try:
                obs = observer.observe(jpeg, kf.t, fmt_time(kf.t), ctx, earlier)
            except BudgetExceeded as e:
                stop.set()
                events.emit("perception.failed", "perception", {"t": kf.t, "error": str(e)})
                return
            except LLMError as e:
                with lock:
                    result.failed += 1
                events.emit("perception.failed", "perception", {"t": kf.t, "error": str(e)})
                continue
            obs.frame = kf.file
            visible, hidden = redactor.apply(obs)
            events.emit(
                "perception.observed",
                source="perception",
                payload={
                    "t": kf.t,
                    "frame": kf.file,
                    "reason": kf.reason,
                    "observation": obs.model_dump(mode="json", exclude={"t", "frame"}),
                    "visible": visible is not None,
                    "hidden": hidden,
                },
            )
            with lock:
                observations.append((obs, visible))
                result.observed += 1
                progress["done"] += 1
                done = progress["done"]
                if visible is not None:
                    for name in visible.names_seen:
                        if name.name.strip():
                            known_people.setdefault(name.name.strip(), name.who.strip())
            if done % 25 == 0:
                spent = session.manifest.cost_usd if session.manifest else 0.0
                say(f"  observed {done}/{len(frames)} frames (${spent:.3f} so far)")

            # Context for this worker's next frame.
            ctx.just_chosen = ""
            if visible is None:
                continue
            if visible.description:
                ctx.previous_description = visible.description
            for line in visible.subtitles:
                ctx.recent_lines = [*ctx.recent_lines, f"{line.speaker}: {line.text}"][-6:]
            if visible.choice.present:
                last_chosen = next(
                    (o.label for o in visible.choice.options if o.highlighted), last_chosen
                )
            elif last_chosen:
                ctx.just_chosen, last_chosen = last_chosen, ""

    workers = min(settings.workers, len(frames))
    size = -(-len(frames) // workers)
    chunks = [frames[i : i + size] for i in range(0, len(frames), size)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, chunks))

    observations.sort(key=lambda pair: pair[0].t)
    with (session.dir / "observations.jsonl").open("w", encoding="utf-8") as f:
        for obs, _ in observations:
            f.write(obs.model_dump_json() + "\n")
    build_transcript(session, result, observations, video, character, known_people)
    return result


def rebuild(
    session: RunSession, source_run: Path, video: Path, character: str
) -> PerceptionResult:
    """Build the transcript again from a previous run's observations (no frames re-read):
    cheap, for tuning everything that happens after the observer."""
    source = source_run / "observations.jsonl"
    if not source.is_file():
        raise ConfigError(f"no observations in {source_run}")
    redactor = Redactor(session.project.app.perception.hidden_patterns)
    observations = []
    known_people: dict[str, str] = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obs = Observation.model_validate_json(line)
        visible, _ = redactor.apply(obs)
        observations.append((obs, visible))
        for name in visible.names_seen if visible else []:
            if name.name.strip():
                known_people.setdefault(name.name.strip(), name.who.strip())
    result = PerceptionResult(keyframes=len(observations), observed=len(observations))
    build_transcript(session, result, observations, video, character, known_people)
    return result


def build_transcript(
    session: RunSession,
    result: PerceptionResult,
    observations: list[tuple[Observation, Observation | None]],
    video: Path,
    character: str,
    known_people: dict[str, str],
) -> None:
    assert session.dir is not None and session.events is not None
    settings = session.project.app.perception
    builder = TranscriptBuilder(character.capitalize(), settings.sample_every_s)
    for _, visible in observations:
        if visible is None:
            result.hidden_frames += 1
        else:
            builder.add(visible)
    result.items = builder.finish()
    infer_picks(result.items)
    result.speakers_changed = resolve_speakers(
        session.llm("observer"), result.items, character.capitalize(), known_people
    )
    write_transcript(session.dir, result.items, f"What the agent saw: {video.name}")
    session.events.emit(
        "perception.transcript",
        source="perception",
        payload={
            "items": len(result.items),
            "choices": sum(1 for i in result.items if i.kind == "choice"),
            "lines": sum(1 for i in result.items if i.kind == "dialogue"),
            "hidden_frames": result.hidden_frames,
            "failed": result.failed,
            "speakers_changed": result.speakers_changed,
        },
    )


def finish_outputs(
    session: RunSession,
    video: Path,
    result: PerceptionResult,
    character: str,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> Evaluation | None:
    """Draft scenario + evaluation against labels (when the recording has a labels file)."""
    assert session.dir is not None and session.events is not None
    if result.items:
        try:
            export_scenario(
                result.items,
                session.dir / "scenario.toml",
                scenario_id=f"rec-{session.dir.name}",
                title=f"Recording {video.stem}",
                character=character,
            )
        except ConfigError as e:
            session.events.emit("perception.export_failed", "perception", {"error": str(e)})
    labels_file = labels_path(video)
    if not labels_file.is_file():
        return None
    evaluation = evaluate(result.items, load_labels(labels_file), start_s, end_s)
    (session.dir / "evaluation.json").write_text(
        evaluation.model_dump_json(indent=1), encoding="utf-8"
    )
    session.events.emit("perception.evaluated", "perception", evaluation.model_dump())
    return evaluation


def write_transcript(run_dir: Path, items: list[TranscriptItem], title: str) -> None:
    (run_dir / "transcript.json").write_text(
        json.dumps([i.model_dump(mode="json") for i in items], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    (run_dir / "transcript.md").write_text(render_markdown(items, title), encoding="utf-8")


def read_transcript(run_dir: Path) -> list[TranscriptItem]:
    data = json.loads((run_dir / "transcript.json").read_text(encoding="utf-8"))
    return [TranscriptItem.model_validate(row) for row in data]
