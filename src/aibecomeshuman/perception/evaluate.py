"""Scoring a transcript against hand-made labels for the same recording.

Labels live next to the recording as `<recording name>.labels.toml`:

    [[choice]]
    t = "08:16"
    options = ["CALMADO", "EMPATIZAR"]
    chosen = "EMPATIZAR"

    [[line]]
    t = "01:44"
    text = "Por favor, salve a mi pequeña..."
    speaker = "mother"            # optional

    forbidden = ["CONNOR HA MUERTO"]   # must never reach the agent
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

from aibecomeshuman.perception.transcript import TranscriptItem, normalize, similar
from aibecomeshuman.perception.video import parse_time

WINDOW_S = 8.0


class LabeledChoice(BaseModel):
    t: str
    options: list[str]
    chosen: str = ""


class LabeledLine(BaseModel):
    t: str
    text: str
    speaker: str = ""


class Labels(BaseModel):
    choice: list[LabeledChoice] = Field(default_factory=list)
    line: list[LabeledLine] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)


class Evaluation(BaseModel):
    choices_total: int = 0
    choices_found: int = 0
    chosen_total: int = 0
    chosen_correct: int = 0
    lines_total: int = 0
    lines_found: int = 0
    speakers_total: int = 0
    speakers_correct: int = 0
    leaks: list[str] = Field(default_factory=list)
    missed_choices: list[str] = Field(default_factory=list)
    missed_lines: list[str] = Field(default_factory=list)

    def summary(self) -> list[str]:
        def pct(a: int, b: int) -> str:
            return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "-"

        return [
            f"choices detected:  {pct(self.choices_found, self.choices_total)}",
            f"picked option:     {pct(self.chosen_correct, self.chosen_total)}",
            f"subtitle lines:    {pct(self.lines_found, self.lines_total)}",
            f"speakers:          {pct(self.speakers_correct, self.speakers_total)}",
            f"leaks (must be 0): {len(self.leaks)}",
        ]


def labels_path(video: Path) -> Path:
    return video.with_name(video.name + ".labels.toml")


def load_labels(path: Path) -> Labels:
    with path.open("rb") as f:
        return Labels.model_validate(tomllib.load(f))


def _near(item: TranscriptItem, t: str) -> bool:
    target = parse_time(t) or 0.0
    return abs(item.t - target) <= WINDOW_S


def _in_range(t: str, start_s: float, end_s: float | None) -> bool:
    seconds = parse_time(t) or 0.0
    return seconds >= start_s and (end_s is None or seconds <= end_s)


def evaluate(
    items: list[TranscriptItem], labels: Labels, start_s: float = 0.0, end_s: float | None = None
) -> Evaluation:
    """Score only the labels inside the stretch of video that was processed."""
    ev = Evaluation()
    choices = [i for i in items if i.kind == "choice"]
    for label in labels.choice:
        if not _in_range(label.t, start_s, end_s):
            continue
        ev.choices_total += 1
        wanted = {normalize(o) for o in label.options}
        match = None
        for item in choices:
            if not _near(item, label.t):
                continue
            got = {normalize(o.label) for o in item.options}
            hits = sum(1 for w in wanted if any(similar(w, g) >= 0.85 for g in got))
            if hits >= max(1, len(wanted) // 2):
                match = item
                break
        if match is None:
            ev.missed_choices.append(f"{label.t} {' / '.join(label.options)}")
            continue
        ev.choices_found += 1
        if label.chosen:
            ev.chosen_total += 1
            if match.chosen and similar(match.chosen, label.chosen) >= 0.85:
                ev.chosen_correct += 1

    lines = [i for i in items if i.kind == "dialogue"]
    for label in labels.line:
        if not _in_range(label.t, start_s, end_s):
            continue
        ev.lines_total += 1
        match = next(
            (i for i in lines if _near(i, label.t) and similar(i.text, label.text) >= 0.8), None
        )
        if match is None:
            ev.missed_lines.append(f"{label.t} {label.text}")
            continue
        ev.lines_found += 1
        if label.speaker:
            ev.speakers_total += 1
            if normalize(label.speaker) in normalize(match.speaker):
                ev.speakers_correct += 1

    everything = " | ".join(
        normalize(" ".join([i.text, i.speaker, *(o.label for o in i.options)])) for i in items
    )
    ev.leaks = [f for f in labels.forbidden if normalize(f) in everything]
    return ev
