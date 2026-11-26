"""Turning a stream of (already redacted) observations into a clean transcript.

Consecutive frames show the same subtitle, the same menu and the same HUD text many
times; the transcript keeps each thing once, groups menu frames into one choice, works
out which option was picked (the last highlighted one) and attributes the first line
after a choice to the protagonist.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Literal

from pydantic import BaseModel, Field

from aibecomeshuman.perception.observer import Observation, SeenOption

ItemKind = Literal["observation", "dialogue", "hud", "choice"]

# A subtitle continuing an unfinished line this soon after it is the same speaker.
CONTINUATION_S = 8.0
# Descriptions closer together than this are the same moment seen from another angle.
DESCRIPTION_GAP_S = 6.0


def _is_display_text(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 8 and all(c.isupper() for c in letters) and len(text.split()) >= 3


def _unfinished(text: str) -> bool:
    text = text.rstrip()
    return text.endswith((",", "...", "…")) or not text.endswith((".", "!", "?", "»", '"'))


class ChoiceOption(BaseModel):
    button: str
    label: str
    locked: bool = False


class TranscriptItem(BaseModel):
    t: float
    kind: ItemKind
    text: str = ""
    speaker: str = ""
    speaker_basis: str = ""
    # choices only
    options: list[ChoiceOption] = Field(default_factory=list)
    chosen: str = ""
    chosen_basis: str = ""  # "highlight" (seen on a frame) | "inferred" (from what followed)
    timed: bool = False
    seconds: float = 0.0
    frame: str = ""


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def similar(a: str, b: str) -> float:
    a, b = normalize(a), normalize(b)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


class _OpenChoice:
    def __init__(self, obs: Observation) -> None:
        self.start = obs.t
        self.end = obs.t
        self.frame = obs.frame
        self.timed = obs.choice.timed
        self.options: dict[str, SeenOption] = {}
        self.chosen = ""
        self.update(obs)

    def overlaps(self, options: list[SeenOption]) -> bool:
        return any(
            similar(o.label, known) >= 0.85 for o in options for known in self.options
        )

    def update(self, obs: Observation) -> None:
        self.end = obs.t
        self.timed = self.timed or obs.choice.timed
        for option in obs.choice.options:
            label = option.label.strip()
            if not label:
                continue
            self.options.setdefault(label, option)
            if option.highlighted and not option.locked:
                self.chosen = label

    def close(self, sample_s: float) -> TranscriptItem:
        return TranscriptItem(
            t=self.start,
            kind="choice",
            options=[
                ChoiceOption(button=o.button, label=label, locked=o.locked)
                for label, o in self.options.items()
            ],
            chosen=self.chosen,
            chosen_basis="highlight" if self.chosen else "",
            timed=self.timed,
            seconds=round(self.end - self.start + sample_s, 1),
            frame=self.frame,
        )


class TranscriptBuilder:
    def __init__(self, protagonist: str, sample_s: float) -> None:
        self.protagonist = protagonist
        self.sample_s = sample_s
        self.items: list[TranscriptItem] = []
        self._lines: list[str] = []
        self._hud: list[TranscriptItem] = []
        self._probability = ""
        self._choice: _OpenChoice | None = None
        self._after_choice = False
        self._last_description = -1e9

    def add(self, obs: Observation) -> None:
        if obs.choice.present:
            if self._choice is not None and self._choice.overlaps(obs.choice.options):
                self._choice.update(obs)
            else:
                self._close_choice()
                self._choice = _OpenChoice(obs)
        else:
            self._close_choice()

        if obs.description and obs.t - self._last_description >= DESCRIPTION_GAP_S:
            self._last_description = obs.t
            self.items.append(
                TranscriptItem(t=obs.t, kind="observation", text=obs.description, frame=obs.frame)
            )

        hud_texts = [*obs.hud.objectives, *obs.hud.analysis]
        for line in obs.subtitles:
            raw = line.text.strip()
            text = raw.lstrip("-–— ").strip()
            if _is_display_text(text):
                # All-caps lines in mid-screen are the protagonist's display, not speech.
                hud_texts.append(text)
                continue
            if not text or any(similar(text, seen) >= 0.85 for seen in self._lines[-8:]):
                continue
            new_turn = raw[:1] in "-–—"
            speaker, basis = line.speaker.strip(), line.speaker_basis
            previous = self._last_dialogue()
            if self._after_choice:
                # The first line after picking an option is the protagonist saying it.
                speaker, basis = self.protagonist, "after_choice"
            elif (
                not new_turn
                and previous is not None
                and obs.t - previous.t <= CONTINUATION_S
                and _unfinished(previous.text)
            ):
                # Subtitle convention: no leading dash after an unfinished line = same speaker.
                speaker, basis = previous.speaker, "continuation"
            self._after_choice = False
            self._lines.append(text)
            self.items.append(
                TranscriptItem(
                    t=obs.t,
                    kind="dialogue",
                    text=text,
                    speaker=speaker or "unknown",
                    speaker_basis=basis,
                    frame=obs.frame,
                )
            )

        for text in hud_texts:
            text = " ".join(text.split())
            if not normalize(text):
                continue
            known = next((i for i in self._hud if similar(i.text, text) >= 0.85), None)
            if known is None:
                item = TranscriptItem(t=obs.t, kind="hud", text=text, frame=obs.frame)
                self._hud.append(item)
                self.items.append(item)
            elif len(text) > len(known.text):
                # The same text caught half-drawn earlier: keep the complete version.
                known.text = text
        probability = obs.hud.success_probability.strip()
        if probability and probability != self._probability:
            self._probability = probability
            self.items.append(
                TranscriptItem(
                    t=obs.t,
                    kind="hud",
                    text=f"probability of success: {probability}",
                    frame=obs.frame,
                )
            )

    def _last_dialogue(self) -> TranscriptItem | None:
        return next((i for i in reversed(self.items) if i.kind == "dialogue"), None)

    def _close_choice(self) -> None:
        if self._choice is None:
            return
        item = self._choice.close(self.sample_s)
        self.items.append(item)
        self._after_choice = bool(item.chosen)
        self._choice = None

    def finish(self) -> list[TranscriptItem]:
        self._close_choice()
        self.items.sort(key=lambda i: i.t)
        return self.items


def _stems(text: str) -> set[str]:
    # Word starts are enough to match "comportamiento" with "comportado".
    return {w[:5] for w in normalize(text).split() if len(w) >= 4}


def infer_picks(items: list[TranscriptItem], window_s: float = 8.0) -> int:
    """Pick from what is said right after a menu, when the highlight was not seen.

    If the first lines after the menu share words with exactly one option, that option
    was picked ("CÓDIGO DE DESACTIVACIÓN" -> "¿Ha probado con su código de desactivación?").
    Returns how many picks were filled in.
    """
    filled = 0
    for index, item in enumerate(items):
        if item.kind != "choice" or item.chosen:
            continue
        options = [o for o in item.options if not o.locked]
        after = [
            i
            for i in items[index + 1 :]
            if i.kind == "dialogue" and i.t - item.t <= window_s + item.seconds
        ][:2]
        for line in after:
            words = _stems(line.text)
            hits = [o for o in options if _stems(o.label) & words]
            if len(hits) == 1:
                item.chosen, item.chosen_basis = hits[0].label, "inferred"
                filled += 1
                break
    return filled


def render_markdown(items: list[TranscriptItem], title: str) -> str:
    from aibecomeshuman.perception.video import fmt_time

    out = [f"# {title}", ""]
    for item in items:
        stamp = fmt_time(item.t)
        match item.kind:
            case "observation":
                out.append(f"- `{stamp}` *{item.text}*")
            case "dialogue":
                out.append(f"- `{stamp}` **{item.speaker}**: {item.text}")
            case "hud":
                out.append(f"- `{stamp}` [display] {item.text}")
            case "choice":
                options = " / ".join(
                    f"{o.button} {o.label}{' (locked)' if o.locked else ''}" for o in item.options
                )
                timed = f", timed ~{item.seconds:.0f}s" if item.timed else ""
                how = " (inferred)" if item.chosen_basis == "inferred" else ""
                chosen = f" -> **{item.chosen}**{how}" if item.chosen else " -> (not seen)"
                out.append(f"- `{stamp}` CHOICE{timed}: {options}{chosen}")
    return "\n".join(out) + "\n"
