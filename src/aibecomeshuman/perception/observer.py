"""The observer: one frame in, one structured observation out (cheap vision model)."""

from __future__ import annotations

import base64
import re
from typing import Literal

from pydantic import BaseModel, Field

from aibecomeshuman.llm.base import LLMProvider
from aibecomeshuman.llm.types import ImagePart, LLMRequest, Message, TextPart

Screen = Literal[
    "gameplay", "cutscene", "main_menu", "pause_menu", "flowchart", "loading", "other_menu"
]
SpeakerBasis = Literal["visible_speaking", "after_choice", "context", "unknown"]
Button = Literal["Y", "X", "B", "A", "stick", "other"]
ChoiceKind = Literal["none", "dialogue", "action", "stick"]

_STR = {"type": "string"}
_STR_LIST = {"type": "array", "items": _STR}

OBSERVATION_SCHEMA = {
    "type": "object",
    "properties": {
        "screen": {"type": "string", "enum": list(Screen.__args__)},
        "description": {
            "type": "string",
            "description": "English, <= 40 words. Empty if nothing changed since the "
            "previous description.",
        },
        "subtitles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": _STR,
                    "speaker": _STR,
                    "speaker_basis": {"type": "string", "enum": list(SpeakerBasis.__args__)},
                },
                "required": ["text", "speaker", "speaker_basis"],
                "additionalProperties": False,
            },
        },
        "choice": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": list(ChoiceKind.__args__)},
                "about": {
                    "type": "string",
                    "description": "what or whom the options refer to, e.g. 'the fish on "
                    "the floor' or 'answering the man holding the girl'; empty if no choice",
                },
                "timed": {"type": "boolean"},
                "options": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "button": {"type": "string", "enum": list(Button.__args__)},
                            "label": _STR,
                            "locked": {"type": "boolean"},
                            "highlighted": {"type": "boolean"},
                        },
                        "required": ["button", "label", "locked", "highlighted"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["kind", "about", "timed", "options"],
            "additionalProperties": False,
        },
        "hud": {
            "type": "object",
            "properties": {
                "objectives": _STR_LIST,
                "success_probability": {
                    "type": "string",
                    "description": "e.g. '57%'; empty if not shown",
                },
                "analysis": _STR_LIST,
            },
            "required": ["objectives", "success_probability", "analysis"],
            "additionalProperties": False,
        },
        "tutorial_hint": _STR,
        "names_seen": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": _STR, "who": _STR},
                "required": ["name", "who"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "screen",
        "description",
        "subtitles",
        "choice",
        "hud",
        "tutorial_hint",
        "names_seen",
    ],
    "additionalProperties": False,
}


class SubtitleLine(BaseModel):
    text: str
    speaker: str = ""
    speaker_basis: SpeakerBasis = "unknown"


class SeenOption(BaseModel):
    button: Button = "other"
    label: str
    locked: bool = False
    highlighted: bool = False


class SeenChoice(BaseModel):
    kind: ChoiceKind = "none"
    about: str = ""  # what the options refer to (the fish, the man...)
    timed: bool = False
    options: list[SeenOption] = Field(default_factory=list)

    @property
    def present(self) -> bool:
        # Real choices have button prompts; a plain list of texts is something else.
        real = [o for o in self.options if o.label.strip() and o.button != "other"]
        return self.kind != "none" and len(real) >= 2


class Hud(BaseModel):
    objectives: list[str] = Field(default_factory=list)
    success_probability: str = ""
    analysis: list[str] = Field(default_factory=list)


class NameSeen(BaseModel):
    name: str
    who: str = ""


class Observation(BaseModel):
    t: float = 0.0
    frame: str = ""  # file name of the keyframe inside the run folder
    screen: Screen = "gameplay"
    description: str = ""
    subtitles: list[SubtitleLine] = Field(default_factory=list)
    choice: SeenChoice = Field(default_factory=SeenChoice)
    hud: Hud = Field(default_factory=Hud)
    tutorial_hint: str = ""
    names_seen: list[NameSeen] = Field(default_factory=list)


class ObserverContext(BaseModel):
    """What the observer is told about the moments before this frame."""

    protagonist: str
    # How the protagonist looks, so it is not confused with others on screen.
    appearance: str = ""
    known_people: dict[str, str] = Field(default_factory=dict)  # name -> who
    previous_description: str = ""
    recent_lines: list[str] = Field(default_factory=list)  # "speaker: text"
    just_chosen: str = ""  # label of an option picked right before this frame

    def render(self, time_label: str) -> str:
        people = "\n".join(f"- {n}: {w}" for n, w in self.known_people.items()) or "- (none yet)"
        lines = "\n".join(f"- {line}" for line in self.recent_lines) or "- (none)"
        parts = [
            "<context>",
            f"time: {time_label}",
            f"protagonist (the character the player controls): {self.protagonist}",
            f"the protagonist looks like: {self.appearance or '(unknown)'}. Anyone who does "
            "not look like this is someone else, even in a close-up.",
            f"known people:\n{people}",
            f"previous description: {self.previous_description or '(none)'}",
            f"recent subtitle lines:\n{lines}",
        ]
        if self.just_chosen:
            parts.append(f"the protagonist just picked the option: {self.just_chosen}")
        parts.append("</context>")
        return "\n".join(parts)


def _image(jpeg: bytes) -> ImagePart:
    return ImagePart(media_type="image/jpeg", data_b64=base64.b64encode(jpeg).decode())


class Observer:
    def __init__(self, provider: LLMProvider, system: str) -> None:
        self.provider = provider
        self.system = system

    def observe(
        self,
        jpeg: bytes,
        t: float,
        time_label: str,
        ctx: ObserverContext,
        earlier: list[tuple[str, bytes]] | None = None,
    ) -> Observation:
        """`earlier`: a few previous frames (time label, small JPEG), oldest first, so fast
        action can be understood as movement instead of a single still."""
        content: list[ImagePart | TextPart] = []
        for label, image in earlier or []:
            content += [TextPart(text=f"earlier frame ({label}), for context only:"), _image(image)]
        content += [
            TextPart(text=f"CURRENT frame ({time_label}): report this one."),
            _image(jpeg),
            TextPart(text=ctx.render(time_label)),
        ]
        request = LLMRequest(
            purpose="observe.frame",
            system=self.system,
            messages=[Message(role="user", content=content)],
            json_schema=OBSERVATION_SCHEMA,
            max_tokens=1500,
        )
        response = self.provider.complete(request)
        observation = Observation.model_validate(response.parsed or {})
        observation.t = t
        for option in observation.choice.options:
            # "[EMMA Y TÚ]" is an option unlocked by a clue; only "[...]" is really locked.
            option.label = option.label.strip().strip("[]").strip()
            option.locked = not re.search(r"\w", option.label)
        return observation
