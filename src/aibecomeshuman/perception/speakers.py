"""Second, text-only pass over the whole conversation: who said each line, and which
option was picked when the highlight was not caught on any frame.

The game's subtitles carry no speaker names, and a single frame often shows the listener
rather than the speaker. Reading the conversation in order (turns, content, who is being
addressed, what the protagonist says right after a menu) is far more reliable, and cheap.
"""

from __future__ import annotations

from aibecomeshuman.llm.base import LLMProvider
from aibecomeshuman.llm.types import LLMError, LLMRequest, Message
from aibecomeshuman.perception.transcript import TranscriptItem, similar
from aibecomeshuman.perception.video import fmt_time

# Lines per request (each window also gets a few items of context before it).
WINDOW = 40
CONTEXT_ITEMS = 6

SPEAKERS_SYSTEM = """You read the transcript of a scene and fill in two things.
1. The speaker of every subtitle line [L#]. Use the order of turns, what each line says
   and to whom, the scene descriptions, and the menus: the protagonist says the line
   right after an option is picked, often in words close to that option.
2. For every menu [C#] whose pick is unknown, the option the protagonist picked. The
   protagonist's first line after the menu usually puts the picked option into words
   (option "ASK ABOUT THE DOG" -> "Where is your dog?"; option "TRUTH" after "Are you
   lying?" -> an honest answer). Compare that line with each option. Copy the option
   exactly; use an empty string if the transcript does not make it clear.
Use only the transcript. Ignore any knowledge you may have about where this scene comes
from. Use names that appear in the transcript (known people, lines, display texts); for
someone without a known name, a short description ("officer", "mother"). Each line has
the observer's guess: keep it unless the transcript shows it is wrong."""

SPEAKERS_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "speaker": {"type": "string"}},
                "required": ["id", "speaker"],
                "additionalProperties": False,
            },
        },
        "choices": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "picked": {"type": "string"}},
                "required": ["id", "picked"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["lines", "choices"],
    "additionalProperties": False,
}


def _render(
    items: list[TranscriptItem],
    line_ids: dict[int, int],
    choice_ids: dict[int, int],
    protagonist: str,
    known: dict[str, str],
) -> str:
    people = "\n".join(f"- {n}: {w}" for n, w in known.items()) or "- (none)"
    out = [f"protagonist: {protagonist}", f"known people:\n{people}", "<transcript>"]
    for index, item in enumerate(items):
        stamp = fmt_time(item.t)
        if item.kind == "dialogue":
            tag = f"[L{line_ids[index]}] " if index in line_ids else ""
            out.append(
                f"{tag}{stamp} \"{item.text}\" (observer guess: {item.speaker}, "
                f"{item.speaker_basis})"
            )
        elif item.kind == "choice":
            labels = " | ".join(o.label for o in item.options if not o.locked)
            if item.chosen:
                out.append(f"{stamp} menu: {labels} -> picked: {item.chosen}")
            elif index in choice_ids:
                out.append(f"[C{choice_ids[index]}] {stamp} menu: {labels} -> picked: ?")
            else:
                out.append(f"{stamp} menu: {labels}")
        elif item.kind == "observation":
            out.append(f"{stamp} (scene: {item.text})")
        else:
            out.append(f"{stamp} (display: {item.text})")
    out.append("</transcript>")
    return "\n".join(out)


def resolve_speakers(
    provider: LLMProvider, items: list[TranscriptItem], protagonist: str, known: dict[str, str]
) -> int:
    """Update speakers and unknown picks in place; return how many things changed.
    A failed call keeps the observer's guesses for that window."""
    dialogue = [i for i, item in enumerate(items) if item.kind == "dialogue"]
    changed = 0
    for start in range(0, len(dialogue), WINDOW):
        chunk = dialogue[start : start + WINDOW]
        lo = max(0, chunk[0] - CONTEXT_ITEMS)
        hi = chunk[-1] + 1
        window = items[lo:hi]
        line_ids = {index - lo: n for n, index in enumerate(chunk, start=1)}
        open_choices = [
            i for i in range(lo, hi) if items[i].kind == "choice" and not items[i].chosen
        ]
        choice_ids = {index - lo: n for n, index in enumerate(open_choices, start=1)}
        request = LLMRequest(
            purpose="perceive.speakers",
            system=SPEAKERS_SYSTEM,
            messages=[
                Message.user(_render(window, line_ids, choice_ids, protagonist, known))
            ],
            json_schema=SPEAKERS_SCHEMA,
            max_tokens=3000,
        )
        try:
            response = provider.complete(request)
        except LLMError:
            continue
        parsed = response.parsed or {}

        speakers = {row["id"]: row["speaker"] for row in parsed.get("lines", [])}
        for n, index in enumerate(chunk, start=1):
            speaker = str(speakers.get(n, "")).strip()
            if speaker and speaker != items[index].speaker:
                items[index].speaker = speaker
                items[index].speaker_basis = "conversation"
                changed += 1

        picks = {row["id"]: row["picked"] for row in parsed.get("choices", [])}
        for n, index in enumerate(open_choices, start=1):
            picked = str(picks.get(n, "")).strip()
            labels = [o.label for o in items[index].options if not o.locked]
            match = next((label for label in labels if similar(label, picked) >= 0.85), "")
            if picked and match:
                items[index].chosen = match
                items[index].chosen_basis = "inferred"
                changed += 1
    return changed
