from __future__ import annotations

from aibecomeshuman.perception.escalate import Escalator
from aibecomeshuman.perception.observer import (
    Observation,
    SeenChoice,
    SeenOption,
    SubtitleLine,
)


def _escalator(spent: list[float], cap: float = 0.15) -> Escalator:
    return Escalator(strong_observer=None, cap_per_chapter_usd=cap, spent=lambda: spent[0])


def test_bursts_of_big_changes() -> None:
    esc = _escalator([0.0])
    assert not esc.note_keyframe(10.0, "change")  # one change is not a burst
    assert esc.note_keyframe(11.5, "change")
    assert not esc.note_keyframe(12.0, "subtitles")
    assert not esc.note_keyframe(30.0, "change")  # long after: a new, single change


def test_only_dramatic_silent_moments_under_the_cap() -> None:
    spent = [0.0]
    esc = _escalator(spent)
    action = Observation(screen="cutscene", description="A man is shot and falls off the ledge")
    assert esc.wanted(action, in_burst=True)
    assert not esc.wanted(action, in_burst=False)
    calm = Observation(screen="cutscene", description="Connor adjusts his tie in the lift")
    assert not esc.wanted(calm, in_burst=True)
    talking = action.model_copy(update={"subtitles": [SubtitleLine(text="¡No!")]})
    assert not esc.wanted(talking, in_burst=True)
    menu = action.model_copy(
        update={"choice": SeenChoice(kind="dialogue", options=[SeenOption(label="A")])}
    )
    assert not esc.wanted(menu, in_burst=True)
    paused = action.model_copy(update={"screen": "pause_menu"})
    assert not esc.wanted(paused, in_burst=True)

    spent[0] = 0.16  # this chapter's cap is used up...
    assert not esc.wanted(action, in_burst=True)
    esc.new_chapter()  # ...until the next chapter starts
    assert esc.wanted(action, in_burst=True)
