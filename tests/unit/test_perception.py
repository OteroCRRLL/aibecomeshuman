from __future__ import annotations

import cv2
import numpy as np

from aibecomeshuman.env.scenario import load_scenario
from aibecomeshuman.perception.evaluate import Labels, evaluate
from aibecomeshuman.perception.export import export_scenario
from aibecomeshuman.perception.observer import (
    Hud,
    Observation,
    SeenChoice,
    SeenOption,
    SubtitleLine,
)
from aibecomeshuman.perception.redact import Redactor
from aibecomeshuman.perception.transcript import TranscriptBuilder, infer_picks
from aibecomeshuman.perception.video import (
    menu_score,
    new_subtitle,
    parse_time,
    subtitle_mask,
)

# -- local frame signals -----------------------------------------------------------


def _frame() -> np.ndarray:
    return np.full((1080, 1920, 3), 40, np.uint8)


def _letter(frame: np.ndarray, letter: str, bgr: tuple[int, int, int], x: int, y: int) -> None:
    cv2.putText(frame, letter, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, bgr, 4)


def test_menu_score_finds_button_letters_in_a_column() -> None:
    frame = _frame()
    assert menu_score(frame) == 0
    _letter(frame, "Y", (0, 220, 255), 1200, 400)  # yellow
    _letter(frame, "X", (255, 140, 0), 1200, 460)  # blue
    _letter(frame, "B", (40, 40, 255), 1200, 520)  # red
    _letter(frame, "A", (40, 230, 120), 1200, 580)  # green
    assert menu_score(frame) >= 3


def test_subtitle_change_is_relative_to_the_text() -> None:
    empty = _frame()
    short = _frame()
    # A short line at real subtitle size: white letters with a dark outline.
    cv2.putText(short, "Joder.", (860, 980), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 10)
    cv2.putText(short, "Joder.", (860, 980), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)
    assert not new_subtitle(subtitle_mask(empty), subtitle_mask(empty))
    assert new_subtitle(subtitle_mask(short), subtitle_mask(empty))
    assert not new_subtitle(subtitle_mask(short), subtitle_mask(short))


def test_parse_time() -> None:
    assert parse_time("08:16") == 496
    assert parse_time("1:00:05") == 3605
    assert parse_time(None) is None


# -- redaction -----------------------------------------------------------------------


def test_redactor_hides_menus_patterns_and_tutorials() -> None:
    redactor = Redactor(["inestabilidad del software"])
    assert redactor.apply(Observation(screen="flowchart"))[0] is None
    obs = Observation(
        hud=Hud(analysis=["INESTABILIDAD DEL SOFTWARE", "INFORMACIÓN ENCONTRADA"]),
        tutorial_hint="Has desbloqueado un diálogo",
    )
    visible, hidden = redactor.apply(obs)
    assert visible is not None
    assert visible.hud.analysis == ["INFORMACIÓN ENCONTRADA"]
    assert visible.tutorial_hint == ""
    assert "analysis" in hidden and "tutorial_hint" in hidden
    assert obs.hud.analysis[0] == "INESTABILIDAD DEL SOFTWARE"  # original untouched


# -- transcript ----------------------------------------------------------------------


def _menu(t: float, labels: list[str], highlighted: str = "") -> Observation:
    buttons = ["Y", "X", "B", "A"]
    return Observation(
        t=t,
        choice=SeenChoice(
            kind="dialogue",
            options=[
                SeenOption(button=buttons[i], label=label, highlighted=label == highlighted)
                for i, label in enumerate(labels)
            ],
        ),
    )


def _line(t: float, text: str, speaker: str = "", basis: str = "unknown") -> Observation:
    line = SubtitleLine(text=text, speaker=speaker, speaker_basis=basis)
    return Observation(t=t, subtitles=[line])


def test_transcript_dedupes_groups_menus_and_attributes_lines() -> None:
    b = TranscriptBuilder("Connor", sample_s=1.0)
    b.add(_line(1, "- ¿Cómo sabes mi nombre?", "Daniel", "visible_speaking"))
    b.add(_line(2, "- ¿Cómo sabes mi nombre?", "Daniel", "visible_speaking"))  # same caption
    b.add(_menu(3, ["CALMADO", "EMPATIZAR"]))
    b.add(_menu(4, ["CALMADO", "EMPATIZAR"], highlighted="EMPATIZAR"))
    b.add(_line(5, "Yo soy un androide,", "Daniel", "context"))  # after the pick
    b.add(_line(6, "lo mismo que tú.", "Daniel", "context"))  # continues the same speaker
    b.add(_line(8, "- ¡Estás de su lado!", "Daniel", "visible_speaking"))
    items = b.finish()

    lines = [(i.speaker, i.text) for i in items if i.kind == "dialogue"]
    assert lines == [
        ("Daniel", "¿Cómo sabes mi nombre?"),
        ("Connor", "Yo soy un androide,"),
        ("Connor", "lo mismo que tú."),
        ("Daniel", "¡Estás de su lado!"),
    ]
    (choice,) = [i for i in items if i.kind == "choice"]
    assert [o.label for o in choice.options] == ["CALMADO", "EMPATIZAR"]
    assert choice.chosen == "EMPATIZAR" and choice.chosen_basis == "highlight"


def test_objective_lists_are_not_choices_and_caps_lines_are_display() -> None:
    b = TranscriptBuilder("Connor", sample_s=1.0)
    listing = Observation(
        t=1,
        choice=SeenChoice(
            kind="dialogue",
            options=[SeenOption(button="other", label=x) for x in ("DEDUCE", "EXAMINA")],
        ),
    )
    b.add(listing)
    b.add(_line(2, "LA NIÑA NO OYÓ LOS DISPAROS", "Connor"))
    b.add(Observation(t=3, hud=Hud(analysis=["HERIDA DE B"])))
    b.add(Observation(t=4, hud=Hud(analysis=["HERIDA DE BALA CALIBRE .355"])))
    items = b.finish()
    assert not [i for i in items if i.kind in ("choice", "dialogue")]
    assert [i.text for i in items if i.kind == "hud"] == [
        "LA NIÑA NO OYÓ LOS DISPAROS",
        "HERIDA DE BALA CALIBRE .355",
    ]


def test_picks_are_inferred_from_the_next_line() -> None:
    b = TranscriptBuilder("Connor", sample_s=1.0)
    b.add(_menu(1, ["NOMBRE DEL DIVERGENTE", "CÓDIGO DE DESACTIVACIÓN"]))
    b.add(_line(3, "- ¿Ha probado con su código de desactivación?", "Connor"))
    items = b.finish()
    assert infer_picks(items) == 1
    (choice,) = [i for i in items if i.kind == "choice"]
    assert choice.chosen == "CÓDIGO DE DESACTIVACIÓN" and choice.chosen_basis == "inferred"


# -- evaluation and export -------------------------------------------------------------


def _sample_items():
    b = TranscriptBuilder("Connor", sample_s=1.0)
    b.add(_line(10, "- Hola.", "Connor", "visible_speaking"))
    b.add(_menu(12, ["MENTIRA", "VERDAD"], highlighted="VERDAD"))
    b.add(_line(14, "- Tengo un arma.", "Connor"))
    return b.finish()


def test_evaluate_scores_only_the_processed_stretch() -> None:
    labels = Labels.model_validate(
        {
            "choice": [
                {"t": "00:12", "options": ["MENTIRA", "VERDAD"], "chosen": "VERDAD"},
                {"t": "05:00", "options": ["A", "B"]},  # outside the stretch
            ],
            "line": [{"t": "00:14", "text": "Tengo un arma.", "speaker": "Connor"}],
            "forbidden": ["connor ha muerto"],
        }
    )
    ev = evaluate(_sample_items(), labels, 0, 60)
    assert (ev.choices_found, ev.choices_total) == (1, 1)
    assert (ev.chosen_correct, ev.lines_found, ev.speakers_correct) == (1, 1, 1)
    assert ev.leaks == []


def test_export_writes_a_playable_scenario(tmp_path) -> None:
    path = tmp_path / "scenario.toml"
    scenario = export_scenario(_sample_items(), path, "rec-test", "Test", "connor")
    assert load_scenario(path).id == "rec-test"
    kinds = [s.kind for s in scenario.steps]
    assert kinds == ["dialogue", "choice", "dialogue"]
    assert "operator picked: VERDAD" in path.read_text(encoding="utf-8")
