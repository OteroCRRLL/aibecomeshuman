from __future__ import annotations

import cv2
import numpy as np
import pytest

from aibecomeshuman.perception.ocr import SubtitleReader, SubtitleTracker, clean, split_speakers


def test_clean_fixes_typical_ocr_mistakes() -> None:
    assert clean("iAtrás!") == "¡Atrás!"
    assert clean("- iNo! iNo, por tavor! i Te lo ruego!") == "- ¡No! ¡No, por tavor! ¡Te lo ruego!"
    assert clean("10 mismo que tú.") == "lo mismo que tú."
    assert clean("Si no, ya 10 habrías hecho.") == "Si no, ya lo habrías hecho."
    assert clean("Hora estimada: 20:10") == "Hora estimada: 20:10"
    assert clean("Tengo 10 balas") == "Tengo lo balas"  # rare; accepted trade-off


def test_split_speakers() -> None:
    assert split_speakers("- Hola. - ¿Qué quieres?") == ["- Hola.", "- ¿Qué quieres?"]
    assert split_speakers("Bien-hecho, sí.") == ["Bien-hecho, sí."]


def test_tracker_reports_each_caption_once_when_stable() -> None:
    tr = SubtitleTracker()
    assert tr.update("¿Estás arma") == []  # first sight: wait for it to settle
    assert tr.update("¿Estás armado?") == ["¿Estás armado?"]
    assert tr.update("¿Estás armado?") == []  # still the same caption
    assert tr.update("") == []
    assert tr.update("- Por favor, ayúdame...") == []
    assert tr.update("- Por favor, ayúdame...") == ["- Por favor, ayúdame..."]
    # A caption that grows a second speaker: only the new part, right away (the rest of it
    # was already stable).
    both = "Aquí no va a morir nadie. - Por favor, ayúdame..."
    assert tr.update(both) == ["Aquí no va a morir nadie."]
    assert tr.update(both) == []


def test_windows_ocr_reads_a_subtitle() -> None:
    reader = SubtitleReader.create()
    if reader is None:
        pytest.skip("Windows OCR with Spanish is not available here")
    frame = np.full((1080, 1920, 3), 30, np.uint8)
    for colour, width in (((0, 0, 0), 10), ((255, 255, 255), 3)):
        cv2.putText(frame, "Tengo un arma.", (720, 980), cv2.FONT_HERSHEY_SIMPLEX, 1.6, colour,
                    width)
    assert "Tengo un arma" in reader.read(frame)
    reader.close()
