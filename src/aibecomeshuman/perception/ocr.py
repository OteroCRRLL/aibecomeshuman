"""Reading subtitles on the PC, for free: the Windows OCR engine (Windows 10/11, CPU).

Subtitles change every few seconds; reading them with a vision model meant sending a
frame for each one. The OCR reads every sampled frame in ~40 ms, so the vision model only
looks at what needs eyes (shots, menus, action). Tested on the first mission: 11 of 12
labelled lines read correctly, plus lines the vision model had missed.

Unavailable outside Windows or without the Spanish OCR language: `SubtitleReader.create`
returns None and perception falls back to sending subtitle frames to the observer.
"""

from __future__ import annotations

import asyncio
import re

import cv2
import numpy as np

from aibecomeshuman.perception.transcript import similar

# Where subtitles sit (fractions of the frame): bottom band, centred.
BAND_Y = (0.78, 0.97)
BAND_X = (0.10, 0.90)


def clean(text: str) -> str:
    """Fix what the OCR typically gets wrong in Spanish subtitles."""
    text = " ".join(text.split())
    # "iAtrás!" / "i Te lo ruego!" -> "¡Atrás!" / "¡Te lo ruego!"
    text = re.sub(r"(^|[\s\-.])i ?(?=[A-ZÁÉÍÓÚÑ¿])", r"\1¡", text)
    # "lo" read as "10" ("10 mismo que tú", "ya 10 habrías hecho")
    text = re.sub(r"(?<![\d:.,])\b10\b(?![\d:.,])", "lo", text)
    return text.strip(" '`")


def split_speakers(text: str) -> list[str]:
    """A caption can hold two speakers: "- Hola. - ¿Qué quieres?" -> two lines (dash kept)."""
    parts = [p.strip() for p in re.split(r"\s+(?=-\s)", text) if p.strip()]
    return parts or ([text] if text else [])


def subtitle_band(frame: np.ndarray) -> np.ndarray:
    """Black text on white, upscaled: what OCR engines read best."""
    h, w = frame.shape[:2]
    band = frame[int(h * BAND_Y[0]) : int(h * BAND_Y[1]), int(w * BAND_X[0]) : int(w * BAND_X[1])]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    white = (hsv[..., 2] > 200) & (hsv[..., 1] < 50)
    image = np.where(white, 0, 255).astype(np.uint8)
    return cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)


class SubtitleReader:
    def __init__(self, engine) -> None:
        self._engine = engine
        self._loop = asyncio.new_event_loop()

    @classmethod
    def create(cls, language: str = "es-ES") -> SubtitleReader | None:
        try:
            from winrt.windows.globalization import Language
            from winrt.windows.media.ocr import OcrEngine
        except ImportError:
            return None
        try:
            engine = OcrEngine.try_create_from_language(Language(language))
        except Exception:
            return None
        return cls(engine) if engine is not None else None

    def read(self, frame: np.ndarray) -> str:
        return clean(self._loop.run_until_complete(self._recognize(subtitle_band(frame))))

    async def _recognize(self, image: np.ndarray) -> str:
        from winrt.windows.graphics.imaging import BitmapDecoder
        from winrt.windows.storage.streams import DataWriter, InMemoryRandomAccessStream

        _, png = cv2.imencode(".png", image)
        stream = InMemoryRandomAccessStream()
        writer = DataWriter(stream)
        writer.write_bytes(png.tobytes())
        await writer.store_async()
        stream.seek(0)
        decoder = await BitmapDecoder.create_async(stream)
        bitmap = await decoder.get_software_bitmap_async()
        result = await self._engine.recognize_async(bitmap)
        return " ".join(line.text for line in result.lines)

    def close(self) -> None:
        self._loop.close()


class SubtitleTracker:
    """Turns per-frame OCR text into new subtitle lines, once each.

    A caption counts once it reads the same on two samples in a row (fades and half-drawn
    text are skipped), and only if it differs from the last caption reported.
    """

    def __init__(self) -> None:
        self._candidate = ""
        self._last_parts: list[str] = []

    def update(self, text: str) -> list[str]:
        if len(text) < 2:
            self._candidate = ""
            return []
        stable = bool(self._candidate) and similar(self._candidate, text) >= 0.85
        self._candidate = text
        if not stable:
            return []
        # Only the parts not already reported ("A. - B." after "B." adds just "A.").
        parts = split_speakers(text)
        new = [p for p in parts if all(similar(p, old) < 0.85 for old in self._last_parts)]
        if new:
            self._last_parts = parts
        return new
