"""Reading recordings and choosing which frames are worth showing to the observer model.

Everything here is local and free: frames are decoded with OpenCV and compared through
tiny thumbnails. Only frames where something changed (or after a quiet stretch) are
kept, so the paid model sees a few hundred frames instead of every second of video.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from aibecomeshuman.config import ConfigError, PerceptionSettings

VIDEO_SUFFIXES = {".mkv", ".mp4", ".mov", ".avi", ".webm"}

# Thumbnail used to detect changes, and the bottom band where subtitles appear.
THUMB_W, THUMB_H = 64, 36
SUBTITLE_BAND = (0.78, 0.97)
# Per-pixel difference (0-255) that counts as "this part of the picture changed".
PIXEL_DELTA = 25
# Share of the subtitle text that must differ to count as a new subtitle, and the least
# amount of text (cells of the small mask) that counts as a subtitle at all.
SUBTITLE_DELTA = 0.35
SUBTITLE_MIN_CELLS = 6
# A choice menu is likely when this many button letters line up; re-check it this often.
MENU_MIN = 2
# While the same menu stays up (the operator thinking) it is re-read only this often: the
# pick is worked out from what is said afterwards, not from a highlighted frame.
MENU_REPEAT_S = 6.0


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frames: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps if self.fps else 0.0


@dataclass
class Keyframe:
    t: float
    image: np.ndarray  # BGR, already resized to the observer width
    reason: str  # "first" | "menu" | "change" | "subtitles" | "heartbeat"


def open_video(path: Path) -> tuple[cv2.VideoCapture, VideoInfo]:
    if not path.is_file():
        raise ConfigError(f"recording not found: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ConfigError(f"cannot open the recording: {path}")
    info = VideoInfo(
        fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
        frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    return cap, info


def video_info(path: Path) -> VideoInfo:
    cap, info = open_video(path)
    cap.release()
    return info


def sample_frames(
    path: Path, every_s: float, start_s: float = 0.0, end_s: float | None = None
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (seconds, frame) every `every_s`, decoding sequentially (faster than seeking)."""
    cap, info = open_video(path)
    try:
        end = min(end_s, info.duration_s) if end_s is not None else info.duration_s
        if start_s > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
        index = int(round(start_s * info.fps))
        step = max(1, int(round(every_s * info.fps)))
        next_index = index
        while True:
            if index >= next_index:
                ok, frame = cap.read()
                if not ok:
                    return
                t = index / info.fps
                if t > end:
                    return
                yield t, frame
                next_index += step
            elif not cap.grab():
                return
            index += 1
    finally:
        cap.release()


def thumbnail(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA).astype(np.int16)


def subtitle_mask(frame: np.ndarray) -> np.ndarray:
    """White text in the subtitle band, as a small mask.

    Subtitle letters are white with a dark outline, so only white pixels that have dark
    pixels right next to them count: bright lights in the background mostly do not.
    """
    h = frame.shape[0]
    band = frame[int(h * SUBTITLE_BAND[0]) : int(h * SUBTITLE_BAND[1])]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    white = (hsv[..., 2] > 225) & (hsv[..., 1] < 30)
    dark = (hsv[..., 2] < 90).astype(np.uint8)
    near_dark = cv2.dilate(dark, np.ones((5, 5), np.uint8)) > 0
    text = (white & near_dark).astype(np.uint8) * 255
    small = cv2.resize(text, (128, 12), interpolation=cv2.INTER_AREA)
    return small > 40


# Button letter colours (OpenCV hue, 0-179): Y yellow, X blue, B red, A green.
BUTTON_HUES: dict[str, tuple[tuple[int, int], ...]] = {
    "Y": ((18, 33),),
    "X": ((95, 115),),
    "B": ((0, 8), (170, 179)),
    "A": ((34, 80),),
}


def menu_score(frame: np.ndarray) -> int:
    """How many different button letters sit in one column (0-4). >= 2 suggests a choice menu.

    Looks for small, strongly coloured, letter-sized blobs: cheap enough to run on every
    frame, so a menu is never missed between two observer calls. False positives (blue
    interface elements, neon) only cost an extra keyframe.
    """
    small = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0]
    strong = (hsv[..., 1] > 110) & (hsv[..., 2] > 140)
    points: list[tuple[str, int]] = []
    for name, ranges in BUTTON_HUES.items():
        mask = np.zeros(strong.shape, dtype=bool)
        for lo, hi in ranges:
            mask |= (hue >= lo) & (hue <= hi)
        n, _, stats, _ = cv2.connectedComponentsWithStats(
            (mask & strong).astype(np.uint8), connectivity=8
        )
        for i in range(1, n):
            x, _, w, h, area = (int(v) for v in stats[i])
            if 8 <= h <= 30 and 5 <= w <= 26 and 20 <= area <= 320 and 0.5 <= h / w <= 2.5:
                points.append((name, x))
    best = 0
    for _, x0 in points:
        best = max(best, len({name for name, x in points if abs(x - x0) <= 40}))
    return best


def new_subtitle(subs: np.ndarray, previous: np.ndarray) -> bool:
    """A different subtitle is on screen: compared to the size of the text, not of the band,
    so a short line ("Yes.") counts as much as a long one."""
    on = int(subs.sum())
    if on < SUBTITLE_MIN_CELLS:
        return False
    changed = int(np.sum(subs != previous))
    return changed / max(on, int(previous.sum())) > SUBTITLE_DELTA


def changed_ratio(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b) > PIXEL_DELTA))


def resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    if frame.shape[1] <= width:
        return frame
    height = int(frame.shape[0] * width / frame.shape[1])
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def shrink_jpeg(jpeg: bytes, width: int, quality: int = 75) -> bytes:
    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    return encode_jpeg(resize_to_width(frame, width), quality)


def encode_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not encode frame as JPEG")
    return buf.tobytes()


def select_keyframes(
    frames: Iterator[tuple[float, np.ndarray]], settings: PerceptionSettings
) -> Iterator[Keyframe]:
    """Keep the frames where a menu shows up, the picture or the subtitles changed, plus a
    heartbeat. While a menu is on screen, a frame every 2 s (to see which option is picked)."""
    last_thumb: np.ndarray | None = None
    last_subs: np.ndarray | None = None
    last_sent = -1e9
    last_menu = 0
    for t, frame in frames:
        if float(frame.mean()) < settings.dark_threshold:
            continue
        thumb = thumbnail(frame)
        subs = subtitle_mask(frame)
        menu = menu_score(frame)
        reason = None
        if last_thumb is None:
            reason = "first"
        elif menu >= MENU_MIN and (last_menu < MENU_MIN or t - last_sent >= MENU_REPEAT_S):
            reason = "menu"
        elif last_subs is not None and new_subtitle(subs, last_subs):
            reason = "subtitles"
        elif changed_ratio(thumb, last_thumb) >= settings.change_ratio:
            reason = "change"
        elif t - last_sent >= settings.heartbeat_s:
            reason = "heartbeat"
        last_menu = menu
        if reason is None:
            continue
        last_thumb, last_subs, last_sent = thumb, subs, t
        width = settings.menu_frame_width if menu >= MENU_MIN else settings.frame_width
        yield Keyframe(t=t, image=resize_to_width(frame, width), reason=reason)


def fmt_time(t: float) -> str:
    minutes, seconds = divmod(int(t), 60)
    return f"{minutes:02d}:{seconds:02d}"


def parse_time(value: str | float | int | None) -> float | None:
    """'08:16' / '1:02:03' / 496 -> seconds."""
    if value is None or value == "":
        return None
    if isinstance(value, int | float):
        return float(value)
    seconds = 0.0
    for part in value.strip().split(":"):
        seconds = seconds * 60 + float(part)
    return seconds
