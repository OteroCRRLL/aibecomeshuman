"""Live frame sources: where the agent's eyes get pictures from while the game runs.

- `ObsSource`: asks OBS for a screenshot of the game capture source (OBS 28+ has the
  WebSocket server built in: Tools -> WebSocket Server Settings). Nothing else to install,
  and it sees exactly what is being recorded.
- `ScreenSource`: grabs a whole monitor (the game in fullscreen or borderless).
- `RecordingSource`: plays a recording as if it were live, in real time or faster. For
  testing everything without the game.
"""

from __future__ import annotations

import base64
import contextlib
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from aibecomeshuman.config import ConfigError
from aibecomeshuman.perception.video import open_video


class FrameSource(Protocol):
    name: str

    def frames(self, every_s: float) -> Iterator[tuple[float, np.ndarray]]:
        """Yield (seconds since start, BGR frame) about every `every_s`, until stopped."""
        ...

    def close(self) -> None: ...


def _paced(grab, every_s: float) -> Iterator[tuple[float, np.ndarray]]:
    start = time.monotonic()
    next_at = start
    while True:
        now = time.monotonic()
        if now < next_at:
            time.sleep(next_at - now)
        frame = grab()
        if frame is not None:
            yield time.monotonic() - start, frame
        next_at += every_s
        if time.monotonic() > next_at + every_s:  # fell behind: do not try to catch up
            next_at = time.monotonic()


class ObsSource:
    def __init__(
        self, source: str, host: str = "localhost", port: int = 4455, password: str = "",
        width: int = 1920,
    ) -> None:
        try:
            import obsws_python as obsws
        except ImportError as e:  # pragma: no cover - dependency is installed with the project
            raise ConfigError("obsws-python is not installed (run `uv sync`)") from e
        self.name = f"obs:{source}"
        self.source = source
        self.width = width
        try:
            self._client = obsws.ReqClient(host=host, port=port, password=password, timeout=5)
        except Exception as e:
            raise ConfigError(
                f"cannot connect to OBS at {host}:{port} ({e}). In OBS: Tools -> WebSocket "
                "Server Settings -> Enable, and put the password in .env as "
                "OBS_WEBSOCKET_PASSWORD."
            ) from e

    def grab(self) -> np.ndarray | None:
        try:
            response = self._client.get_source_screenshot(self.source, "jpg", self.width, None, 90)
        except Exception as e:
            raise ConfigError(
                f"OBS could not capture source '{self.source}' ({e}). Check the source name "
                "in config/default.toml ([live] obs_source)."
            ) from e
        data = response.image_data.split(",", 1)[-1]
        return cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_COLOR)

    def frames(self, every_s: float) -> Iterator[tuple[float, np.ndarray]]:
        return _paced(self.grab, every_s)

    def close(self) -> None:
        with contextlib.suppress(Exception):  # closing a dead connection is fine
            self._client.disconnect()


class ScreenSource:
    def __init__(self, monitor: int = 1) -> None:
        import mss

        self.name = f"screen:{monitor}"
        self._mss = mss.mss()
        if monitor >= len(self._mss.monitors):
            raise ConfigError(f"there is no monitor {monitor}")
        self._monitor = self._mss.monitors[monitor]

    def grab(self) -> np.ndarray:
        return np.ascontiguousarray(np.array(self._mss.grab(self._monitor))[:, :, :3])

    def frames(self, every_s: float) -> Iterator[tuple[float, np.ndarray]]:
        return _paced(self.grab, every_s)

    def close(self) -> None:
        self._mss.close()


def build_source(
    spec: str, project, speed: float = 1.0, start_s: float = 0.0, recordings_dir: Path | None = None
) -> FrameSource:
    """'obs' | 'screen' | 'screen:2' | a recording path (or a file name in the recordings
    folder)."""
    import os

    if spec == "obs":
        live = project.app.live
        return ObsSource(
            live.obs_source, live.obs_host, live.obs_port,
            os.environ.get("OBS_WEBSOCKET_PASSWORD", ""),
        )
    if spec.startswith("screen"):
        _, _, monitor = spec.partition(":")
        return ScreenSource(int(monitor or 1))
    path = Path(spec)
    if not path.is_file() and recordings_dir is not None:
        path = recordings_dir / spec
    if not path.is_file():
        raise ConfigError(f"unknown source '{spec}': use obs, screen or a recording file")
    return RecordingSource(path, speed=speed, start_s=start_s)


class RecordingSource:
    """A recording played back on the wall clock (`speed` 2 = twice as fast)."""

    def __init__(self, path: Path, speed: float = 1.0, start_s: float = 0.0) -> None:
        self.name = f"recording:{path.name}"
        self.path = path
        self.speed = speed
        self.start_s = start_s
        self._cap, self.info = open_video(path)
        if start_s:
            self._cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)

    def frames(self, every_s: float) -> Iterator[tuple[float, np.ndarray]]:
        wall_start = time.monotonic()
        fps = self.info.fps
        index = int(round(self.start_s * fps))
        while True:
            wanted = self.start_s + (time.monotonic() - wall_start) * self.speed
            target = int(wanted * fps)
            while index < target:  # skip what the clock has already passed
                if not self._cap.grab():
                    return
                index += 1
            ok, frame = self._cap.read()
            if not ok:
                return
            index += 1
            yield index / fps, frame
            time.sleep(max(0.0, every_s - (time.monotonic() - wall_start) % every_s))

    def close(self) -> None:
        self._cap.release()
