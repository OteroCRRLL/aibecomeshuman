from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from aibecomeshuman.config import ProjectConfig
from aibecomeshuman.core.events import read_events
from aibecomeshuman.live.shadow import ShadowSession
from aibecomeshuman.live.sources import RecordingSource, SourceLost
from aibecomeshuman.perception.observer import (
    Observation,
    SeenChoice,
    SeenOption,
    SubtitleLine,
)
from aibecomeshuman.run.session import RunSession
from aibecomeshuman.ui.server import create_app

from .test_ui import wait_for


def _clip(path: Path, seconds: int = 7) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 180))
    for i in range(seconds * 10):
        # A different brightness every second: every second is a new "shot".
        writer.write(np.full((180, 320, 3), 40 + 25 * ((i // 10) % 8), np.uint8))
    writer.release()
    return path


class ScriptedObserver:
    """Daniel speaks, a menu stays up from 2 to 7 s (the operator hesitates), then the
    protagonist says the picked line."""

    def observe(self, jpeg, t, label, ctx, earlier=None) -> Observation:
        if t < 2:
            lines = [SubtitleLine(text="- ¿Cómo sabes mi nombre?", speaker="Daniel",
                                  speaker_basis="visible_speaking")]
            return Observation(t=t, subtitles=lines)
        if t < 7:
            options = [SeenOption(button="Y", label="CALMADO"),
                       SeenOption(button="A", label="EMPATIZAR")]
            return Observation(t=t, choice=SeenChoice(kind="dialogue", timed=True, options=options))
        line = SubtitleLine(text=f"- Quiero empatizar contigo ({int(t)}).", speaker="Connor",
                            speaker_basis="after_choice")
        return Observation(t=t, subtitles=[line])


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    return _clip(tmp_path / "clip.avi", seconds=10)


def test_shadow_suggests_and_compares_with_the_operator(project: ProjectConfig, clip) -> None:
    with RunSession(project, "offline") as session:
        shadow = ShadowSession(session, RecordingSource(clip, speed=2), "connor")
        shadow.observer = ScriptedObserver()
        report = shadow.run()

    events = list(read_events(session.dir / "events.jsonl"))
    types = [e.type for e in events]
    assert report.observed == report.keyframes
    menu_times = [
        e.payload["t"] for e in events
        if e.type == "perception.observed" and e.payload["observation"]["choice"]["options"]
    ]
    assert len(menu_times) >= 2 and max(menu_times) - min(menu_times) > 3.5
    assert report.suggestions == 1  # one menu, however long the operator takes
    suggestion = next(e for e in events if e.type == "shadow.suggestion").payload
    assert suggestion["options"] == ["CALMADO", "EMPATIZAR"]
    compare = next(e for e in events if e.type == "shadow.compare").payload
    assert compare["picked"] == "EMPATIZAR" and compare["picked_by"] == "words"
    # Memory is kept, but the personality does not learn from the operator's picks.
    assert "memory.consolidated" in types
    skipped = next(e for e in events if e.type == "personality.skipped")
    assert "operator" in skipped.payload["reason"]
    assert types[-1] == "run.finished"


def test_shadow_stops_on_request(project: ProjectConfig, tmp_path: Path) -> None:
    stop = threading.Event()
    stop.set()
    with RunSession(project, "offline") as session:
        source = RecordingSource(_clip(tmp_path / "long.avi", seconds=30), speed=1)
        report = ShadowSession(session, source, "connor", stop=stop).run()
    assert report.keyframes == 0


class ClosingGame:
    """A source whose game is closed after a few frames."""

    name = "test:closing"

    def frames(self, every_s):
        for i in range(4):
            yield float(i), np.full((180, 320, 3), 40 + 40 * i, np.uint8)
        raise SourceLost("no picture")

    def close(self) -> None:
        pass


def test_closing_the_game_ends_normally(project: ProjectConfig) -> None:
    with RunSession(project, "offline") as session:
        ShadowSession(session, ClosingGame(), "connor").run()
    events = list(read_events(session.dir / "events.jsonl"))
    types = [e.type for e in events]
    assert "shadow.source_lost" in types and "memory.consolidated" in types
    assert events[-1].payload["status"] == "ok"


def test_live_tab_runs_a_recording(project_dir: Path) -> None:
    folder = project_dir / "data" / "recordings"
    folder.mkdir(parents=True)
    _clip(folder / "clip.avi", seconds=3)
    client = TestClient(create_app(project_dir))
    res = client.post(
        "/api/live/start", json={"source": "clip.avi", "profile": "offline", "speed": 6}
    )
    assert res.status_code == 200, res.text
    job = wait_for(client, res.json()["job_id"])
    assert job["status"] == "ok", job["error"]
    assert any(e["type"] == "shadow.finished" for e in job["all_events"])
    assert client.post("/api/live/stop", json={}).json() == {"stopping": True}
    bad = client.post("/api/live/start", json={"source": "../.env", "profile": "offline"})
    assert bad.status_code == 400
