from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from aibecomeshuman.cli.main import main
from aibecomeshuman.config import ProjectConfig
from aibecomeshuman.core.events import read_events
from aibecomeshuman.perception.pipeline import finish_outputs, perceive, rebuild
from aibecomeshuman.run.session import RunSession


@pytest.fixture
def video(tmp_path: Path) -> Path:
    """6 s synthetic clip: a cut at 2 s and a subtitle from 4 s."""
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 360))
    for i in range(60):
        frame = np.full((360, 640, 3), 60 if i < 20 else 150, np.uint8)
        if i >= 40:
            cv2.putText(frame, "Hola.", (280, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 6)
            cv2.putText(frame, "Hola.", (280, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,) * 3, 2)
        writer.write(frame)
    writer.release()
    return path


def test_perceive_offline_writes_transcript_and_events(project: ProjectConfig, video) -> None:
    with RunSession(project, "offline") as session:
        result = perceive(session, video, "connor")
        finish_outputs(session, video, result, "connor")

    assert 2 <= result.keyframes <= 6  # first frame, the cut, the subtitle (+ heartbeat)
    assert result.observed == result.keyframes and result.failed == 0
    assert (session.dir / "transcript.md").is_file()
    assert (session.dir / "observations.jsonl").is_file()
    assert len(list((session.dir / "frames").glob("*.jpg"))) == result.keyframes
    types = [e.type for e in read_events(session.dir / "events.jsonl")]
    assert types.count("perception.observed") == result.keyframes
    assert "perception.transcript" in types
    assert session.manifest.cost_usd == 0


def test_dry_run_calls_no_model(project: ProjectConfig, video) -> None:
    with RunSession(project, "offline") as session:
        result = perceive(session, video, "connor", dry_run=True)
    types = [e.type for e in read_events(session.dir / "events.jsonl")]
    assert result.keyframes > 0 and result.observed == 0
    assert "llm.call" not in types and "perception.observed" not in types


def test_rebuild_uses_saved_observations(project: ProjectConfig, video) -> None:
    with RunSession(project, "offline") as first:
        original = perceive(first, video, "connor")
    with RunSession(project, "offline") as second:
        again = rebuild(second, first.dir, video, "connor")
    assert again.observed == original.observed
    assert not (second.dir / "frames").exists()
    assert (second.dir / "transcript.json").is_file()


def test_cli_perceive(project: ProjectConfig, video, capsys) -> None:
    code = main(["--root", str(project.root), "perceive", str(video), "-p", "offline"])
    assert code == 0
    assert "frames observed" in capsys.readouterr().out
