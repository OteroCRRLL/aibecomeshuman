from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from aibecomeshuman.ui.server import create_app

from .test_ui import wait_for


@pytest.fixture
def client(project_dir: Path) -> TestClient:
    folder = project_dir / "data" / "recordings"
    folder.mkdir(parents=True)
    writer = cv2.VideoWriter(
        str(folder / "clip.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 180)
    )
    for i in range(40):
        writer.write(np.full((180, 320, 3), 60 if i < 20 else 160, np.uint8))
    writer.release()
    (folder / "notes.txt").write_text("not a video", encoding="utf-8")
    return TestClient(create_app(project_dir))


def test_lists_only_videos(client: TestClient) -> None:
    rows = client.get("/api/recordings").json()
    assert [r["file"] for r in rows] == ["clip.avi"]
    assert rows[0]["duration_s"] == 4 and rows[0]["labels"] is False


def test_perceive_then_view_transcript_and_frames(client: TestClient) -> None:
    res = client.post("/api/perceive", json={"recording": "clip.avi", "profile": "offline"})
    assert res.status_code == 200, res.text
    job = wait_for(client, res.json()["job_id"])
    assert job["status"] == "ok", job["error"]
    run_id = job["run_id"]

    data = client.get(f"/api/runs/{run_id}/perception").json()
    assert data["has_observations"] and data["transcript"]
    frame = data["transcript"][0]["frame"]
    image = client.get(f"/api/runs/{run_id}/frames/{frame}")
    assert image.status_code == 200 and image.headers["content-type"] == "image/jpeg"

    runs = client.get("/api/runs").json()
    assert next(r for r in runs if r["run_id"] == run_id)["scene"] == "Recording: clip.avi"

    rebuilt = client.post(
        "/api/perceive",
        json={"recording": "clip.avi", "profile": "offline", "rebuild_from": run_id},
    )
    assert wait_for(client, rebuilt.json()["job_id"])["status"] == "ok"


def test_dry_run_and_bad_inputs(client: TestClient) -> None:
    res = client.post(
        "/api/perceive", json={"recording": "clip.avi", "profile": "offline", "dry_run": True}
    )
    job = wait_for(client, res.json()["job_id"])
    started = next(e for e in job["all_events"] if e["type"] == "perception.started")
    assert started["payload"]["dry_run"] is True
    assert not any(e["type"] == "llm.call" for e in job["all_events"])

    for name in ("../config/default.toml", "notes.txt", "missing.avi"):
        res = client.post("/api/perceive", json={"recording": name, "profile": "offline"})
        assert res.status_code in (400, 404)
    assert client.get(f"/api/runs/{job['run_id']}/frames/..%2Fmanifest.json").status_code in (
        400,
        404,
    )
    assert client.get("/api/runs/nope/perception").status_code == 404
