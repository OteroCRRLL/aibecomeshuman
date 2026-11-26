from __future__ import annotations

import shutil
import time
import tomllib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aibecomeshuman.ui.server import create_app

from ..conftest import REPO_ROOT


@pytest.fixture
def client(project_dir: Path) -> TestClient:
    shutil.copytree(REPO_ROOT / "scenarios", project_dir / "scenarios")
    return TestClient(create_app(project_dir))


def wait_for(client: TestClient, job_id: str, timeout: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout
    events: list[dict] = []
    after = 0
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}", params={"after": after}).json()
        events += job["events"]
        after = job["next"]
        if job["status"] != "running":
            return job | {"all_events": events}
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_index_and_options(client: TestClient) -> None:
    assert "aibecomeshuman" in client.get("/").text
    options = client.get("/api/options").json()
    assert {p["name"] for p in options["profiles"]} >= {"default", "offline", "video-claude"}
    assert next(p for p in options["profiles"] if p["name"] == "video-claude")["locked"]
    assert {s["file"] for s in options["scenarios"]} >= {"tram-stop.toml", "first-snow.toml"}
    assert "bad_ideas" in options["directives"]


def test_status_never_exposes_the_key(client: TestClient, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-value")
    body = client.get("/api/status").text
    assert "sk-ant-secret-value" not in body
    assert client.get("/api/status").json()["api_key_configured"] is True


def test_play_streams_events_and_lists_run(client: TestClient) -> None:
    res = client.post(
        "/api/play",
        json={"scenario": "tram-stop.toml", "profile": "offline", "directive": "bad_ideas"},
    )
    assert res.status_code == 200
    job = wait_for(client, res.json()["job_id"])
    assert job["status"] == "ok", job["error"]
    types = [e["type"] for e in job["all_events"]]
    assert types[0] == "run.started" and types[-1] == "run.finished"
    assert types.count("decision.made") == 2

    runs = client.get("/api/runs").json()
    assert runs[0]["run_id"] == job["run_id"] and runs[0]["decisions"] == 2
    assert runs[0]["scene"] == "The tram stop"
    events = client.get(f"/api/runs/{job['run_id']}/events").json()
    assert len(events) == len(job["all_events"])


def test_play_rejects_bad_requests(client: TestClient) -> None:
    locked = client.post(
        "/api/play", json={"scenario": "tram-stop.toml", "profile": "video-claude"}
    )
    assert locked.status_code == 400 and "unlock-premium" in locked.json()["detail"]
    traversal = client.post(
        "/api/play", json={"scenario": "../config/default.toml", "profile": "offline"}
    )
    assert traversal.status_code == 400
    assert client.get("/api/runs/..%2Fconfig/events").status_code == 404


def test_budget_update_is_validated_and_written(client: TestClient, project_dir: Path) -> None:
    ok = client.put(
        "/api/settings/budget",
        json={"project_cap_usd": 30, "run_cap_usd": 1.5, "warn_ratio": 0.9},
    )
    assert ok.status_code == 200
    text = (project_dir / "config" / "default.toml").read_text(encoding="utf-8")
    assert tomllib.loads(text)["budget"]["project_cap_usd"] == 30
    assert "# Hard caps in USD" in text  # comments survive

    bad = client.put(
        "/api/settings/budget",
        json={"project_cap_usd": -1, "run_cap_usd": 1, "warn_ratio": 0.9},
    )
    assert bad.status_code == 400


def test_roles_update_and_rejections(client: TestClient, project_dir: Path) -> None:
    path = project_dir / "config" / "runs" / "default.toml"
    ok = client.put("/api/settings/runs/default/roles", json={"judge": "fake"})
    assert ok.status_code == 200 and ok.json()["judge"] == "fake"
    assert tomllib.loads(path.read_text(encoding="utf-8"))["roles"]["judge"] == "fake"

    before = path.read_text(encoding="utf-8")
    assert client.put("/api/settings/runs/default/roles", json={"judge": "nope"}).status_code == 400
    assert client.put("/api/settings/runs/default/roles", json={"pilot": "fake"}).status_code == 400
    assert client.put("/api/settings/runs/missing/roles", json={"judge": "fake"}).status_code == 404
    assert path.read_text(encoding="utf-8") == before


def test_api_key_saved_to_env_file_and_picked_up(
    client: TestClient, project_dir: Path, monkeypatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert client.get("/api/status").json()["api_key_configured"] is False

    bad = client.put("/api/settings/api-key", json={"key": "hello"})
    assert bad.status_code == 400

    key = "sk-ant-test-" + "x" * 40
    res = client.put("/api/settings/api-key", json={"key": key})
    assert res.status_code == 200 and res.json()["overridden_by_environment"] is False
    assert f"ANTHROPIC_API_KEY={key}" in (project_dir / ".env").read_text(encoding="utf-8")

    status = client.get("/api/status")
    assert status.json()["api_key_configured"] is True
    assert status.json()["api_key_source"] == "file"
    assert key not in status.text  # never echoed back
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_key_added_after_startup_is_picked_up(
    client: TestClient, project_dir: Path, monkeypatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert client.get("/api/status").json()["api_key_configured"] is False
    (project_dir / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-later-" + "y" * 30 + "\n")
    assert client.get("/api/status").json()["api_key_configured"] is True
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_ping_with_fake_model(client: TestClient) -> None:
    res = client.post("/api/ping", json={"profile": "offline", "role": "decider_fast"})
    assert res.status_code == 200, res.text
    assert res.json()["model"] == "fake-1" and res.json()["cost_usd"] == 0
    assert client.post("/api/ping", json={"profile": "video-claude"}).status_code == 400


def test_local_only_protection(client: TestClient) -> None:
    assert client.get("/api/status", headers={"host": "evil.example"}).status_code == 403
    cross_site = client.put(
        "/api/settings/budget",
        json={"project_cap_usd": 99, "run_cap_usd": 1, "warn_ratio": 0.8},
        headers={"origin": "https://evil.example"},
    )
    assert cross_site.status_code == 403
    form_post = client.post(
        "/api/play",
        content="scenario=tram-stop.toml",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    assert form_post.status_code == 415
    same_site = client.get("/api/status", headers={"origin": "http://127.0.0.1:8765"})
    assert same_site.status_code == 200


def test_prompts_endpoint(client: TestClient) -> None:
    data = client.get("/api/prompts").json()
    assert {"connor", "kara", "markus"} <= set(data["characters"])
    assert data["directives"]["bad_ideas"]["meta"]["priority"] == "order"
