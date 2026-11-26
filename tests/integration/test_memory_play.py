from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from fastapi.testclient import TestClient

from aibecomeshuman.cli.main import main
from aibecomeshuman.config import ProjectConfig, load_project
from aibecomeshuman.core.events import read_events
from aibecomeshuman.env.scenario import load_chain, load_scenario
from aibecomeshuman.memory.store import MemoryStore
from aibecomeshuman.play import play_scenarios
from aibecomeshuman.run.session import RunSession
from aibecomeshuman.ui.server import create_app

from ..conftest import REPO_ROOT

SCENARIOS = REPO_ROOT / "scenarios"
CHAIN = SCENARIOS / "chains" / "markus-week.toml"


def chain_scenes():
    return load_chain(CHAIN, SCENARIOS)[1]


def test_chain_shares_memory_and_follows_earlier_choices(project: ProjectConfig) -> None:
    with RunSession(project, "offline") as session:
        decisions = play_scenarios(session, chain_scenes(), campaign="week")
    events = list(read_events(session.dir / "events.jsonl"))

    # The fake model always picks option A: walked her home -> asked why late.
    choices = [e.payload["choice_id"] for e in events if e.type == "decision.made"]
    assert choices == [
        "platform", "escort", "explain_late", "right_thing", "thursdays", "afternoon"
    ]
    assert len(decisions) == 6
    assert sum(e.type == "memory.consolidated" for e in events) == 4

    starts = [e.payload for e in events if e.type == "scene.started"]
    assert [s["memories"] for s in starts] == [0, 1, 2, 3]
    assert all(s["campaign"] == "week" for s in starts)

    store = MemoryStore(project.campaigns_dir / "week")
    memory = store.read("markus")
    assert [i.id for i in memory.items] == ["M1", "M2", "M3", "M4"]
    assert store.load_state().choices["tram-stop.escort"] == "A"


def test_memory_reaches_later_prompts(project: ProjectConfig) -> None:
    with RunSession(project, "offline") as session:
        play_scenarios(session, chain_scenes()[:2], campaign="prompts")
    calls = [
        json.loads(line)
        for line in (session.dir / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    stance_calls = [c for c in calls if c["request"]["purpose"] == "stance.update"]
    first, last = stance_calls[0], stance_calls[-1]
    first_text = first["request"]["messages"][0]["content"][0]["text"]
    last_text = last["request"]["messages"][0]["content"][0]["text"]
    assert "no memories" in first_text
    assert '<recollection id="M1"' in last_text
    assert "<personality" in last["request"]["system"]
    purposes = {c["request"]["purpose"] for c in calls}
    assert {"memory.consolidate", "personality.reflect"} <= purposes


def test_campaign_persists_across_runs(project: ProjectConfig) -> None:
    scenes = chain_scenes()
    with RunSession(project, "offline") as first:
        play_scenarios(first, scenes[:2], campaign="saga")
    with RunSession(project, "offline") as second:
        play_scenarios(second, scenes[2:], campaign="saga")
    memory = MemoryStore(project.campaigns_dir / "saga").read("markus")
    assert len(memory.items) == 4
    events = list(read_events(second.dir / "events.jsonl"))
    # The market scene depended on a choice made in the first run.
    lines = [e.payload["text"] for e in events if e.type == "scene.perceived"]
    assert any("Me ayudaste a levantarme" in t for t in lines)


def test_without_campaign_memory_stays_in_the_run(project: ProjectConfig) -> None:
    with RunSession(project, "offline") as session:
        play_scenarios(session, [load_scenario(SCENARIOS / "first-snow.toml")])
    assert (session.dir / "memory" / "kara.json").is_file()
    assert not project.campaigns_dir.exists()


def test_unknown_condition_skips_step(project: ProjectConfig) -> None:
    # Played alone, the market scene cannot know what happened on the platform.
    market = load_scenario(SCENARIOS / "markus-week" / "03-market.toml")
    with RunSession(project, "offline") as session:
        play_scenarios(session, [market])
    events = list(read_events(session.dir / "events.jsonl"))
    unknown = [e.payload["choice"] for e in events if e.type == "scene.condition_unknown"]
    assert unknown == ["tram-stop.platform"] * 3
    assert sum(e.type == "decision.made" for e in events) == 1


def test_imposed_directive_does_not_reshape_personality(project: ProjectConfig) -> None:
    with RunSession(project, "offline") as session:
        play_scenarios(
            session, [load_scenario(SCENARIOS / "tram-stop.toml")], directive="bad_ideas"
        )
    events = list(read_events(session.dir / "events.jsonl"))
    assert any(e.type == "personality.skipped" for e in events)
    calls = (session.dir / "llm_calls.jsonl").read_text(encoding="utf-8")
    assert "personality.reflect" not in calls


def test_cli_plays_a_chain_with_campaign(project_dir: Path, capsys) -> None:
    assert main(["--root", str(project_dir), "play", str(CHAIN), "-p", "offline", "-c", "cli"]) == 0
    out = capsys.readouterr().out
    assert "6 decisions" in out and "campaign 'cli'" in out and "## Memory M4" in out
    assert main(["--root", str(project_dir), "play", str(CHAIN), "-p", "offline", "-c", "a/b"]) == 2


def test_ui_chain_play_and_memory_view(project_dir: Path) -> None:
    shutil.copytree(SCENARIOS, project_dir / "scenarios")
    client = TestClient(create_app(project_dir))
    options = client.get("/api/options").json()
    assert options["chains"][0]["file"] == "chains/markus-week.toml"
    assert options["campaigns"] == []

    job_id = client.post(
        "/api/play",
        json={"scenario": "chains/markus-week.toml", "profile": "offline", "campaign": "ui"},
    ).json()["job_id"]
    deadline = time.monotonic() + 30
    while client.get(f"/api/jobs/{job_id}").json()["status"] == "running":
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "ok"

    assert client.get("/api/options").json()["campaigns"] == ["ui"]
    assert client.get("/api/campaigns/ui").json()["characters"] == ["markus"]
    memory = client.get("/api/campaigns/ui/markus").json()
    assert len(memory["items"]) == 4 and "obedience" in memory["personality"]["traits"]

    assert client.get("/api/campaigns/nope").status_code == 404
    assert client.get("/api/campaigns/ui/kara").status_code == 404
    bad = client.post(
        "/api/play", json={"scenario": "tram-stop.toml", "profile": "offline", "campaign": "../x"}
    )
    assert bad.status_code == 400
    assert load_project(project_dir).campaigns_dir == project_dir / "campaigns"
