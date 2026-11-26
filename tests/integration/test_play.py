from __future__ import annotations

import pytest

from aibecomeshuman.cli.main import main
from aibecomeshuman.config import ConfigError, ProjectConfig
from aibecomeshuman.core.events import read_events
from aibecomeshuman.env.scenario import load_scenario
from aibecomeshuman.play import ScenarioRunner
from aibecomeshuman.run.session import RunSession

from ..conftest import REPO_ROOT

SCENARIOS = sorted((REPO_ROOT / "scenarios").glob("*.toml"))


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_every_scenario_plays_offline(project: ProjectConfig, path) -> None:
    scenario = load_scenario(path)
    lines: list[str] = []
    with RunSession(project, "offline") as session:
        decisions = ScenarioRunner(session, scenario, echo=lines.append).run()

    choices = [s for s in scenario.steps if s.kind == "choice"]
    assert len(decisions) == len(choices)
    types = [e.type for e in read_events(session.dir / "events.jsonl")]
    assert types[0] == "run.started" and types[-1] == "run.finished"
    assert types.count("decision.made") == len(choices)
    assert "scene.started" in types and "scene.finished" in types
    assert "stance.updated" in types
    assert any(line.startswith("  =>") for line in lines)


def test_directive_override_and_hot_swap(project: ProjectConfig) -> None:
    scenario = load_scenario(REPO_ROOT / "scenarios" / "first-snow.toml")
    with RunSession(project, "offline") as session:
        runner = ScenarioRunner(session, scenario, directive="bad_ideas")
        runner.change_directive("default")
        runner.run()
    events = list(read_events(session.dir / "events.jsonl"))
    changed = next(e for e in events if e.type == "directive.changed")
    assert changed.payload == {"from": "bad_ideas", "to": "default"}
    decisions = [e for e in events if e.type == "decision.made"]
    assert {d.payload["directive"] for d in decisions} == {"default"}


def test_character_must_belong_to_run(project_dir) -> None:
    from aibecomeshuman.config import load_project

    (project_dir / "config" / "runs" / "only-connor.toml").write_text(
        'characters = ["connor"]\n[roles]\ndecider = "fake"\n', encoding="utf-8"
    )
    scenario = load_scenario(REPO_ROOT / "scenarios" / "first-snow.toml")  # kara
    with (
        RunSession(load_project(project_dir), "only-connor") as session,
        pytest.raises(ConfigError, match="kara"),
    ):
        ScenarioRunner(session, scenario)


def test_cli_play_and_budget(project_dir, capsys) -> None:
    root = ["--root", str(project_dir)]
    scenario = str(REPO_ROOT / "scenarios" / "tram-stop.toml")
    assert main([*root, "play", scenario, "-p", "offline"]) == 0
    out = capsys.readouterr().out
    assert "The tram stop" in out and "2 decisions" in out

    assert main([*root, "budget"]) == 0
    assert "Project spend: $0.0000" in capsys.readouterr().out


def test_cli_premium_run_refused_without_unlock(project_dir, capsys) -> None:
    scenario = str(REPO_ROOT / "scenarios" / "tram-stop.toml")
    assert main(["--root", str(project_dir), "play", scenario, "-p", "video-claude"]) == 2
    assert "unlock-premium" in capsys.readouterr().err
