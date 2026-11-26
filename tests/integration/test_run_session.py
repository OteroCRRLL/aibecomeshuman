from __future__ import annotations

import json

import pytest

from aibecomeshuman.cli.main import main
from aibecomeshuman.config import ProjectConfig
from aibecomeshuman.core.clock import FakeClock
from aibecomeshuman.core.events import read_events
from aibecomeshuman.llm.types import LLMRequest, Message
from aibecomeshuman.run.session import RunSession


def test_empty_run_writes_manifest_and_events(project: ProjectConfig, clock: FakeClock) -> None:
    with RunSession(project, "offline", clock=clock) as session:
        pass

    manifest = json.loads((session.dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["profile"]["name"] == "offline"
    assert manifest["models"]["decider"]["provider"] == "fake"
    assert "constitution/base.md" in manifest["prompts"]
    assert manifest["ended_at"] is not None

    types = [e.type for e in read_events(session.dir / "events.jsonl")]
    assert types == ["run.started", "run.finished"]


def test_llm_calls_are_recorded_and_accounted(project: ProjectConfig, clock: FakeClock) -> None:
    with RunSession(project, "offline", clock=clock) as session:
        decider = session.llm("decider")
        assert session.llm("decider") is decider  # una instancia por rol
        decider.complete(LLMRequest(purpose="t", messages=[Message.user("hola")]))
        decider.complete(LLMRequest(purpose="t", messages=[Message.user("otra vez")]))

    manifest = json.loads((session.dir / "manifest.json").read_text(encoding="utf-8"))
    stats = manifest["usage_by_role"]["decider"]
    assert stats["calls"] == 2 and stats["errors"] == 0
    assert stats["usage"]["output_tokens"] > 0
    lines = (session.dir / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_exception_marks_run_as_error(project: ProjectConfig) -> None:
    session = RunSession(project, "offline")
    with pytest.raises(RuntimeError), session:
        raise RuntimeError("fallo de prueba")
    manifest = json.loads((session.dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "error" and "fallo de prueba" in manifest["error"]


def test_same_second_runs_get_unique_dirs(project: ProjectConfig) -> None:
    fixed = FakeClock()
    with RunSession(project, "offline", clock=fixed) as first:
        pass
    with RunSession(project, "offline", clock=fixed) as second:
        pass
    assert first.dir != second.dir


def test_unassigned_role_raises(project: ProjectConfig, project_dir) -> None:
    (project_dir / "config" / "runs" / "minimal.toml").write_text(
        'character = "connor"\n[roles]\ndecider = "fake"\n', encoding="utf-8"
    )
    from aibecomeshuman.config import load_project

    with RunSession(load_project(project_dir), "minimal") as session, pytest.raises(KeyError):
        session.llm("observer")


# -- CLI -------------------------------------------------------------------------


def test_cli_check_run_ping_events(project_dir, capsys) -> None:
    root = ["--root", str(project_dir)]
    assert main([*root, "check"]) == 0
    assert main([*root, "run", "-p", "offline"]) == 0
    assert main([*root, "ping", "-p", "offline", "-r", "decider", "hola agente"]) == 0
    out = capsys.readouterr().out
    assert "[fake/fake-1]" in out and "hola agente" in out

    run_dir = sorted((project_dir / "runs").iterdir())[-1]
    assert main(["events", str(run_dir), "-t", "llm.call"]) == 0
    assert "llm.call" in capsys.readouterr().out


def test_cli_unknown_profile_returns_error(project_dir, capsys) -> None:
    assert main(["--root", str(project_dir), "run", "-p", "nope"]) == 2
    assert "nope" in capsys.readouterr().err
