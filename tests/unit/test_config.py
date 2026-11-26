from __future__ import annotations

from pathlib import Path

import pytest

from aibecomeshuman.config import ConfigError, load_project

from ..conftest import REPO_ROOT


def test_repo_config_is_valid() -> None:
    project = load_project(REPO_ROOT)
    assert {"default", "offline"} <= set(project.runs)
    assert {"fake", "claude-opus", "claude-haiku"} <= set(project.models)
    offline_models = project.models_for(project.runs["offline"]).values()
    assert {m.name for m in offline_models} == {"fake"}


def test_finds_root_from_subdirectory(project_dir: Path) -> None:
    sub = project_dir / "prompts" / "constitution"
    assert load_project(sub).root == project_dir.resolve()


def test_unknown_model_reference_fails(project_dir: Path) -> None:
    (project_dir / "config" / "runs" / "broken.toml").write_text(
        'characters = ["connor"]\n[roles]\ndecider = "missing-model"\n', encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="missing-model"):
        load_project(project_dir)


def test_missing_directive_file_fails(project_dir: Path) -> None:
    (project_dir / "config" / "runs" / "broken.toml").write_text(
        'characters = ["connor"]\ndirective = "chaos"\n[roles]\ndecider = "fake"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="directives/chaos.md"):
        load_project(project_dir)


def test_unknown_role_fails(project_dir: Path) -> None:
    (project_dir / "config" / "runs" / "broken.toml").write_text(
        'characters = ["connor"]\n[roles]\ndecider = "fake"\npilot = "fake"\n', encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="pilot"):
        load_project(project_dir)


def test_unknown_model_field_fails(project_dir: Path) -> None:
    (project_dir / "config" / "models" / "typo.toml").write_text(
        'provider = "fake"\nmodel = "x"\nefort = "low"\n', encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="efort"):
        load_project(project_dir)
