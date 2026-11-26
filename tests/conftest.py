from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from aibecomeshuman.config import ProjectConfig, load_project
from aibecomeshuman.core.clock import FakeClock

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Copia de config/ y prompts/ en un directorio temporal (las runs no ensucian el repo)."""
    shutil.copytree(REPO_ROOT / "config", tmp_path / "config")
    shutil.copytree(REPO_ROOT / "prompts", tmp_path / "prompts")
    return tmp_path


@pytest.fixture
def project(project_dir: Path) -> ProjectConfig:
    return load_project(project_dir)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(auto_step=0.5)
