from __future__ import annotations

import os
from pathlib import Path

import pytest

from aibecomeshuman.envfile import load_dotenv, set_env_value, source_of


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("AIBH_TEST_A", "AIBH_TEST_B", "AIBH_TEST_EMPTY"):
        monkeypatch.delenv(key, raising=False)
    yield


def test_loads_without_overriding_real_environment(tmp_path: Path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# comment\nAIBH_TEST_A='from-file'\nAIBH_TEST_B=from-file\nAIBH_TEST_EMPTY=\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AIBH_TEST_B", "already-set")
    load_dotenv(env)

    assert os.environ["AIBH_TEST_A"] == "from-file"
    assert os.environ["AIBH_TEST_B"] == "already-set"
    assert "AIBH_TEST_EMPTY" not in os.environ
    assert source_of("AIBH_TEST_A") == "file"
    assert source_of("AIBH_TEST_B") == "environment"
    assert source_of("AIBH_TEST_EMPTY") is None


def test_reload_picks_up_a_changed_file(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("AIBH_TEST_A=old\n", encoding="utf-8")
    load_dotenv(env)
    env.write_text("AIBH_TEST_A=new\n", encoding="utf-8")
    load_dotenv(env)
    assert os.environ["AIBH_TEST_A"] == "new"


def test_set_env_value_replaces_or_appends(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    set_env_value(env, "AIBH_TEST_A", "one")  # creates the file
    env.write_text(env.read_text(encoding="utf-8") + "# keep me\nOTHER=x\n", encoding="utf-8")
    set_env_value(env, "AIBH_TEST_A", "two")
    set_env_value(env, "AIBH_TEST_B", "three")
    assert env.read_text(encoding="utf-8").splitlines() == [
        "AIBH_TEST_A=two",
        "# keep me",
        "OTHER=x",
        "AIBH_TEST_B=three",
    ]
    with pytest.raises(ValueError):
        set_env_value(env, "AIBH_TEST_A", "bad\nvalue")
