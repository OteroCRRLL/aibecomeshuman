from __future__ import annotations

from pathlib import Path

import pytest

from aibecomeshuman.config import ConfigError
from aibecomeshuman.env.scenario import ChoiceStep, load_scenario

from ..conftest import REPO_ROOT

SCENARIOS = sorted((REPO_ROOT / "scenarios").glob("*.toml"))


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_repo_scenarios_are_valid(path: Path) -> None:
    scenario = load_scenario(path)
    assert scenario.synthetic is True
    assert (REPO_ROOT / "prompts" / "characters" / f"{scenario.character}.md").is_file()
    assert any(isinstance(s, ChoiceStep) for s in scenario.steps)


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "s.toml"
    path.write_text('id = "s"\ntitle = "t"\ncharacter = "kara"\n' + body, encoding="utf-8")
    return path


CHOICE = """
[[steps]]
kind = "choice"
id = "c"
{extra}
  [[steps.options]]
  id = "A"
  label = "Uno"
  outcome = "o1"
  {a_extra}
  [[steps.options]]
  id = "{b_id}"
  label = "Dos"
  outcome = "o2"
  {b_extra}
"""


def choice(extra="", a_extra="", b_id="B", b_extra="") -> str:
    return CHOICE.format(extra=extra, a_extra=a_extra, b_id=b_id, b_extra=b_extra)


def test_minimal_choice_loads(tmp_path: Path) -> None:
    step = load_scenario(write(tmp_path, choice())).steps[0]
    assert isinstance(step, ChoiceStep)
    assert step.to_choice().option_ids == ["A", "B"]
    assert step.outcome_of("B") == "o2"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (choice(b_id="A"), "unique"),
        (choice(a_extra="default = true", b_extra="default = true"), "default"),
        (choice(extra="timed = true"), "seconds"),
        (choice() + choice(), "unique within"),
    ],
)
def test_invalid_choices(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_scenario(write(tmp_path, body))


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_scenario(tmp_path / "nope.toml")
