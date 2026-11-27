from __future__ import annotations

from pathlib import Path

import pytest

from aibecomeshuman.cognition.context import SceneContext as _SceneContext
from aibecomeshuman.config import ConfigError
from aibecomeshuman.prompts import PromptLibrary, split_front_matter

from ..conftest import REPO_ROOT


def test_split_front_matter_with_comments() -> None:
    meta, body = split_front_matter("---\npriority: order   # note\nscope: all\n---\n\nHello\n")
    assert meta == {"priority": "order", "scope": "all"}
    assert body == "Hello"


def test_no_front_matter_returns_body() -> None:
    assert split_front_matter("  just text \n") == ({}, "just text")


def test_repo_directives_parse() -> None:
    lib = PromptLibrary(REPO_ROOT / "prompts")
    default = lib.directive("default")
    assert default.priority == "flavor" and default.shapes_personality is True
    bad = lib.directive("bad_ideas")
    assert bad.priority == "order" and bad.shapes_personality is False
    assert "worst idea" in bad.text


def test_system_prompt_layers_in_order() -> None:
    lib = PromptLibrary(REPO_ROOT / "prompts")
    system = lib.system_prompt("decider", "kara", lib.directive("bad_ideas"), "Spanish")
    order = ["<constitution>", "<role>", "<character>", "<directive", "<output_language>"]
    positions = [system.index(tag) for tag in order]
    assert positions == sorted(positions)
    assert "OVERRIDES" in system and "in Spanish" in system
    assert "trait.obedience" not in system  # front matter is stripped


def test_personality_goes_between_character_and_directive() -> None:
    lib = PromptLibrary(REPO_ROOT / "prompts")
    system = lib.system_prompt(
        "decider", "kara", lib.directive("default"), "Spanish", "<personality>X</personality>"
    )
    assert system.index("<character>") < system.index("<personality>") < system.index("<directive")


@pytest.mark.parametrize("name", ["connor", "kara", "markus"])
def test_character_traits_parse(name: str) -> None:
    traits = PromptLibrary(REPO_ROOT / "prompts").character_traits(name)
    assert "obedience" in traits and "empathy" in traits
    value, meaning = traits["obedience"]
    assert 0 <= value <= 10 and meaning


@pytest.mark.parametrize(
    ("header", "message"),
    [
        ("trait.x: high", "needs a number"),
        ("trait.x: 11", "between 0 and 10"),
        ("status: v1", "no 'trait"),
    ],
)
def test_bad_traits_are_rejected(tmp_path: Path, header: str, message: str) -> None:
    (tmp_path / "characters").mkdir()
    (tmp_path / "characters" / "x.md").write_text(f"---\n{header}\n---\nbody", encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        PromptLibrary(tmp_path).character_traits("x")


def test_missing_prompt_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="missing prompt file"):
        PromptLibrary(tmp_path).character("nobody")


def test_invalid_directive_priority(tmp_path: Path) -> None:
    (tmp_path / "directives").mkdir()
    (tmp_path / "directives" / "x.md").write_text(
        "---\npriority: maybe\n---\ntext", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="priority"):
        PromptLibrary(tmp_path).directive("x")


def test_briefings_are_valid_and_always_in_context() -> None:
    library = PromptLibrary(REPO_ROOT / "prompts")
    briefings = library.briefings()
    assert briefings and briefings[0].chapter == 1 and briefings[0].character == "connor"
    assert library.briefing(999) is None
    ctx = _SceneContext("s", briefing="You are the negotiator.")
    for i in range(30):
        ctx.add("observation", f"thing {i}")
    assert "<briefing>You are the negotiator.</briefing>" in ctx.render(last_n=5)
