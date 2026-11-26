from __future__ import annotations

from pathlib import Path

import pytest

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
    assert "status: v1" not in system  # front matter is stripped


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
