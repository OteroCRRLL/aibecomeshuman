from __future__ import annotations

import tomllib

from aibecomeshuman.tomlpatch import set_value

TEXT = """# header comment
top = 1

[budget]
# caps
project_cap_usd = 10.0   # raise later
run_cap_usd = 0.5

[roles]
decider = "fake"
"""


def test_replaces_value_and_keeps_comments() -> None:
    out = set_value(TEXT, "budget", "project_cap_usd", 25.0)
    assert "project_cap_usd = 25.0   # raise later" in out
    assert "# header comment" in out and "# caps" in out
    assert tomllib.loads(out)["budget"]["project_cap_usd"] == 25.0


def test_only_touches_the_right_section() -> None:
    out = set_value(TEXT, "roles", "decider", "claude-haiku")
    data = tomllib.loads(out)
    assert data["roles"]["decider"] == "claude-haiku"
    assert data["budget"]["run_cap_usd"] == 0.5


def test_inserts_missing_key_inside_section() -> None:
    out = set_value(TEXT, "budget", "warn_ratio", 0.8)
    data = tomllib.loads(out)
    assert data["budget"]["warn_ratio"] == 0.8
    assert "warn_ratio" not in data["roles"]


def test_creates_missing_section_and_top_level_keys() -> None:
    out = set_value(TEXT, "timing", "execution_margin_s", 1.5)
    assert tomllib.loads(out)["timing"]["execution_margin_s"] == 1.5
    out = set_value(TEXT, None, "top", 2)
    assert tomllib.loads(out)["top"] == 2


def test_string_values_are_quoted() -> None:
    out = set_value(TEXT, "roles", "observer", 'quote "this"')
    assert tomllib.loads(out)["roles"]["observer"] == 'quote "this"'
