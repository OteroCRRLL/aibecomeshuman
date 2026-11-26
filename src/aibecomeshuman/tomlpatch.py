"""Minimal in-place TOML editing that keeps comments and layout.

Only supports what the UI needs: setting a scalar `key = value` inside a `[section]`
(or at the top level with `section=None`). Values are numbers, booleans or plain strings.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_HEADER = re.compile(r"^\s*\[")


def format_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    raise TypeError(f"unsupported TOML value: {value!r}")


def set_value(text: str, section: str | None, key: str, value: object) -> str:
    lines = text.splitlines()
    literal = format_value(value)
    key_re = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*)(.*?)(\s+#.*)?$")

    if section is None:
        start, end = 0, next((i for i, ln in enumerate(lines) if _HEADER.match(ln)), len(lines))
    else:
        header = f"[{section}]"
        try:
            start = next(i for i, ln in enumerate(lines) if ln.strip() == header) + 1
        except StopIteration:
            tail = "\n" if text and not text.endswith("\n") else ""
            return f"{text}{tail}\n{header}\n{key} = {literal}\n"
        end = next((i for i in range(start, len(lines)) if _HEADER.match(lines[i])), len(lines))

    for i in range(start, end):
        match = key_re.match(lines[i])
        if match:
            lines[i] = f"{match.group(1)}{literal}{match.group(3) or ''}"
            return "\n".join(lines) + "\n"

    insert_at = end
    while insert_at > start and not lines[insert_at - 1].strip():
        insert_at -= 1
    lines.insert(insert_at, f"{key} = {literal}")
    return "\n".join(lines) + "\n"


def patch_file(path: Path, section: str | None, updates: dict[str, object]) -> str:
    """Apply updates to a TOML file; returns the previous text so callers can roll back."""
    original = path.read_text(encoding="utf-8")
    text = original
    for key, value in updates.items():
        text = set_value(text, section, key, value)
    path.write_text(text, encoding="utf-8")
    return original
