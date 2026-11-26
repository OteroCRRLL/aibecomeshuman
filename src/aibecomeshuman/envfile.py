"""Reading and writing the project's .env file (API keys).

Variables set in the real environment always win over the file. Variables that came
from the file are refreshed on every load, so a key changed in .env (by hand or from the
control panel) takes effect without restarting anything.
"""

from __future__ import annotations

import os
from pathlib import Path

_FROM_FILE: dict[str, str] = {}


def _parse(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value:
            values[key] = value
    return values


def load_dotenv(path: Path) -> None:
    """Load `KEY=VALUE` lines into the environment, without overriding real variables."""
    for key, value in _parse(path).items():
        current = os.environ.get(key)
        if current is not None and _FROM_FILE.get(key) != current:
            continue  # set outside the .env file: the real environment wins
        os.environ[key] = value
        _FROM_FILE[key] = value


def source_of(key: str) -> str | None:
    """'file' if the variable came from .env, 'environment' if set elsewhere, else None."""
    value = os.environ.get(key)
    if not value:
        return None
    return "file" if _FROM_FILE.get(key) == value else "environment"


def set_env_value(path: Path, key: str, value: str) -> None:
    """Create or replace `KEY=value` in the .env file, keeping every other line."""
    if "\n" in value or "\r" in value:
        raise ValueError("value must be a single line")
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    new_line = f"{key}={value}"
    for i, line in enumerate(lines):
        if line.strip().split("=", 1)[0].strip() == key:
            lines[i] = new_line
            break
    else:
        lines.append(new_line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
