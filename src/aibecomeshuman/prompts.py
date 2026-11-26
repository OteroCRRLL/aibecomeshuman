"""Prompt library: reads the files in prompts/ and assembles the layered system prompt.

Layer order (most fixed first, so the shared prefix stays stable and cache-friendly):
constitution -> role -> character -> directive -> output language.
"""

from __future__ import annotations

import re
from html import escape
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from aibecomeshuman.config import ConfigError

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.S)


def split_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Split a `---`-delimited `key: value` header from the body. `#` starts a comment."""
    match = _FRONT_MATTER.match(text)
    if not match:
        return {}, text.strip()
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        line = line.split("#", 1)[0].strip()
        key, sep, value = line.partition(":")
        if sep and key.strip():
            meta[key.strip()] = value.strip()
    return meta, text[match.end() :].strip()


class Directive(BaseModel):
    name: str
    text: str
    scope: str = "all characters"
    # flavor = colors the personality · order = overrides it when they conflict
    priority: Literal["flavor", "order"] = "flavor"
    shapes_personality: bool = True


class PromptLibrary:
    def __init__(self, prompts_dir: Path) -> None:
        self.dir = prompts_dir

    def _read(self, *parts: str) -> tuple[dict[str, str], str]:
        path = self.dir.joinpath(*parts)
        if not path.is_file():
            raise ConfigError(f"missing prompt file: {path}")
        return split_front_matter(path.read_text(encoding="utf-8"))

    def constitution(self) -> str:
        return self._read("constitution", "base.md")[1]

    def role(self, name: str) -> str:
        return self._read("roles", f"{name}.md")[1]

    def character(self, name: str) -> str:
        return self._read("characters", f"{name}.md")[1]

    def directive(self, name: str) -> Directive:
        meta, body = self._read("directives", f"{name}.md")
        priority = meta.get("priority", "flavor")
        if priority not in ("flavor", "order"):
            raise ConfigError(f"directive '{name}': priority must be 'flavor' or 'order'")
        return Directive(
            name=name,
            text=body,
            scope=meta.get("scope", "all characters"),
            priority=priority,  # type: ignore[arg-type]
            shapes_personality=meta.get("shapes_personality", "true").lower() == "true",
        )

    def system_prompt(
        self,
        role: str,
        character: str,
        directive: Directive,
        output_language: str,
    ) -> str:
        priority_note = (
            "This directive OVERRIDES your personality when they conflict."
            if directive.priority == "order"
            else "This directive colors how you act; your personality still leads."
        )
        sections = [
            f"<constitution>\n{self.constitution()}\n</constitution>",
            f"<role>\n{self.role(role)}\n</role>",
            f"<character>\n{self.character(character)}\n</character>",
            f'<directive priority="{escape(directive.priority)}">\n'
            f"{directive.text}\n\n{priority_note}\n</directive>",
            "<output_language>\n"
            f"Write every free-text field of your answers in {output_language}. "
            "Option ids and event ids stay exactly as given.\n"
            "</output_language>",
        ]
        return "\n\n".join(sections)
