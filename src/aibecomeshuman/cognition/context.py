"""Scene context: what the character has perceived in the current scene, with stable ids.

Each entry gets an id (E1, E2...) so decisions can cite the evidence they rely on.
Phase 2 replaces this short-term buffer with layered memory; the ids stay.
"""

from __future__ import annotations

from html import escape
from typing import Literal

from pydantic import BaseModel

EntryKind = Literal["observation", "dialogue", "outcome", "decision"]


class ContextEntry(BaseModel):
    id: str
    kind: EntryKind
    text: str
    speaker: str | None = None


class SceneContext:
    def __init__(self, scene_id: str, location: str = "") -> None:
        self.scene_id = scene_id
        self.location = location
        self.entries: list[ContextEntry] = []

    def add(self, kind: EntryKind, text: str, speaker: str | None = None) -> ContextEntry:
        entry = ContextEntry(id=f"E{len(self.entries) + 1}", kind=kind, text=text, speaker=speaker)
        self.entries.append(entry)
        return entry

    @property
    def ids(self) -> set[str]:
        return {e.id for e in self.entries}

    def render(self, last_n: int | None = None) -> str:
        entries = self.entries if last_n is None else self.entries[-last_n:]
        lines = [f'<scene location="{escape(self.location)}">']
        for e in entries:
            speaker = f' speaker="{escape(e.speaker)}"' if e.speaker else ""
            lines.append(
                f'<event id="{e.id}" kind="{e.kind}"{speaker}>{escape(e.text, quote=False)}</event>'
            )
        lines.append("</scene>")
        return "\n".join(lines)
