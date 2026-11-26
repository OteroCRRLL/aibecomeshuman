"""Persistence of memory: one JSON file per character, plus a campaign-wide state.

    campaigns/<campaign>/
        state.json         choices made so far (for scenes whose steps depend on them)
        <character>.json   that character's CharacterMemory

Without a campaign, the store lives inside the run folder and starts empty.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from aibecomeshuman.config import ConfigError
from aibecomeshuman.memory.models import CharacterMemory, Personality

CAMPAIGN_NAME = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def check_campaign_name(name: str) -> str:
    if not CAMPAIGN_NAME.match(name):
        raise ConfigError(
            f"invalid campaign name '{name}': use 1-40 letters, digits, '-' or '_'"
        )
    return name


class CampaignState(BaseModel):
    # "<scenario id>.<choice id>" -> option id chosen
    choices: dict[str, str] = Field(default_factory=dict)


def _write(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(model.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)


class MemoryStore:
    def __init__(self, directory: Path) -> None:
        self.dir = directory

    def load(
        self, character: str, initial_personality: Callable[[], Personality]
    ) -> CharacterMemory:
        saved = self.read(character)
        if saved is not None:
            return saved
        return CharacterMemory(character=character, personality=initial_personality())

    def read(self, character: str) -> CharacterMemory | None:
        """The saved memory of a character, or None if it has none yet."""
        path = self.dir / f"{character}.json"
        if not path.is_file():
            return None
        return CharacterMemory.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, memory: CharacterMemory) -> None:
        _write(self.dir / f"{memory.character}.json", memory)

    def load_state(self) -> CampaignState:
        path = self.dir / "state.json"
        if path.is_file():
            return CampaignState.model_validate_json(path.read_text(encoding="utf-8"))
        return CampaignState()

    def save_state(self, state: CampaignState) -> None:
        _write(self.dir / "state.json", state)

    def characters(self) -> list[str]:
        if not self.dir.is_dir():
            return []
        return sorted(p.stem for p in self.dir.glob("*.json") if p.stem != "state")
