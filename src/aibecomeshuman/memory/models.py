"""Long-term memory of one character, as data.

    episodic memory   one summary per scene lived (M1, M2...), with an importance 1-5
    entities          people, places and things known, with facts, a feeling and trust
    commitments       promises made (C1, C2...): active, kept, broken or dropped
    personality       traits (0-10) that drift with experience, plus a self-description,
                      with the full history of every change and its reason

Everything the agent "knows" between scenes lives here. It only ever contains what this
character lived itself: knowledge is never shared between characters.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, Field

EntityKind = Literal["person", "place", "object", "group", "other"]
CommitmentStatus = Literal["active", "kept", "broken", "dropped"]


def entity_key(name: str) -> str:
    """Normalized key so 'Mujer mayor' and 'mujer  mayor' are the same entity."""
    decomposed = unicodedata.normalize("NFKD", name.lower())
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", plain).strip()


class Trait(BaseModel):
    value: float = Field(ge=0, le=10)
    meaning: str = ""


class TraitChange(BaseModel):
    trait: str
    old: float
    new: float
    reason: str
    scene_id: str
    version: int


class Personality(BaseModel):
    traits: dict[str, Trait]
    initial: dict[str, float] = Field(default_factory=dict)
    # Empty until the first reflection: the character sheet's self-description applies.
    self_description: str = ""
    version: int = 0
    history: list[TraitChange] = Field(default_factory=list)

    @classmethod
    def from_traits(cls, traits: dict[str, Trait]) -> Personality:
        return cls(traits=traits, initial={k: t.value for k, t in traits.items()})


class MemoryItem(BaseModel):
    id: str
    scene_id: str
    scene_title: str
    text: str
    importance: int = Field(default=3, ge=1, le=5)
    entities: list[str] = Field(default_factory=list)
    run_id: str | None = None


class Entity(BaseModel):
    name: str
    kind: EntityKind = "other"
    facts: list[str] = Field(default_factory=list)
    feeling: str = ""
    trust: int = Field(default=0, ge=-5, le=5)
    first_scene: str
    last_scene: str


class Commitment(BaseModel):
    id: str
    text: str
    to: str = ""
    status: CommitmentStatus = "active"
    scene_id: str
    resolved_scene_id: str | None = None
    note: str = ""


class CharacterMemory(BaseModel):
    character: str
    items: list[MemoryItem] = Field(default_factory=list)
    entities: dict[str, Entity] = Field(default_factory=dict)
    commitments: list[Commitment] = Field(default_factory=list)
    personality: Personality
    scenes_played: list[str] = Field(default_factory=list)

    @property
    def ids(self) -> set[str]:
        return {i.id for i in self.items} | {c.id for c in self.commitments}

    @property
    def active_commitments(self) -> list[Commitment]:
        return [c for c in self.commitments if c.status == "active"]

    def next_item_id(self) -> str:
        return f"M{len(self.items) + 1}"

    def next_commitment_id(self) -> str:
        return f"C{len(self.commitments) + 1}"
