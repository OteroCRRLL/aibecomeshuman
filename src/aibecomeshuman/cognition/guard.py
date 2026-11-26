"""Anti-metagaming guard: checks that what the agent writes comes from what it lived.

Two layers:
- `Lexicon` (free, instant): story names and player vocabulary from `config/guard.toml`
  that appear in the agent's text without having appeared in anything it perceived,
  remembers or reads in its own character sheet.
- `Judge` (cheap model, role "judge"): reads the scene as the agent lived it and says
  which of the agent's statements rely on something that is not there. It audits every
  stance and decision at the end of a scene (measurement, no delay during play) and
  filters personality changes before they are applied (enforcement).
"""

from __future__ import annotations

import re
import tomllib
import unicodedata
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from aibecomeshuman.config import ConfigError
from aibecomeshuman.llm.base import LLMProvider
from aibecomeshuman.llm.types import LLMRequest, Message


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


class Lexicon(BaseModel):
    story: list[str] = Field(default_factory=list)
    meta: list[str] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> Lexicon:
        if not path.is_file():
            return cls()
        try:
            with path.open("rb") as f:
                return cls.model_validate(tomllib.load(f))
        except (tomllib.TOMLDecodeError, ValueError) as e:
            raise ConfigError(f"{path}: {e}") from e

    def unexplained(self, output: str, known: str) -> list[str]:
        """Terms used in `output` that never appear in `known` (whole words, any case)."""
        out, seen = _fold(output), _fold(known)
        found = []
        for term in (*self.story, *self.meta):
            pattern = r"(?<!\w)" + re.escape(_fold(term)) + r"(?!\w)"
            if re.search(pattern, out) and not re.search(pattern, seen):
                found.append(term)
        return found


class Statement(BaseModel):
    id: str  # e.g. "stance-3", "decision-c2", "trait-empathy"
    kind: str
    text: str


class Verdict(BaseModel):
    id: str
    grounded: bool
    unsupported: list[str] = Field(default_factory=list)


# Claim statuses: only the last two make a statement ungrounded.
GROUNDED = ("in_record", "interpretation")
UNGROUNDED = ("not_in_record", "contradicts_record")

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "statements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "claim": {"type": "string"},
                                "status": {"type": "string", "enum": [*GROUNDED, *UNGROUNDED]},
                            },
                            "required": ["claim", "status"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["id", "claims"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["statements"],
    "additionalProperties": False,
}

JUDGE_TASK = """For each statement, list the factual claims it makes (events, names,
places, relationships, what someone said or did) and give each one a status:
- in_record: the record shows it (paraphrases count).
- interpretation: a reading, feeling, guess, generalisation or plan built on things that
  ARE in the record. Always allowed, even if you would read it differently.
- not_in_record: an event, name or place the record never mentions, or knowledge of what
  happens later.
- contradicts_record: the record shows something different (e.g. "I lied" when the
  record shows the character told the truth).
Skip pure feelings and intentions; they are not claims."""


class Judge:
    def __init__(self, provider: LLMProvider, system: str) -> None:
        self.provider = provider
        self.system = system

    def review(self, record: str, statements: list[Statement]) -> list[Verdict]:
        if not statements:
            return []
        listing = "\n".join(f'<statement id="{s.id}" kind="{s.kind}">{s.text}</statement>'
                            for s in statements)
        request = LLMRequest(
            purpose="guard.judge",
            system=self.system,
            messages=[
                Message.user(
                    f"<record>\n{record}\n</record>\n\n<statements>\n{listing}\n</statements>"
                    f"\n\n<task>{JUDGE_TASK}</task>"
                )
            ],
            json_schema=JUDGE_SCHEMA,
            max_tokens=3000,
        )
        parsed = self.provider.complete(request).parsed or {}
        known = {s.id for s in statements}
        verdicts = []
        for row in parsed.get("statements", []):
            if row.get("id") not in known:
                continue
            bad = [
                f"{c['claim']} ({c['status'].replace('_', ' ')})"
                for c in row.get("claims", [])
                if c.get("status") in UNGROUNDED
            ]
            verdicts.append(Verdict(id=row["id"], grounded=not bad, unsupported=bad))
        return verdicts
