"""Scenario environment: scripted scenes described in TOML, played without the game.

A scenario stands in for what perception will produce from the real game:
observations (written by the observer, in English) and dialogue / options (the game's
on-screen text, in Spanish). See scenarios/README.md for the format.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from aibecomeshuman.cognition.decider import Choice, Option
from aibecomeshuman.config import ConfigError


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ObservationStep(_Strict):
    kind: Literal["observation"]
    text: str
    # Important moments trigger a stance update before play continues.
    significant: bool = False


class DialogueStep(_Strict):
    kind: Literal["dialogue"]
    speaker: str
    text: str
    significant: bool = False


class ScenarioOption(_Strict):
    id: str
    label: str
    outcome: str
    default: bool = False


class ChoiceStep(_Strict):
    kind: Literal["choice"]
    id: str
    prompt: str = ""
    timed: bool = False
    seconds: float | None = Field(default=None, gt=0)
    options: list[ScenarioOption] = Field(min_length=2)

    @model_validator(mode="after")
    def _check(self) -> ChoiceStep:
        ids = [o.id for o in self.options]
        if len(ids) != len(set(ids)):
            raise ValueError(f"choice '{self.id}': option ids must be unique")
        if sum(o.default for o in self.options) > 1:
            raise ValueError(f"choice '{self.id}': at most one option can be the default")
        if self.timed and self.seconds is None:
            raise ValueError(f"choice '{self.id}': timed choices need 'seconds'")
        return self

    def to_choice(self) -> Choice:
        return Choice(
            id=self.id,
            prompt=self.prompt,
            timed=self.timed,
            seconds=self.seconds,
            options=[Option(id=o.id, label=o.label, default=o.default) for o in self.options],
        )

    def outcome_of(self, option_id: str) -> str:
        return next(o.outcome for o in self.options if o.id == option_id)


Step = Annotated[ObservationStep | DialogueStep | ChoiceStep, Field(discriminator="kind")]


class Scenario(_Strict):
    id: str
    title: str
    character: str
    location: str = ""
    synthetic: bool = Field(
        default=True, description="True for invented scenes (not taken from the game)"
    )
    steps: list[Step] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_choices(self) -> Scenario:
        ids = [s.id for s in self.steps if isinstance(s, ChoiceStep)]
        if len(ids) != len(set(ids)):
            raise ValueError("choice ids must be unique within a scenario")
        return self


def load_scenario(path: Path) -> Scenario:
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        raise ConfigError(f"scenario not found: {path}") from None
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: invalid TOML: {e}") from e
    try:
        return Scenario.model_validate(data)
    except ValidationError as e:
        raise ConfigError(f"{path}:\n{e}") from e
