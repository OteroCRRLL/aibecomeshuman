"""Loading and validation of the project configuration.

On-disk layout (relative to the project root):

    config/default.toml          global settings
    config/models/<name>.toml    one model profile per file
    config/runs/<name>.toml      one run profile per file
    prompts/characters/<x>.md    character sheets referenced by runs
    prompts/directives/<x>.md    directives referenced by runs

Everything is validated on load: a run pointing to a missing model, character or
directive fails before play starts, not halfway through a session.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

# Technical roles the system can request. Each run assigns a model profile to each one.
ROLES: tuple[str, ...] = (
    "observer",  # cheap VLM: turns keyframes into observations
    "observer_strong",  # better eyes for key action moments only (capped per chapter)
    "decider",  # critical untimed decisions + maintaining the scene stance
    "decider_fast",  # timed choices: stance + options -> option
    "reflector",  # personality evolution when closing scenes/chapters
    "judge",  # metagaming detector
    "summarizer",  # memory consolidation
)

Provider = Literal["fake", "anthropic"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]


class ConfigError(Exception):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Strict):
    runs_dir: Path = Path("runs")
    prompts_dir: Path = Path("prompts")
    # One folder per playthrough: memory and personality persist across sessions.
    campaigns_dir: Path = Path("campaigns")


class LLMSettings(_Strict):
    record_calls: bool = True


class BudgetSettings(_Strict):
    # Hard caps in USD. Paid calls are refused once a cap is reached.
    project_cap_usd: float = Field(default=10.0, ge=0)
    run_cap_usd: float = Field(default=0.5, ge=0)
    # Emit a `budget.warning` event when this fraction of a cap is spent.
    warn_ratio: float = Field(default=0.8, gt=0, le=1)


class TimingSettings(_Strict):
    # Seconds reserved, out of a timed choice's countdown, to actually press the button.
    execution_margin_s: float = Field(default=1.0, ge=0)


class MemorySettings(_Strict):
    # Largest change a single reflection can make to one trait (traits go from 0 to 10).
    max_trait_step: float = Field(default=1.0, gt=0, le=10)
    # How much memory goes into each decision prompt.
    recent_scenes: int = Field(default=5, ge=0)
    important_scenes: int = Field(default=5, ge=0)
    max_entities: int = Field(default=6, ge=0)
    max_facts_per_entity: int = Field(default=6, ge=1)


class PerceptionSettings(_Strict):
    # Recordings the panel lists (relative to the project root).
    recordings_dir: Path = Path("data/recordings")
    # One frame is checked every `sample_every_s`; only frames that changed go to the model.
    sample_every_s: float = Field(default=1.0, gt=0)
    # Send a frame anyway if nothing was sent for this long (catches slow changes).
    heartbeat_s: float = Field(default=6.0, gt=0)
    # Fraction of the small thumbnail that must change to count as a new shot. The camera
    # moves almost all the time, so only big changes count; subtitles have their own check.
    change_ratio: float = Field(default=0.4, gt=0, le=1)
    # Frames darker than this mean brightness (0-255) are skipped (fades, loading).
    dark_threshold: float = Field(default=12.0, ge=0)
    # Width of the frames sent to the model. Menus keep more detail (small labels).
    frame_width: int = Field(default=1280, ge=320)
    menu_frame_width: int = Field(default=1568, ge=320)
    # Parallel model calls (each worker reads a contiguous stretch of the video).
    workers: int = Field(default=4, ge=1, le=16)
    # On-screen text matching these (case-insensitive regex) never reaches the agent.
    hidden_patterns: list[str] = Field(default_factory=list)


class LiveSettings(_Strict):
    # OBS WebSocket (OBS 28+: Tools -> WebSocket Server Settings). The password goes in .env
    # as OBS_WEBSOCKET_PASSWORD, never here.
    obs_host: str = "localhost"
    obs_port: int = Field(default=4455, gt=0, lt=65536)
    # Name of the game capture source in OBS, exactly as shown in the Sources list.
    obs_source: str = "Detroit"
    # Frames checked per second (local, free); only changes reach the observer model.
    frames_per_s: float = Field(default=2.0, gt=0, le=10)
    # Observer calls in flight at once (menus always get their own, right away).
    workers: int = Field(default=3, ge=1, le=8)
    # The game's countdown is not readable; assume this much time for a timed choice.
    timed_choice_s: float = Field(default=6.0, gt=0)
    # Update the stance after this many new things perceived (lines, notices).
    stance_every: int = Field(default=12, ge=1)
    # Key action moments (fast changes, no dialogue, something dramatic) are read again by
    # the `observer_strong` model, up to this much spend per game chapter.
    strong_cap_per_chapter_usd: float = Field(default=0.15, ge=0)


class AppConfig(_Strict):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    budget: BudgetSettings = Field(default_factory=BudgetSettings)
    timing: TimingSettings = Field(default_factory=TimingSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    perception: PerceptionSettings = Field(default_factory=PerceptionSettings)
    live: LiveSettings = Field(default_factory=LiveSettings)


class ModelProfile(_Strict):
    name: str
    provider: Provider
    model: str
    max_tokens: int = Field(default=16000, gt=0)
    effort: Effort | None = None
    timeout_s: float = Field(default=120.0, gt=0)
    supports_vision: bool = False
    # Automatic retry on another model if the primary one refuses the request
    # (only providers/models that support it; see docs/architecture.md).
    refusal_fallback: bool = False
    # USD per million tokens, used for cost tracking and budget caps.
    price_input_per_mtok: float = Field(default=0.0, ge=0)
    price_output_per_mtok: float = Field(default=0.0, ge=0)
    price_cache_read_per_mtok: float = Field(default=0.0, ge=0)
    # Locked models (the expensive ones) can only be used with an explicit unlock.
    locked: bool = False
    # Provider-specific options (e.g. scripted responses for the fake provider).
    options: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_paid(self) -> bool:
        return any(
            (self.price_input_per_mtok, self.price_output_per_mtok, self.price_cache_read_per_mtok)
        )


class RunProfile(_Strict):
    name: str
    description: str = ""
    characters: list[str] = Field(min_length=1)
    directive: str = "default"
    # Language for every free-text field the agent writes (reasoning, stance...).
    output_language: str = "Spanish"
    roles: dict[str, str]

    @field_validator("roles")
    @classmethod
    def _known_roles(cls, roles: dict[str, str]) -> dict[str, str]:
        unknown = sorted(set(roles) - set(ROLES))
        if unknown:
            raise ValueError(f"unknown roles: {unknown}. Valid: {list(ROLES)}")
        if "decider" not in roles:
            raise ValueError("every run needs at least the 'decider' role")
        return roles


class ProjectConfig(BaseModel):
    root: Path
    app: AppConfig
    models: dict[str, ModelProfile]
    runs: dict[str, RunProfile]

    def path(self, relative: Path) -> Path:
        return relative if relative.is_absolute() else self.root / relative

    @property
    def runs_dir(self) -> Path:
        return self.path(self.app.paths.runs_dir)

    @property
    def prompts_dir(self) -> Path:
        return self.path(self.app.paths.prompts_dir)

    @property
    def campaigns_dir(self) -> Path:
        return self.path(self.app.paths.campaigns_dir)

    @property
    def recordings_dir(self) -> Path:
        return self.path(self.app.perception.recordings_dir)

    def run_profile(self, name: str) -> RunProfile:
        try:
            return self.runs[name]
        except KeyError:
            raise ConfigError(
                f"run profile '{name}' does not exist. Available: {sorted(self.runs)}"
            ) from None

    def models_for(self, run: RunProfile) -> dict[str, ModelProfile]:
        return {role: self.models[model_name] for role, model_name in run.roles.items()}


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: invalid TOML: {e}") from e


def _load_named(directory: Path, model: type[BaseModel]) -> dict[str, Any]:
    items: dict[str, Any] = {}
    if not directory.is_dir():
        return items
    for file in sorted(directory.glob("*.toml")):
        data = _read_toml(file)
        data.setdefault("name", file.stem)
        if data["name"] != file.stem:
            raise ConfigError(f"{file}: 'name' ({data['name']}) must match the file name")
        try:
            items[file.stem] = model.model_validate(data)
        except ValidationError as e:
            raise ConfigError(f"{file}:\n{e}") from e
    return items


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from `start` until `config/default.toml` is found."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "config" / "default.toml").is_file():
            return candidate
    raise ConfigError(f"config/default.toml not found from {current} upwards")


def load_project(root: Path | None = None) -> ProjectConfig:
    root = find_project_root(root)
    config_dir = root / "config"

    try:
        app = AppConfig.model_validate(_read_toml(config_dir / "default.toml"))
    except ValidationError as e:
        raise ConfigError(f"{config_dir / 'default.toml'}:\n{e}") from e

    project = ProjectConfig(
        root=root,
        app=app,
        models=_load_named(config_dir / "models", ModelProfile),
        runs=_load_named(config_dir / "runs", RunProfile),
    )
    _check_references(project)
    return project


def _check_references(project: ProjectConfig) -> None:
    problems: list[str] = []
    prompts = project.prompts_dir
    for run in project.runs.values():
        for role, model_name in run.roles.items():
            if model_name not in project.models:
                problems.append(
                    f"run '{run.name}': role '{role}' points to model '{model_name}', "
                    f"which does not exist in config/models/"
                )
        for character in run.characters:
            if not (prompts / "characters" / f"{character}.md").is_file():
                problems.append(f"run '{run.name}': missing prompts/characters/{character}.md")
        if not (prompts / "directives" / f"{run.directive}.md").is_file():
            problems.append(f"run '{run.name}': missing prompts/directives/{run.directive}.md")
    if problems:
        raise ConfigError("inconsistent configuration:\n  - " + "\n  - ".join(problems))
