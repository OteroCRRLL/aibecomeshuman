"""Carga y validación de la configuración del proyecto.

Estructura en disco (relativa a la raíz del proyecto):

    config/default.toml          ajustes globales
    config/models/<nombre>.toml  un perfil de modelo por fichero
    config/runs/<nombre>.toml    un perfil de run por fichero
    prompts/characters/<x>.md    fichas de personaje referenciadas por los runs
    prompts/directives/<x>.md    directivas referenciadas por los runs

Todo se valida al cargar: un run que apunte a un modelo, personaje o directiva
inexistente falla antes de empezar a jugar, no a mitad de partida.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

# Roles técnicos que puede pedir el sistema. Cada run asigna un perfil de modelo a cada uno.
ROLES: tuple[str, ...] = (
    "observer",  # VLM barato: convierte fotogramas clave en observaciones
    "decider",  # decisiones críticas sin temporizador + mantener la postura de escena
    "decider_fast",  # elecciones con temporizador: postura + opciones -> opción
    "reflector",  # evolución de personalidad al cerrar escenas/capítulos
    "judge",  # detector de metagaming
    "summarizer",  # consolidación de memoria
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


class LLMSettings(_Strict):
    record_calls: bool = True


class AppConfig(_Strict):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    llm: LLMSettings = Field(default_factory=LLMSettings)


class ModelProfile(_Strict):
    name: str
    provider: Provider
    model: str
    max_tokens: int = Field(default=16000, gt=0)
    effort: Effort | None = None
    timeout_s: float = Field(default=120.0, gt=0)
    supports_vision: bool = False
    # Reintento automático en otro modelo si el principal rechaza la petición
    # (solo proveedores/modelos que lo soportan; ver docs/architecture.md).
    refusal_fallback: bool = False
    # Opciones propias del proveedor (p. ej. respuestas guionizadas del proveedor falso).
    options: dict[str, Any] = Field(default_factory=dict)


class RunProfile(_Strict):
    name: str
    description: str = ""
    character: str
    directive: str = "default"
    roles: dict[str, str]

    @field_validator("roles")
    @classmethod
    def _known_roles(cls, roles: dict[str, str]) -> dict[str, str]:
        unknown = sorted(set(roles) - set(ROLES))
        if unknown:
            raise ValueError(f"roles desconocidos: {unknown}. Válidos: {list(ROLES)}")
        if "decider" not in roles:
            raise ValueError("todo run necesita al menos el rol 'decider'")
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

    def run_profile(self, name: str) -> RunProfile:
        try:
            return self.runs[name]
        except KeyError:
            raise ConfigError(
                f"perfil de run '{name}' no existe. Disponibles: {sorted(self.runs)}"
            ) from None

    def models_for(self, run: RunProfile) -> dict[str, ModelProfile]:
        return {role: self.models[model_name] for role, model_name in run.roles.items()}


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: TOML inválido: {e}") from e


def _load_named(directory: Path, model: type[BaseModel]) -> dict[str, Any]:
    items: dict[str, Any] = {}
    if not directory.is_dir():
        return items
    for file in sorted(directory.glob("*.toml")):
        data = _read_toml(file)
        data.setdefault("name", file.stem)
        if data["name"] != file.stem:
            raise ConfigError(f"{file}: 'name' ({data['name']}) debe coincidir con el fichero")
        try:
            items[file.stem] = model.model_validate(data)
        except ValidationError as e:
            raise ConfigError(f"{file}:\n{e}") from e
    return items


def find_project_root(start: Path | None = None) -> Path:
    """Sube desde `start` hasta encontrar `config/default.toml`."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "config" / "default.toml").is_file():
            return candidate
    raise ConfigError(f"no se encontró config/default.toml desde {current} hacia arriba")


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
                    f"run '{run.name}': rol '{role}' apunta al modelo '{model_name}', "
                    f"que no existe en config/models/"
                )
        if not (prompts / "characters" / f"{run.character}.md").is_file():
            problems.append(f"run '{run.name}': falta prompts/characters/{run.character}.md")
        if not (prompts / "directives" / f"{run.directive}.md").is_file():
            problems.append(f"run '{run.name}': falta prompts/directives/{run.directive}.md")
    if problems:
        raise ConfigError("configuración inconsistente:\n  - " + "\n  - ".join(problems))
