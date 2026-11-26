"""Run manifest: everything needed to know what a run was played with and to rebuild it."""

from __future__ import annotations

import hashlib
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from aibecomeshuman import __version__
from aibecomeshuman.config import ModelProfile, RunProfile
from aibecomeshuman.llm.types import Usage

RunStatus = Literal["running", "ok", "error", "aborted"]


class RoleUsage(BaseModel):
    calls: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0
    usage: Usage = Field(default_factory=Usage)


class RunManifest(BaseModel):
    run_id: str
    status: RunStatus = "running"
    started_at: datetime
    ended_at: datetime | None = None
    profile: RunProfile
    models: dict[str, ModelProfile]
    prompts: dict[str, str] = Field(description="relative path -> sha256 of each prompt used")
    package_version: str = __version__
    python_version: str = Field(default_factory=lambda: sys.version.split()[0])
    platform: str = Field(default_factory=platform.platform)
    usage_by_role: dict[str, RoleUsage] = Field(default_factory=dict)
    error: str | None = None

    def write(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)


def hash_prompts(prompts_dir: Path) -> dict[str, str]:
    if not prompts_dir.is_dir():
        return {}
    hashes: dict[str, str] = {}
    for file in sorted(p for p in prompts_dir.rglob("*") if p.is_file()):
        rel = file.relative_to(prompts_dir).as_posix()
        hashes[rel] = hashlib.sha256(file.read_bytes()).hexdigest()
    return hashes
