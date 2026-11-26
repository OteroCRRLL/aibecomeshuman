"""Run session: creates the run folder, the event log, the manifest and the per-role
model providers (wrapped in recording when enabled).

    runs/<YYYYMMDD-HHMMSS>_<profile>/
        manifest.json     what the run was played with and what it cost
        events.jsonl      event log (source of truth)
        llm_calls.jsonl   every model call, replayable as a cassette
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from types import TracebackType

from aibecomeshuman.config import ModelProfile, ProjectConfig
from aibecomeshuman.core.clock import Clock, SystemClock
from aibecomeshuman.core.events import Event, EventLog
from aibecomeshuman.llm.base import LLMProvider
from aibecomeshuman.llm.recording import RecordingProvider
from aibecomeshuman.llm.registry import build_provider
from aibecomeshuman.run.manifest import RoleUsage, RunManifest, RunStatus, hash_prompts

ProviderBuilder = Callable[[ModelProfile, Clock | None], LLMProvider]


class RunSession:
    def __init__(
        self,
        project: ProjectConfig,
        profile_name: str,
        clock: Clock | None = None,
        provider_builder: ProviderBuilder = build_provider,
    ) -> None:
        self.project = project
        self.profile = project.run_profile(profile_name)
        self.clock = clock or SystemClock()
        self._provider_builder = provider_builder
        self._providers: dict[str, LLMProvider] = {}
        self.dir: Path | None = None
        self.events: EventLog | None = None
        self.manifest: RunManifest | None = None

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> RunSession:
        started = self.clock.now()
        run_id = self._unique_run_id(started.strftime("%Y%m%d-%H%M%S"))
        self.dir = self.project.runs_dir / run_id
        self.dir.mkdir(parents=True)

        models = self.project.models_for(self.profile)
        self.manifest = RunManifest(
            run_id=run_id,
            started_at=started,
            profile=self.profile,
            models=models,
            prompts=hash_prompts(self.project.prompts_dir),
        )
        self._save_manifest()

        self.events = EventLog(self.dir / "events.jsonl", clock=self.clock)
        self.events.subscribe("llm.call", self._account_llm_call)
        self.events.emit(
            "run.started",
            source="run",
            payload={
                "run_id": run_id,
                "profile": self.profile.name,
                "character": self.profile.character,
                "directive": self.profile.directive,
                "roles": dict(self.profile.roles),
            },
        )
        return self

    def finish(self, status: RunStatus = "ok", error: str | None = None) -> None:
        manifest, events = self._require_started()
        if manifest.status != "running":
            return
        events.emit("run.finished", source="run", payload={"status": status, "error": error})
        manifest.status = status
        manifest.error = error
        manifest.ended_at = self.clock.now()
        self._save_manifest()

    def __enter__(self) -> RunSession:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc is None:
            self.finish("ok")
        elif isinstance(exc, KeyboardInterrupt):
            self.finish("aborted", error="interrupted by the operator")
        else:
            self.finish("error", error=f"{type(exc).__name__}: {exc}")

    # -- models ----------------------------------------------------------------

    def llm(self, role: str) -> LLMProvider:
        """Provider assigned to a role in this run (built on first use)."""
        if role in self._providers:
            return self._providers[role]
        _, events = self._require_started()
        if role not in self.profile.roles:
            raise KeyError(f"run '{self.profile.name}' assigns no model to role '{role}'")

        profile = self.project.models[self.profile.roles[role]]
        provider = self._provider_builder(profile, self.clock)
        if self.project.app.llm.record_calls:
            assert self.dir is not None
            provider = RecordingProvider(
                provider, self.dir / "llm_calls.jsonl", role=role, events=events, clock=self.clock
            )
        self._providers[role] = provider
        return provider

    # -- internals -------------------------------------------------------------

    def _account_llm_call(self, event: Event) -> None:
        manifest, _ = self._require_started()
        role = event.payload["role"]
        stats = manifest.usage_by_role.setdefault(role, RoleUsage())
        stats.calls += 1
        if not event.payload.get("ok"):
            stats.errors += 1
            return
        stats.total_latency_ms += event.payload["latency_ms"]
        usage = event.payload["usage"]
        stats.usage.input_tokens += usage["input_tokens"]
        stats.usage.output_tokens += usage["output_tokens"]
        stats.usage.cache_read_tokens += usage["cache_read_tokens"]

    def _unique_run_id(self, stamp: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_-]", "-", self.profile.name)
        base = f"{stamp}_{safe}"
        candidate, n = base, 1
        while (self.project.runs_dir / candidate).exists():
            n += 1
            candidate = f"{base}_{n}"
        return candidate

    def _save_manifest(self) -> None:
        assert self.manifest is not None and self.dir is not None
        self.manifest.write(self.dir / "manifest.json")

    def _require_started(self) -> tuple[RunManifest, EventLog]:
        if self.manifest is None or self.events is None:
            raise RuntimeError("session not started (call start() or use 'with')")
        return self.manifest, self.events
