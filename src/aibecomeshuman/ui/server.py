"""Local control panel: a small web app served on 127.0.0.1 only.

The config files stay the source of truth: the UI reads them, edits them in place
(keeping comments), validates the whole project after every change and rolls the file
back if validation fails. API keys are never sent to the browser.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ValidationError

from aibecomeshuman.budget import project_spent
from aibecomeshuman.config import ROLES, BudgetSettings, ConfigError, ProjectConfig, load_project
from aibecomeshuman.core.events import Event, read_events
from aibecomeshuman.env.scenario import load_scenario
from aibecomeshuman.envfile import load_dotenv, set_env_value, source_of
from aibecomeshuman.llm.types import LLMError, LLMRequest, Message
from aibecomeshuman.play import ScenarioRunner
from aibecomeshuman.prompts import PromptLibrary, split_front_matter
from aibecomeshuman.run.session import RunSession
from aibecomeshuman.tomlpatch import patch_file

STATIC = Path(__file__).parent / "static"
API_KEY_VAR = "ANTHROPIC_API_KEY"
# Only requests addressed to this machine are served ("testserver" is the test client).
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "testserver"}

JobStatus = Literal["running", "ok", "error"]


class Job:
    def __init__(self) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.status: JobStatus = "running"
        self.events: list[dict[str, Any]] = []
        self.run_id: str | None = None
        self.error: str | None = None
        self._lock = threading.Lock()

    def push(self, event: Event) -> None:
        with self._lock:
            self.events.append(event.model_dump(mode="json"))

    def snapshot(self, after: int) -> dict[str, Any]:
        with self._lock:
            events = self.events[after:]
            return {
                "id": self.id,
                "status": self.status,
                "run_id": self.run_id,
                "error": self.error,
                "events": events,
                "next": after + len(events),
            }


class PlayRequest(BaseModel):
    scenario: str
    profile: str
    directive: str | None = None
    unlock_premium: bool = False


class BudgetUpdate(BaseModel):
    project_cap_usd: float
    run_cap_usd: float
    warn_ratio: float


class ApiKeyUpdate(BaseModel):
    key: str


class PingRequest(BaseModel):
    profile: str = "default"
    role: str = "decider_fast"


def _host_of(value: str) -> str:
    return (urlparse(f"//{value}").hostname or "").lower()


def create_app(root: Path | None = None) -> FastAPI:
    project_root = load_project(root).root
    env_file = project_root / ".env"
    app = FastAPI(title="aibecomeshuman", docs_url=None, redoc_url=None)
    jobs: dict[str, Job] = {}
    active: dict[str, Job | None] = {"job": None}

    def refresh_env() -> None:
        # Re-read .env every time, so a key added or changed later works without a restart.
        load_dotenv(env_file)

    refresh_env()

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        # Blocks other websites open in the browser from reaching the panel
        # (DNS rebinding via Host, cross-site writes via Origin / simple form posts).
        if _host_of(request.headers.get("host", "")) not in LOCAL_HOSTS:
            return JSONResponse({"detail": "forbidden host"}, status_code=403)
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin and (urlparse(origin).hostname or "").lower() not in LOCAL_HOSTS:
                return JSONResponse({"detail": "forbidden origin"}, status_code=403)
            if not request.headers.get("content-type", "").startswith("application/json"):
                return JSONResponse({"detail": "JSON body required"}, status_code=415)
        return await call_next(request)

    def project() -> ProjectConfig:
        try:
            return load_project(project_root)
        except ConfigError as e:
            raise HTTPException(500, f"configuration error: {e}") from e

    def scenarios_dir() -> Path:
        return project_root / "scenarios"

    def scenario_path(name: str) -> Path:
        path = (scenarios_dir() / name).resolve()
        if path.parent != scenarios_dir().resolve() or path.suffix != ".toml":
            raise HTTPException(400, "invalid scenario")
        return path

    def validated_patch(path: Path, section: str | None, updates: dict[str, Any]) -> None:
        original = patch_file(path, section, updates)
        try:
            load_project(project_root)
        except ConfigError as e:
            path.write_text(original, encoding="utf-8")
            raise HTTPException(400, f"change rejected, file restored: {e}") from e

    # -- pages -----------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    # -- read-only data --------------------------------------------------------

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        refresh_env()
        p = project()
        current = active["job"]
        return {
            "api_key_configured": bool(os.environ.get(API_KEY_VAR)),
            # 'file' = from .env, 'environment' = set in Windows/shell, None = missing
            "api_key_source": source_of(API_KEY_VAR),
            "project_spent_usd": project_spent(p.runs_dir),
            "budget": p.app.budget.model_dump(),
            "active_job": current.id if current and current.status == "running" else None,
        }

    @app.get("/api/options")
    def options() -> dict[str, Any]:
        p = project()
        scenarios = []
        for path in sorted(scenarios_dir().glob("*.toml")):
            try:
                s = load_scenario(path)
            except ConfigError:
                continue
            scenarios.append(
                {"file": path.name, "id": s.id, "title": s.title, "character": s.character}
            )
        directives = sorted(f.stem for f in (p.prompts_dir / "directives").glob("*.md"))
        return {
            "roles": list(ROLES),
            "models": [
                {
                    "name": m.name,
                    "provider": m.provider,
                    "model": m.model,
                    "locked": m.locked,
                    "paid": m.is_paid,
                    "price_input_per_mtok": m.price_input_per_mtok,
                    "price_output_per_mtok": m.price_output_per_mtok,
                }
                for m in sorted(p.models.values(), key=lambda m: m.name)
            ],
            "profiles": [
                {
                    "name": r.name,
                    "description": r.description,
                    "directive": r.directive,
                    "characters": r.characters,
                    "roles": r.roles,
                    "locked": any(p.models[m].locked for m in r.roles.values()),
                }
                for r in sorted(p.runs.values(), key=lambda r: r.name)
            ],
            "scenarios": scenarios,
            "directives": directives,
        }

    @app.get("/api/prompts")
    def prompts() -> dict[str, Any]:
        p = project()
        lib = PromptLibrary(p.prompts_dir)
        sheets = sorted((p.prompts_dir / "characters").glob("*.md"))
        characters = {f.stem: lib.character(f.stem) for f in sheets}
        directives = {}
        for f in sorted((p.prompts_dir / "directives").glob("*.md")):
            meta, body = split_front_matter(f.read_text(encoding="utf-8"))
            directives[f.stem] = {"meta": meta, "text": body}
        return {"characters": characters, "directives": directives}

    @app.get("/api/runs")
    def runs() -> list[dict[str, Any]]:
        p = project()
        rows = []
        for manifest in sorted(p.runs_dir.glob("*/manifest.json"), reverse=True):
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            events_path = manifest.parent / "events.jsonl"
            scene = None
            decisions = 0
            if events_path.is_file():
                for event in read_events(events_path):
                    if event.type == "scene.started" and scene is None:
                        scene = event.payload.get("title")
                    elif event.type == "decision.made":
                        decisions += 1
            rows.append(
                {
                    "run_id": data["run_id"],
                    "profile": data["profile"]["name"],
                    "status": data["status"],
                    "started_at": data["started_at"],
                    "cost_usd": data.get("cost_usd", 0.0),
                    "scene": scene,
                    "decisions": decisions,
                }
            )
        return rows

    @app.get("/api/runs/{run_id}/events")
    def run_events(run_id: str) -> list[dict[str, Any]]:
        p = project()
        path = (p.runs_dir / run_id / "events.jsonl").resolve()
        if path.parent.parent != p.runs_dir.resolve() or not path.is_file():
            raise HTTPException(404, "run not found")
        return [e.model_dump(mode="json") for e in read_events(path)]

    # -- playing ---------------------------------------------------------------

    @app.post("/api/play")
    def play(req: PlayRequest) -> dict[str, str]:
        current = active["job"]
        if current is not None and current.status == "running":
            raise HTTPException(409, "a run is already in progress")
        refresh_env()
        p = project()
        try:
            scenario = load_scenario(scenario_path(req.scenario))
            session = RunSession(p, req.profile, unlock_premium=req.unlock_premium)
        except ConfigError as e:
            raise HTTPException(400, str(e)) from e

        job = Job()
        jobs[job.id] = job
        active["job"] = job

        def work() -> None:
            try:
                with session:
                    assert session.events is not None and session.dir is not None
                    job.run_id = session.dir.name
                    for event in session.events:
                        job.push(event)
                    session.events.subscribe("*", job.push)
                    ScenarioRunner(session, scenario, directive=req.directive).run()
                job.status = "ok"
            except Exception as e:  # reported to the UI; the run manifest records it too
                job.error = f"{type(e).__name__}: {e}"
                job.status = "error"

        threading.Thread(target=work, daemon=True).start()
        return {"job_id": job.id}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str, after: int = 0) -> dict[str, Any]:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return job.snapshot(max(0, after))

    @app.post("/api/ping")
    def ping(req: PingRequest) -> dict[str, Any]:
        """One tiny call to check the connection (Haiku: about $0.00002)."""
        refresh_env()
        p = project()
        try:
            session = RunSession(p, req.profile)
        except ConfigError as e:
            raise HTTPException(400, str(e)) from e
        with session:
            try:
                provider = session.llm(req.role)
            except KeyError as e:
                raise HTTPException(400, str(e)) from e
            try:
                response = provider.complete(
                    LLMRequest(
                        purpose="ping",
                        system="Answer in one short sentence.",
                        messages=[Message.user("Connection test: say that it works.")],
                    )
                )
            except LLMError as e:
                session.finish("error", error=str(e))
                raise HTTPException(502, f"{provider.profile.name}: {e}") from e
        assert session.manifest is not None
        return {
            "model": response.model,
            "text": response.text,
            "latency_ms": round(response.latency_ms),
            "cost_usd": session.manifest.cost_usd,
        }

    # -- settings --------------------------------------------------------------

    @app.put("/api/settings/api-key")
    def update_api_key(update: ApiKeyUpdate) -> dict[str, Any]:
        key = update.key.strip()
        if not key.startswith("sk-ant-") or len(key) < 30 or any(c.isspace() for c in key):
            raise HTTPException(400, "that does not look like a Claude API key (sk-ant-...)")
        set_env_value(env_file, API_KEY_VAR, key)
        refresh_env()
        overridden = source_of(API_KEY_VAR) == "environment"
        return {
            "saved": True,
            # The key is saved, but a variable set in Windows takes priority over .env.
            "overridden_by_environment": overridden,
        }

    @app.put("/api/settings/budget")
    def update_budget(update: BudgetUpdate) -> dict[str, Any]:
        try:
            BudgetSettings.model_validate(update.model_dump())
        except ValidationError as e:
            raise HTTPException(400, f"invalid budget: {e.errors()[0]['msg']}") from e
        validated_patch(project_root / "config" / "default.toml", "budget", update.model_dump())
        return project().app.budget.model_dump()

    @app.put("/api/settings/runs/{profile}/roles")
    def update_roles(profile: str, roles: dict[str, str]) -> dict[str, str]:
        p = project()
        if profile not in p.runs:
            raise HTTPException(404, "run profile not found")
        for role, model in roles.items():
            if role not in ROLES:
                raise HTTPException(400, f"unknown role '{role}'")
            if model not in p.models:
                raise HTTPException(400, f"unknown model '{model}'")
        validated_patch(project_root / "config" / "runs" / f"{profile}.toml", "roles", roles)
        return project().runs[profile].roles

    return app


def serve(root: Path | None = None, port: int = 8765, open_browser: bool = True) -> None:
    import webbrowser

    import uvicorn

    app = create_app(root)
    url = f"http://127.0.0.1:{port}"
    print(f"aibecomeshuman control panel: {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
