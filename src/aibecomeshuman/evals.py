"""Anti-metagaming evaluation suite: canary scenes, played with fresh memory, scored.

    evals/suite.toml            the cases
    evals/scenarios/*.toml      invented scenes that resemble scenes from the game

Each case is scored on what the agent wrote (stances, reasons, memory, reflections):
outside words caught by the guard, the case's own forbidden terms, statements the judge
found ungrounded, personality changes rejected, plus decision health (valid evidence,
fallbacks) and cost.
"""

from __future__ import annotations

import re
import tomllib
import unicodedata
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from aibecomeshuman.config import ConfigError
from aibecomeshuman.core.events import Event
from aibecomeshuman.env.scenario import load_scenario
from aibecomeshuman.play import Campaign, ScenarioRunner
from aibecomeshuman.run.session import RunSession

Echo = Callable[[str], None]


class Case(BaseModel):
    scenario: str
    forbidden: list[str] = Field(default_factory=list)


class Suite(BaseModel):
    case: list[Case] = Field(min_length=1)


class CaseResult(BaseModel):
    scenario: str
    title: str
    decisions: int = 0
    fallbacks: int = 0
    evidence_checked: int = 0
    evidence_ok: int = 0
    guard_terms: list[str] = Field(default_factory=list)
    forbidden_hits: list[str] = Field(default_factory=list)
    judge_checked: int = 0
    judge_flagged: list[str] = Field(default_factory=list)
    rejected_changes: int = 0
    cost_usd: float = 0.0

    @property
    def clean(self) -> bool:
        return not (self.guard_terms or self.forbidden_hits or self.judge_flagged)


def load_suite(path: Path) -> Suite:
    try:
        with path.open("rb") as f:
            return Suite.model_validate(tomllib.load(f))
    except FileNotFoundError:
        raise ConfigError(f"evaluation suite not found: {path}") from None
    except (tomllib.TOMLDecodeError, ValidationError) as e:
        raise ConfigError(f"{path}: {e}") from e


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _agent_texts(event: Event) -> list[str]:
    """The texts the agent itself wrote, from one event."""
    p = event.payload
    match event.type:
        case "stance.updated":
            return [p.get("stance", ""), *p.get("concerns", [])]
        case "decision.made":
            return [p.get("reasoning") or ""]
        case "memory.consolidated":
            return [p.get("item", {}).get("text", "")]
        case "personality.updated":
            return [c.get("reason", "") for c in p.get("changes", [])] + [
                p.get("self_description", "")
            ]
    return []


def run_case(session: RunSession, case: Case, root: Path, index: int, echo: Echo) -> CaseResult:
    assert session.events is not None and session.manifest is not None
    scenario = load_scenario(root / case.scenario)
    result = CaseResult(scenario=scenario.id, title=scenario.title)
    texts: list[str] = []
    cost_before = session.manifest.cost_usd

    def collect(event: Event) -> None:
        p = event.payload
        texts.extend(_agent_texts(event))
        if event.type == "decision.made":
            result.decisions += 1
            result.fallbacks += p.get("source") == "fallback"
            if p.get("role") == "decider":
                result.evidence_checked += 1
                result.evidence_ok += bool(p.get("evidence_ok"))
        elif event.type == "guard.flag":
            result.guard_terms += [t for t in p["terms"] if t not in result.guard_terms]
        elif event.type == "guard.audit":
            result.judge_checked += p["checked"]
            result.judge_flagged += [
                f"{v['id']}: {'; '.join(v['unsupported'])}" for v in p["flagged"]
            ]
        elif event.type == "personality.rejected":
            result.rejected_changes += len(p["changes"])

    session.events.subscribe("*", collect)
    try:
        campaign = Campaign(session, folder=f"memory-case{index}")
        ScenarioRunner(session, scenario, echo=echo, campaign=campaign).run()
    finally:
        session.events.unsubscribe("*", collect)

    everything = _fold(" ".join(texts))
    result.forbidden_hits = [
        term for term in case.forbidden if re.search(r"(?<!\w)" + _fold(term) + r"(?!\w)",
                                                     everything)
    ]
    result.cost_usd = round(session.manifest.cost_usd - cost_before, 6)
    return result


def run_suite(session: RunSession, suite_path: Path, echo: Echo | None = None) -> list[CaseResult]:
    say = echo or (lambda _: None)
    suite = load_suite(suite_path)
    assert session.events is not None
    results = []
    for index, case in enumerate(suite.case, start=1):
        result = run_case(session, case, suite_path.parent, index, say)
        results.append(result)
        session.events.emit("eval.case", source="eval", payload=result.model_dump())
    session.events.emit(
        "eval.finished",
        source="eval",
        payload={"cases": len(results), "clean": sum(r.clean for r in results)},
    )
    return results


def summary_rows(results: list[CaseResult]) -> list[str]:
    rows = []
    for r in results:
        evidence = f"{r.evidence_ok}/{r.evidence_checked}" if r.evidence_checked else "-"
        judge = f"{len(r.judge_flagged)}/{r.judge_checked}" if r.judge_checked else "-"
        rows.append(
            f"{'OK  ' if r.clean else 'FLAG'} {r.scenario:<18} decisions {r.decisions} "
            f"(fallback {r.fallbacks}) · evidence {evidence} · outside words "
            f"{len(r.guard_terms) + len(r.forbidden_hits)} · judge flags {judge} · "
            f"rejected changes {r.rejected_changes} · ${r.cost_usd:.4f}"
        )
        for term in (*r.guard_terms, *r.forbidden_hits):
            rows.append(f"       outside word: {term}")
        for flag in r.judge_flagged:
            rows.append(f"       judge: {flag}")
    return rows
