from __future__ import annotations

import json
from pathlib import Path

import pytest

from aibecomeshuman.budget import BudgetGuard, call_cost, project_spent
from aibecomeshuman.config import BudgetSettings, ConfigError, ModelProfile, load_project
from aibecomeshuman.core.events import EventLog
from aibecomeshuman.llm.types import BudgetExceeded, Usage
from aibecomeshuman.run.session import RunSession

PAID = ModelProfile(
    name="paid",
    provider="fake",
    model="x",
    price_input_per_mtok=1.0,
    price_output_per_mtok=10.0,
    price_cache_read_per_mtok=0.1,
)
FREE = ModelProfile(name="free", provider="fake", model="x")


def test_call_cost() -> None:
    usage = Usage(input_tokens=1_000_000, output_tokens=100_000, cache_read_tokens=1_000_000)
    assert call_cost(PAID, usage) == pytest.approx(1.0 + 1.0 + 0.1)
    assert call_cost(FREE, usage) == 0.0


def test_run_cap_blocks_paid_but_not_free() -> None:
    guard = BudgetGuard(BudgetSettings(run_cap_usd=0.10, project_cap_usd=100))
    guard.check(PAID)
    guard.record(0.10)
    with pytest.raises(BudgetExceeded, match="run budget"):
        guard.check(PAID)
    guard.check(FREE)  # free models are never blocked


def test_project_cap_counts_previous_runs() -> None:
    guard = BudgetGuard(BudgetSettings(run_cap_usd=5, project_cap_usd=1.0), 0.99)
    guard.check(PAID)
    guard.record(0.02)
    with pytest.raises(BudgetExceeded, match="project budget"):
        guard.check(PAID)


def test_warning_emitted_once(tmp_path: Path) -> None:
    events = EventLog(tmp_path / "events.jsonl")
    guard = BudgetGuard(BudgetSettings(run_cap_usd=1.0, project_cap_usd=100), events=events)
    guard.record(0.85)
    guard.record(0.05)
    warnings = [e for e in events if e.type == "budget.warning"]
    assert [w.payload["scope"] for w in warnings] == ["run"]


def test_project_spent_sums_manifests(tmp_path: Path) -> None:
    for name, cost in [("a", 0.25), ("b", 0.5)]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "manifest.json").write_text(
            json.dumps({"cost_usd": cost}), encoding="utf-8"
        )
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "manifest.json").write_text("{not json", encoding="utf-8")
    assert project_spent(tmp_path) == pytest.approx(0.75)
    assert project_spent(tmp_path, exclude=tmp_path / "a") == pytest.approx(0.5)


def test_locked_models_need_unlock(project_dir: Path) -> None:
    project = load_project(project_dir)
    with pytest.raises(ConfigError, match="unlock-premium"):
        RunSession(project, "video-claude")
    RunSession(project, "video-claude", unlock_premium=True)  # no error


def test_paid_calls_are_costed_and_blocked(project_dir: Path) -> None:
    # A "paid" fake model: free to run, but priced, so the budget logic is exercised.
    (project_dir / "config" / "models" / "pricey.toml").write_text(
        'provider = "fake"\nmodel = "pricey"\n'
        "price_input_per_mtok = 100000.0\nprice_output_per_mtok = 100000.0\n",
        encoding="utf-8",
    )
    (project_dir / "config" / "runs" / "pricey.toml").write_text(
        'characters = ["kara"]\n[roles]\ndecider = "pricey"\n', encoding="utf-8"
    )
    from aibecomeshuman.llm.types import LLMRequest, Message

    with RunSession(load_project(project_dir), "pricey") as session:
        decider = session.llm("decider")
        decider.complete(LLMRequest(purpose="t", messages=[Message.user("hola")]))
        with pytest.raises(BudgetExceeded):
            decider.complete(LLMRequest(purpose="t", messages=[Message.user("otra")]))

    manifest = json.loads((session.dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cost_usd"] > 0.5
    assert manifest["usage_by_role"]["decider"]["errors"] == 1
