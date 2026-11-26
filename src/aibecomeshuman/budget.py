"""Cost tracking and budget caps.

- `call_cost` turns a call's token usage into USD using the model profile's prices.
- `project_spent` adds up the cost of every previous run (from their manifests).
- `BudgetGuard` refuses paid calls once the run cap or the project cap is reached,
  and emits a `budget.warning` event when a cap is close.
"""

from __future__ import annotations

import json
from pathlib import Path

from aibecomeshuman.config import BudgetSettings, ModelProfile
from aibecomeshuman.core.events import EventLog
from aibecomeshuman.llm.base import LLMProvider
from aibecomeshuman.llm.types import BudgetExceeded, LLMRequest, LLMResponse, Usage


def call_cost(profile: ModelProfile, usage: Usage) -> float:
    return (
        usage.input_tokens * profile.price_input_per_mtok
        + usage.output_tokens * profile.price_output_per_mtok
        + usage.cache_read_tokens * profile.price_cache_read_per_mtok
    ) / 1_000_000


def project_spent(runs_dir: Path, exclude: Path | None = None) -> float:
    """Total USD spent by every run with a manifest in `runs_dir`."""
    total = 0.0
    if not runs_dir.is_dir():
        return total
    for manifest in runs_dir.glob("*/manifest.json"):
        if exclude is not None and manifest.parent == exclude:
            continue
        try:
            total += float(json.loads(manifest.read_text(encoding="utf-8")).get("cost_usd", 0))
        except (OSError, ValueError):
            continue
    return total


class BudgetGuard:
    def __init__(
        self,
        settings: BudgetSettings,
        prior_project_spent: float = 0.0,
        events: EventLog | None = None,
    ) -> None:
        self.settings = settings
        self.prior_project_spent = prior_project_spent
        self.run_spent = 0.0
        self._events = events
        self._warned: set[str] = set()

    @property
    def project_spent(self) -> float:
        return self.prior_project_spent + self.run_spent

    def check(self, profile: ModelProfile) -> None:
        """Raise `BudgetExceeded` if a paid call would go over a cap. Free models always pass."""
        if not profile.is_paid:
            return
        s = self.settings
        if self.run_spent >= s.run_cap_usd:
            raise BudgetExceeded(
                f"run budget reached (${self.run_spent:.4f} of ${s.run_cap_usd:.2f})"
            )
        if self.project_spent >= s.project_cap_usd:
            raise BudgetExceeded(
                f"project budget reached (${self.project_spent:.4f} of ${s.project_cap_usd:.2f})"
            )

    def record(self, cost_usd: float) -> None:
        self.run_spent += cost_usd
        s = self.settings
        self._maybe_warn("run", self.run_spent, s.run_cap_usd)
        self._maybe_warn("project", self.project_spent, s.project_cap_usd)

    def _maybe_warn(self, scope: str, spent: float, cap: float) -> None:
        if scope in self._warned or cap <= 0 or spent < cap * self.settings.warn_ratio:
            return
        self._warned.add(scope)
        if self._events is not None:
            self._events.emit(
                "budget.warning",
                source="budget",
                payload={"scope": scope, "spent_usd": round(spent, 6), "cap_usd": cap},
            )


class BudgetedProvider:
    """Checks the budget before every call to the wrapped provider."""

    def __init__(self, inner: LLMProvider, guard: BudgetGuard) -> None:
        self.inner = inner
        self.profile = inner.profile
        self._guard = guard

    def complete(self, request: LLMRequest) -> LLMResponse:
        self._guard.check(self.profile)
        return self.inner.complete(request)
