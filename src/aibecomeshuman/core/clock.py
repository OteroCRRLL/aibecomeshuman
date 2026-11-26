"""Injectable clock: real during play, controllable in tests and simulation."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime:
        """Wall-clock time (UTC) used to timestamp events."""
        ...

    def monotonic(self) -> float:
        """Monotonic seconds used to measure latency."""
        ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return time.perf_counter()


class FakeClock:
    """Deterministic clock. Only advances when told to (or by `auto_step` on each read)."""

    def __init__(
        self,
        start: datetime | None = None,
        auto_step: float = 0.0,
    ) -> None:
        self._now = start or datetime(2038, 11, 5, 20, 0, tzinfo=UTC)
        self._mono = 0.0
        self._auto_step = auto_step

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)
        self._mono += seconds

    def now(self) -> datetime:
        value = self._now
        if self._auto_step:
            self.advance(self._auto_step)
        return value

    def monotonic(self) -> float:
        return self._mono
