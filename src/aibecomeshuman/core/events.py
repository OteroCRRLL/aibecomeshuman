"""Append-only event log: the single source of truth for a run.

Perception, control and cognition publish events here. Memory, the overlay and the
editing markers are projections built by reading them (live, or by replaying the file).
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from aibecomeshuman.core.clock import Clock, SystemClock

log = logging.getLogger(__name__)

WILDCARD = "*"


class Event(BaseModel):
    seq: int
    ts: datetime
    type: str
    source: str
    payload: dict[str, Any] = Field(default_factory=dict)


Handler = Callable[[Event], None]


def read_events(path: Path) -> Iterator[Event]:
    """Read an event file. A truncated last line (abrupt shutdown) is skipped."""
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield Event.model_validate_json(line)
            except ValueError:
                log.warning("Line %d of %s is unreadable; skipping", lineno, path)


class EventLog:
    """Writes events as JSONL and dispatches them to subscribers on the same thread.

    A failing subscriber is logged but never prevents the event from being stored or
    delivered to the others: the log is never lost because a projection crashed.
    """

    def __init__(self, path: Path, clock: Clock | None = None) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock or SystemClock()
        self._lock = threading.Lock()
        self._subscribers: dict[str, list[Handler]] = defaultdict(list)
        self._seq = self._last_seq()

    def _last_seq(self) -> int:
        if not self.path.exists():
            return 0
        last = 0
        for event in read_events(self.path):
            last = event.seq
        return last

    @property
    def last_seq(self) -> int:
        return self._seq

    def subscribe(self, event_type: str, handler: Handler) -> None:
        """Subscribe to one event type (`"llm.call"`) or to all of them (`"*"`)."""
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Handler) -> None:
        handlers = self._subscribers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    def emit(self, type: str, source: str, payload: dict[str, Any] | None = None) -> Event:
        with self._lock:
            self._seq += 1
            event = Event(
                seq=self._seq,
                ts=self._clock.now(),
                type=type,
                source=source,
                payload=payload or {},
            )
            with self.path.open("a", encoding="utf-8") as f:
                f.write(event.model_dump_json() + "\n")

        for handler in (*self._subscribers.get(type, ()), *self._subscribers.get(WILDCARD, ())):
            try:
                handler(event)
            except Exception:
                log.exception("Subscriber failed handling %s #%d", event.type, event.seq)
        return event

    def __iter__(self) -> Iterator[Event]:
        if self.path.exists():
            yield from read_events(self.path)
