from __future__ import annotations

from pathlib import Path

from aibecomeshuman.core.clock import FakeClock
from aibecomeshuman.core.events import EventLog, read_events


def test_emit_appends_with_monotonic_seq(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.jsonl", clock=FakeClock(auto_step=1))
    log.emit("a", source="test", payload={"x": 1})
    log.emit("b", source="test")

    events = list(read_events(log.path))
    assert [e.seq for e in events] == [1, 2]
    assert [e.type for e in events] == ["a", "b"]
    assert events[0].payload == {"x": 1}
    assert events[1].ts > events[0].ts


def test_reopening_continues_sequence(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    EventLog(path).emit("a", source="test")
    reopened = EventLog(path)
    assert reopened.last_seq == 1
    assert reopened.emit("b", source="test").seq == 2


def test_subscribers_by_type_and_wildcard(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.jsonl")
    typed, everything = [], []
    log.subscribe("llm.call", typed.append)
    log.subscribe("*", everything.append)

    log.emit("llm.call", source="llm")
    log.emit("run.started", source="run")

    assert [e.type for e in typed] == ["llm.call"]
    assert [e.type for e in everything] == ["llm.call", "run.started"]


def test_failing_subscriber_does_not_lose_event(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.jsonl")
    received = []

    def broken(_event):
        raise RuntimeError("boom")

    log.subscribe("*", broken)
    log.subscribe("*", received.append)
    log.emit("a", source="test")

    assert len(received) == 1
    assert len(list(read_events(log.path))) == 1


def test_truncated_last_line_is_ignored(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.jsonl")
    log.emit("a", source="test")
    with log.path.open("a", encoding="utf-8") as f:
        f.write('{"seq": 2, "ts": "2038-')  # cierre brusco a mitad de escritura

    assert [e.seq for e in read_events(log.path)] == [1]
