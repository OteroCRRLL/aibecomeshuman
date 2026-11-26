"""Recording and replay of model calls.

- `RecordingProvider` wraps any provider: it stores every call as JSONL and publishes an
  `llm.call` event (metrics only, no content) to the run's event log.
- `CassetteProvider` replays a recorded file: same request -> same response, with no
  network and no cost. The basis for repeatable tests and the multi-model arena.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from aibecomeshuman.config import ModelProfile
from aibecomeshuman.core.clock import Clock, SystemClock
from aibecomeshuman.core.events import EventLog
from aibecomeshuman.llm.base import LLMProvider
from aibecomeshuman.llm.types import LLMError, LLMRequest, LLMResponse


class RecordingProvider:
    def __init__(
        self,
        inner: LLMProvider,
        path: Path,
        role: str,
        events: EventLog | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.inner = inner
        self.profile: ModelProfile = inner.profile
        self.path = path
        self.role = role
        self._events = events
        self._clock = clock or SystemClock()
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def complete(self, request: LLMRequest) -> LLMResponse:
        fingerprint = request.fingerprint()
        record: dict[str, Any] = {
            "ts": self._clock.now().isoformat(),
            "role": self.role,
            "profile": self.profile.name,
            "fingerprint": fingerprint,
            "request": request.redacted(),
        }
        try:
            response = self.inner.complete(request)
        except LLMError as e:
            record["error"] = {"type": type(e).__name__, "message": str(e)}
            self._write(record)
            self._emit(request, fingerprint, error=str(e))
            raise
        record["response"] = response.model_dump(mode="json")
        self._write(record)
        self._emit(request, fingerprint, response=response)
        return response

    def _write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _emit(
        self,
        request: LLMRequest,
        fingerprint: str,
        response: LLMResponse | None = None,
        error: str | None = None,
    ) -> None:
        if self._events is None:
            return
        payload: dict[str, Any] = {
            "role": self.role,
            "profile": self.profile.name,
            "purpose": request.purpose,
            "fingerprint": fingerprint,
        }
        if response is not None:
            from aibecomeshuman.budget import call_cost

            payload |= {
                "ok": True,
                "model": response.model,
                "latency_ms": round(response.latency_ms, 1),
                "usage": response.usage.model_dump(),
                "cost_usd": round(call_cost(self.profile, response.usage), 8),
            }
        else:
            payload |= {"ok": False, "error": error}
        self._events.emit("llm.call", source="llm", payload=payload)


class CassetteProvider:
    """Replays recorded responses, indexed by request fingerprint."""

    def __init__(self, profile: ModelProfile, path: Path) -> None:
        self.profile = profile
        self._responses: dict[str, LLMResponse] = {}
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if "response" in record:
                    self._responses[record["fingerprint"]] = LLMResponse.model_validate(
                        record["response"]
                    )

    def __len__(self) -> int:
        return len(self._responses)

    def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            return self._responses[request.fingerprint()]
        except KeyError:
            raise LLMError(
                f"the cassette has no response for this request ({request.purpose}); "
                "has the prompt changed since it was recorded?"
            ) from None
