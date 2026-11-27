"""Better eyes for key moments: fast action without dialogue is read again by a stronger
vision model, within a spend cap per chapter.

The cheap observer reads every keyframe. A frame is escalated only when all of these hold:
- it is part of a burst of big picture changes (several within a few seconds),
- it has no subtitles and no menu (dialogue and choices are read fine by the cheap model),
- the cheap observer's description mentions something dramatic (shots, falls, blood...),
- this chapter's escalation spend is under the cap.
Tested on the first mission's ending: the cheap model mistook the android being shot for
the protagonist; the stronger one did not.
"""

from __future__ import annotations

import re
import threading
from collections import deque

from aibecomeshuman.perception.observer import Observation

# Burst: at least BURST_MIN big changes within BURST_WINDOW_S seconds.
BURST_WINDOW_S = 4.0
BURST_MIN = 2

DRAMATIC = re.compile(
    r"shot|shoot|fir(e|es|ing)\b|gun|sniper|bullet|hit\b|struck|strike|fall|fell|drop|"
    r"collaps|jump|leap|lung|tackl|grab|push|shov|fight|punch|attack|stab|blood|wound|"
    r"explo|burst|dissolv|break|smash|crash|dead|die|dying|kill|bod(y|ies)",
    re.IGNORECASE,
)


class Escalator:
    def __init__(self, strong_observer, cap_per_chapter_usd: float, spent) -> None:
        """`spent()` returns the escalation spend so far (USD) for the whole run."""
        self.strong = strong_observer
        self.cap = cap_per_chapter_usd
        self._spent = spent
        self._chapter_start = 0.0
        self._changes: deque[float] = deque(maxlen=16)
        self._lock = threading.Lock()

    def note_keyframe(self, t: float, reason: str) -> bool:
        """Record a keyframe (in capture order); True if it belongs to a burst of changes."""
        with self._lock:
            if reason != "change":
                return False
            self._changes.append(t)
            return sum(1 for x in self._changes if t - x <= BURST_WINDOW_S) >= BURST_MIN

    def new_chapter(self) -> None:
        with self._lock:
            self._chapter_start = self._spent()

    def chapter_spend(self) -> float:
        return self._spent() - self._chapter_start

    def wanted(self, obs: Observation, in_burst: bool) -> bool:
        return (
            in_burst
            and not obs.subtitles
            and obs.choice.kind == "none"
            and obs.screen in ("cutscene", "gameplay")
            and bool(DRAMATIC.search(obs.description))
            and self.chapter_spend() < self.cap
        )
