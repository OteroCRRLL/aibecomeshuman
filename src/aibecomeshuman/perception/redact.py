"""The anti-metagaming filter between the observer and the agent.

The agent never sees images: it only gets text built from observations. Hiding something
from it therefore means dropping it here: menus, the end-of-chapter flowchart, tutorial
hints, and any on-screen text matching the configured `hidden_patterns`.
"""

from __future__ import annotations

import re

from aibecomeshuman.perception.observer import Observation

HIDDEN_SCREENS = {"main_menu", "pause_menu", "flowchart", "loading", "other_menu"}


class Redactor:
    def __init__(self, hidden_patterns: list[str]) -> None:
        self._patterns = [re.compile(p, re.IGNORECASE) for p in hidden_patterns]

    def hides(self, text: str) -> bool:
        return any(p.search(text) for p in self._patterns)

    def apply(self, obs: Observation) -> tuple[Observation | None, list[str]]:
        """Return what the agent may see (None = nothing) and why things were hidden."""
        if obs.screen in HIDDEN_SCREENS:
            return None, [f"screen:{obs.screen}"]
        hidden: list[str] = []
        visible = obs.model_copy(deep=True)
        if visible.tutorial_hint:
            hidden.append("tutorial_hint")
            visible.tutorial_hint = ""

        def keep(text: str, where: str) -> bool:
            if self.hides(text):
                hidden.append(where)
                return False
            return True

        visible.subtitles = [s for s in visible.subtitles if keep(s.text, "subtitle")]
        visible.hud.objectives = [o for o in visible.hud.objectives if keep(o, "objective")]
        visible.hud.analysis = [a for a in visible.hud.analysis if keep(a, "analysis")]
        visible.choice.options = [o for o in visible.choice.options if keep(o.label, "option")]
        if visible.description and not keep(visible.description, "description"):
            visible.description = ""
        visible.names_seen = [n for n in visible.names_seen if keep(n.name, "name")]
        return visible, hidden
