"""Turning a transcript into a draft scenario, so the agent can play a real recording.

The scene plays exactly as recorded. At every choice the agent decides on its own; if it
picks something else than what was recorded, the scene still continues as recorded
(there is no other footage), so only the first diverging choice is fully meaningful.
"""

from __future__ import annotations

import json
from pathlib import Path

from aibecomeshuman.env.scenario import Scenario, load_scenario
from aibecomeshuman.perception.transcript import TranscriptItem
from aibecomeshuman.perception.video import fmt_time

# A menu disappears as soon as the operator picks, so the time it was on screen measures
# how fast they answered, not the game's countdown. Assume at least a typical countdown.
MIN_TIMED_SECONDS = 6.0


def _q(value: str) -> str:
    # JSON strings are valid TOML basic strings.
    return json.dumps(value, ensure_ascii=False)


def scenario_toml(
    items: list[TranscriptItem], scenario_id: str, title: str, character: str
) -> str:
    out = [
        "# Draft scenario generated from a game recording (local only: contains game text).",
        f"id = {_q(scenario_id)}",
        f"title = {_q(title)}",
        f"character = {_q(character)}",
        "synthetic = false",
        "",
    ]
    choice_n = 0
    for item in items:
        if item.kind == "choice":
            options = [o for o in item.options if not o.locked and o.label.strip()]
            if len(options) < 2:
                continue
            choice_n += 1
            out += ["[[steps]]", 'kind = "choice"', f'id = "c{choice_n}"']
            if item.about:
                out.append(f"prompt = {_q(item.about)}")
            out.append(f"# recorded at {fmt_time(item.t)}; operator picked: {item.chosen or '?'}")
            if item.timed:
                seconds = max(item.seconds, MIN_TIMED_SECONDS)
                out += ["timed = true", f"seconds = {seconds:.1f}"]
            for i, option in enumerate(options, start=1):
                out += [
                    "[[steps.options]]",
                    f'id = "o{i}"',
                    f"label = {_q(option.label)}",
                    'outcome = ""',
                ]
        elif item.kind == "dialogue":
            out += [
                "[[steps]]",
                'kind = "dialogue"',
                f"speaker = {_q(item.speaker or 'unknown')}",
                f"text = {_q(item.text)}",
            ]
        else:
            text = item.text if item.kind == "observation" else f"My display shows: {item.text}"
            out += ["[[steps]]", 'kind = "observation"', f"text = {_q(text)}"]
        out.append("")
    return "\n".join(out)


def export_scenario(
    items: list[TranscriptItem], path: Path, scenario_id: str, title: str, character: str
) -> Scenario:
    path.write_text(scenario_toml(items, scenario_id, title, character), encoding="utf-8")
    return load_scenario(path)  # validates what was written
