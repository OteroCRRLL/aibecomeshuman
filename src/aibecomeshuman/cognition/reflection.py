"""End-of-scene memory work.

1. `consolidate` (summarizer role, cheap): turns the scene into one first-person
   recollection, facts about people/things, feelings and trust, and new or resolved
   promises.
2. `reflect` (reflector role): proposes small, justified trait changes and an updated
   self-description. Changes are bounded in code, whatever the model proposes.

Both only use what happened in the scene and what the character already remembers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from aibecomeshuman.cognition.context import SceneContext
from aibecomeshuman.cognition.decider import Decision
from aibecomeshuman.config import MemorySettings
from aibecomeshuman.llm.base import LLMProvider
from aibecomeshuman.llm.types import LLMRequest, Message
from aibecomeshuman.memory.models import (
    CharacterMemory,
    Commitment,
    Entity,
    MemoryItem,
    TraitChange,
    entity_key,
)
from aibecomeshuman.memory.render import render_memory, render_personality

ENTITY_KINDS = ["person", "place", "object", "group", "other"]

CONSOLIDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "importance": {"type": "integer"},
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string"},
                    "kind": {"type": "string", "enum": ENTITY_KINDS},
                    "fact": {"type": "string"},
                },
                "required": ["entity", "kind", "fact"],
                "additionalProperties": False,
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string"},
                    "feeling": {"type": "string"},
                    "trust_change": {"type": "integer"},
                },
                "required": ["entity", "feeling", "trust_change"],
                "additionalProperties": False,
            },
        },
        "new_promises": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"to": {"type": "string"}, "text": {"type": "string"}},
                "required": ["to", "text"],
                "additionalProperties": False,
            },
        },
        "resolved_promises": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "status": {"type": "string", "enum": ["kept", "broken", "dropped"]},
                    "note": {"type": "string"},
                },
                "required": ["id", "status", "note"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "summary",
        "importance",
        "facts",
        "relationships",
        "new_promises",
        "resolved_promises",
    ],
    "additionalProperties": False,
}

CONSOLIDATE_TASK = (
    "The scene is over. Write it into your long-term memory.\n"
    "- `summary`: what happened and what you did, in first person, in two to four "
    "sentences. Only what is in the scene; nothing invented.\n"
    "- `importance`: 1 (routine) to 5 (life-changing) for you.\n"
    "- `facts`: things you learned about people, places or objects. Use the same names "
    "as the speaker labels when there are any, and the same names as in your memory for "
    "known entities.\n"
    "- `relationships`: for each person who mattered in the scene, how you feel about "
    "them now and how your trust changed (-2 to +2, 0 if unchanged).\n"
    "- `new_promises`: promises you made in this scene, if any.\n"
    "- `resolved_promises`: ids of earlier promises this scene kept, broke or made moot."
)


def reflect_schema(trait_names: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "changed": {"type": "boolean"},
            "trait_changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "trait": {"type": "string", "enum": trait_names},
                        "delta": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": ["trait", "delta", "reason"],
                    "additionalProperties": False,
                },
            },
            "self_description": {"type": "string"},
        },
        "required": ["changed", "trait_changes", "self_description"],
        "additionalProperties": False,
    }


REFLECT_TASK = (
    "Reflect on the scene you just lived and on the choices you made.\n"
    "- Did it change who you are? Most scenes change nothing, or very little. Set "
    "`changed` to false if nothing about you changed.\n"
    "- `trait_changes`: only traits that this scene really moved, each with a `delta` "
    "between -{step:g} and +{step:g} and a `reason` that points to something concrete "
    "that happened. Choices marked as reflexes were made under time pressure: weigh them "
    "less.\n"
    "- `self_description`: how you see yourself now, in first person, in one or two "
    "sentences. If nothing changed, repeat the current one (or leave it empty if you have "
    "none yet)."
)


class SceneReport(BaseModel):
    item: MemoryItem
    new_commitments: list[Commitment] = Field(default_factory=list)
    resolved_commitments: list[Commitment] = Field(default_factory=list)
    entities_touched: list[str] = Field(default_factory=list)


def render_decisions(decisions: list[tuple[Decision, str]]) -> str:
    if not decisions:
        return "<choices_made>None.</choices_made>"
    lines = ["<choices_made>"]
    for d, label in decisions:
        kind = "reflex" if d.source == "fallback" or d.role == "decider_fast" else "deliberate"
        reason = f" Why: {d.reasoning}" if d.reasoning else ""
        lines.append(f'<choice kind="{kind}">I chose: {label}.{reason}</choice>')
    lines.append("</choices_made>")
    return "\n".join(lines)


def consolidate(
    provider: LLMProvider,
    system: str,
    memory: CharacterMemory,
    ctx: SceneContext,
    scene_title: str,
    settings: MemorySettings,
    run_id: str | None,
) -> SceneReport:
    """Ask the summarizer to write the scene into memory and apply the result."""
    scene_text = ctx.render()
    request = LLMRequest(
        purpose="memory.consolidate",
        system=system,
        messages=[
            Message.user(
                "\n\n".join(
                    [
                        render_memory(memory, scene_text, settings),
                        scene_text,
                        f"<task>{CONSOLIDATE_TASK}</task>",
                    ]
                )
            )
        ],
        json_schema=CONSOLIDATE_SCHEMA,
    )
    parsed = provider.complete(request).parsed or {}
    return apply_consolidation(memory, parsed, ctx.scene_id, scene_title, settings, run_id)


def apply_consolidation(
    memory: CharacterMemory,
    parsed: dict[str, Any],
    scene_id: str,
    scene_title: str,
    settings: MemorySettings,
    run_id: str | None = None,
) -> SceneReport:
    touched: list[str] = []

    def entity(name: str, kind: str = "other") -> Entity | None:
        key = entity_key(name)
        if not key:
            return None
        if key not in memory.entities:
            memory.entities[key] = Entity(
                name=name.strip(),
                kind=kind if kind in ENTITY_KINDS else "other",  # type: ignore[arg-type]
                first_scene=scene_id,
                last_scene=scene_id,
            )
        e = memory.entities[key]
        e.last_scene = scene_id
        if e.name not in touched:
            touched.append(e.name)
        return e

    for fact in parsed.get("facts", []):
        e = entity(str(fact.get("entity", "")), str(fact.get("kind", "other")))
        text = str(fact.get("fact", "")).strip()
        if e is not None and text and text not in e.facts:
            e.facts.append(text)
            e.facts = e.facts[-settings.max_facts_per_entity * 2 :]

    for rel in parsed.get("relationships", []):
        e = entity(str(rel.get("entity", "")), "person")
        if e is None:
            continue
        if rel.get("feeling"):
            e.feeling = str(rel["feeling"]).strip()
        change = max(-2, min(2, int(rel.get("trust_change", 0) or 0)))
        e.trust = max(-5, min(5, e.trust + change))

    raw_importance = parsed.get("importance")
    importance = 3 if raw_importance is None else int(raw_importance)
    item = MemoryItem(
        id=memory.next_item_id(),
        scene_id=scene_id,
        scene_title=scene_title,
        text=str(parsed.get("summary", "")).strip() or f"I lived the scene '{scene_title}'.",
        importance=max(1, min(5, importance)),
        entities=list(touched),
        run_id=run_id,
    )
    memory.items.append(item)
    if scene_id not in memory.scenes_played:
        memory.scenes_played.append(scene_id)

    new: list[Commitment] = []
    for promise in parsed.get("new_promises", []):
        text = str(promise.get("text", "")).strip()
        if not text:
            continue
        c = Commitment(
            id=memory.next_commitment_id(),
            text=text,
            to=str(promise.get("to", "")).strip(),
            scene_id=scene_id,
        )
        memory.commitments.append(c)
        new.append(c)

    resolved: list[Commitment] = []
    by_id = {c.id: c for c in memory.commitments}
    for res in parsed.get("resolved_promises", []):
        c = by_id.get(str(res.get("id", "")))
        status = res.get("status")
        valid_status = status in ("kept", "broken", "dropped")
        if c is None or c.status != "active" or c in new or not valid_status:
            continue
        c.status = status
        c.resolved_scene_id = scene_id
        c.note = str(res.get("note", "")).strip()
        resolved.append(c)

    return SceneReport(
        item=item, new_commitments=new, resolved_commitments=resolved, entities_touched=touched
    )


def reflect(
    provider: LLMProvider,
    system: str,
    memory: CharacterMemory,
    report: SceneReport,
    decisions: list[tuple[Decision, str]],
    settings: MemorySettings,
    review: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> list[TraitChange]:
    """Ask the reflector how the scene changed the character and apply bounded changes.

    `review` sees the proposal before it is applied and may drop parts of it (the
    anti-metagaming judge removes changes whose reasons did not happen)."""
    request = LLMRequest(
        purpose="personality.reflect",
        system=system,
        messages=[
            Message.user(
                "\n\n".join(
                    [
                        render_personality(memory),
                        f'<scene_recollection id="{report.item.id}">{report.item.text}'
                        "</scene_recollection>",
                        render_decisions(decisions),
                        f"<task>{REFLECT_TASK.format(step=settings.max_trait_step)}</task>",
                    ]
                )
            )
        ],
        json_schema=reflect_schema(list(memory.personality.traits)),
    )
    parsed = provider.complete(request).parsed or {}
    if review is not None:
        parsed = review(parsed)
    return apply_reflection(memory, parsed, report.item.scene_id, settings)


def apply_reflection(
    memory: CharacterMemory,
    parsed: dict[str, Any],
    scene_id: str,
    settings: MemorySettings,
) -> list[TraitChange]:
    p = memory.personality
    changes: list[TraitChange] = []
    if parsed.get("changed", True):
        seen: set[str] = set()
        for proposal in parsed.get("trait_changes", []):
            name = proposal.get("trait")
            reason = str(proposal.get("reason", "")).strip()
            if name not in p.traits or name in seen or not reason:
                continue
            step = settings.max_trait_step
            delta = max(-step, min(step, float(proposal.get("delta", 0) or 0)))
            old = p.traits[name].value
            new = round(max(0.0, min(10.0, old + delta)), 1)
            if abs(new - old) < 0.05:
                continue
            seen.add(name)
            changes.append(
                TraitChange(
                    trait=name, old=old, new=new, reason=reason,
                    scene_id=scene_id, version=p.version + 1,
                )
            )
        description = str(parsed.get("self_description", "")).strip()
        if changes or (description and description != p.self_description):
            p.version += 1
            for ch in changes:
                ch.version = p.version
                p.traits[ch.trait].value = ch.new
            p.history.extend(changes)
            if description:
                p.self_description = description
    return changes
