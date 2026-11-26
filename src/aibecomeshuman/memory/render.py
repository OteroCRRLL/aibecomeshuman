"""Turns a CharacterMemory into prompt text with a fixed budget.

The prompt never grows with the length of the playthrough: it gets the latest scenes,
the most important older ones, the people/things present in the current scene, the
active promises, and the current personality.
"""

from __future__ import annotations

from html import escape

from aibecomeshuman.config import MemorySettings
from aibecomeshuman.memory.models import CharacterMemory, Entity, MemoryItem, entity_key

RECENT_CHANGES = 3


def select_items(memory: CharacterMemory, settings: MemorySettings) -> list[MemoryItem]:
    recent = memory.items[-settings.recent_scenes :] if settings.recent_scenes else []
    recent_ids = {i.id for i in recent}
    older = [i for i in memory.items if i.id not in recent_ids]
    important = sorted(older, key=lambda i: (-i.importance, -memory.items.index(i)))
    important = [i for i in important if i.importance >= 4][: settings.important_scenes]
    chosen = {i.id for i in recent} | {i.id for i in important}
    return [i for i in memory.items if i.id in chosen]  # chronological order


def select_entities(
    memory: CharacterMemory, scene_text: str, settings: MemorySettings
) -> list[Entity]:
    """Entities mentioned in the current scene first, then the most recently seen."""
    haystack = f" {entity_key(scene_text)} "
    present = [e for k, e in memory.entities.items() if k and f" {k} " in haystack]
    others = [e for e in memory.entities.values() if e not in present]
    order = {sid: n for n, sid in enumerate(memory.scenes_played)}
    others.sort(key=lambda e: order.get(e.last_scene, -1), reverse=True)
    return (present + others)[: settings.max_entities]


def render_memory(memory: CharacterMemory, scene_text: str, settings: MemorySettings) -> str:
    if not memory.items and not memory.entities and not memory.commitments:
        return "<memory>You have no memories from before this scene.</memory>"
    lines = ["<memory>"]
    items = select_items(memory, settings)
    if items:
        lines.append("<past_scenes>")
        for i in items:
            lines.append(
                f'<recollection id="{i.id}" scene="{escape(i.scene_title)}" '
                f'importance="{i.importance}">{escape(i.text, quote=False)}</recollection>'
            )
        lines.append("</past_scenes>")
    entities = select_entities(memory, scene_text, settings)
    if entities:
        lines.append("<known>")
        for e in entities:
            facts = "; ".join(e.facts[-settings.max_facts_per_entity :])
            feeling = f" Feeling: {e.feeling}." if e.feeling else ""
            lines.append(
                f'<entity name="{escape(e.name)}" kind="{e.kind}" trust="{e.trust:+d}">'
                f"{escape(facts, quote=False)}{escape(feeling, quote=False)}</entity>"
            )
        lines.append("</known>")
    active = memory.active_commitments
    if active:
        lines.append("<promises>")
        for c in active:
            lines.append(
                f'<promise id="{c.id}" to="{escape(c.to)}">{escape(c.text, quote=False)}</promise>'
            )
        lines.append("</promises>")
    lines.append("</memory>")
    return "\n".join(lines)


def render_personality(memory: CharacterMemory) -> str:
    p = memory.personality
    lines = [f'<personality version="{p.version}">', "Your traits right now (0-10):"]
    for name, trait in p.traits.items():
        meaning = f" ({trait.meaning})" if trait.meaning else ""
        lines.append(f"- {name}{meaning}: {trait.value:g}")
    if p.self_description:
        lines.append(f"How you see yourself now: {p.self_description}")
    if p.history:
        lines.append("How you have changed recently:")
        for ch in p.history[-RECENT_CHANGES:]:
            lines.append(f"- {ch.trait} {ch.old:g} -> {ch.new:g}: {ch.reason}")
    lines.append(
        "Act according to these current values. They outrank the starting values implied "
        "by your character sheet."
    )
    lines.append("</personality>")
    return "\n".join(lines)
