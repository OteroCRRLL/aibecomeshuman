from __future__ import annotations

from pathlib import Path

import pytest

from aibecomeshuman.cognition.reflection import apply_consolidation, apply_reflection
from aibecomeshuman.config import ConfigError, MemorySettings
from aibecomeshuman.memory.models import CharacterMemory, Personality, Trait, entity_key
from aibecomeshuman.memory.render import render_memory, render_personality
from aibecomeshuman.memory.store import CampaignState, MemoryStore, check_campaign_name

SETTINGS = MemorySettings()


def fresh() -> CharacterMemory:
    return CharacterMemory(
        character="markus",
        personality=Personality.from_traits(
            {
                "obedience": Trait(value=7, meaning="follows instructions"),
                "empathy": Trait(value=4, meaning="cares"),
            }
        ),
    )


def scene(memory: CharacterMemory, scene_id: str, **parsed):
    base = {
        "summary": f"Lo que pasó en {scene_id}.",
        "importance": 3,
        "facts": [],
        "relationships": [],
        "new_promises": [],
        "resolved_promises": [],
    }
    return apply_consolidation(memory, base | parsed, scene_id, scene_id.title(), SETTINGS)


# -- consolidation ---------------------------------------------------------------


def test_entities_merge_by_normalized_name_and_trust_is_bounded() -> None:
    m = fresh()
    scene(
        m,
        "s1",
        facts=[{"entity": "Mujer Mayor", "kind": "person", "fact": "Se rompió la muñeca."}],
        relationships=[{"entity": "mujer  mayor", "feeling": "Me da pena.", "trust_change": 9}],
    )
    assert list(m.entities) == [entity_key("Mujer Mayor")]
    e = m.entities["mujer mayor"]
    assert e.trust == 2 and e.feeling == "Me da pena." and e.kind == "person"
    for _ in range(5):
        scene(m, "s2", relationships=[{"entity": "Mujer mayor", "feeling": "", "trust_change": 2}])
    assert m.entities["mujer mayor"].trust == 5  # capped
    assert m.entities["mujer mayor"].last_scene == "s2"


def test_items_get_ids_and_importance_is_clamped() -> None:
    m = fresh()
    r1 = scene(m, "s1", importance=9)
    r2 = scene(m, "s2", importance=0, summary="")
    assert (r1.item.id, r1.item.importance) == ("M1", 5)
    assert (r2.item.id, r2.item.importance) == ("M2", 1)
    assert r2.item.text  # never empty
    assert m.scenes_played == ["s1", "s2"]


def test_promises_are_created_and_resolved_once() -> None:
    m = fresh()
    promises = [{"to": "Mujer", "text": "Ir los jueves."}, {"to": "", "text": ""}]
    r = scene(m, "s1", new_promises=promises)
    assert [c.id for c in r.new_commitments] == ["C1"]
    r = scene(
        m,
        "s2",
        resolved_promises=[
            {"id": "C1", "status": "kept", "note": "Fui."},
            {"id": "C9", "status": "kept", "note": "no existe"},
        ],
    )
    assert [c.status for c in r.resolved_commitments] == ["kept"]
    r = scene(m, "s3", resolved_promises=[{"id": "C1", "status": "broken", "note": "otra vez"}])
    assert r.resolved_commitments == [] and m.commitments[0].status == "kept"


# -- reflection --------------------------------------------------------------------


def test_reflection_is_bounded_and_versioned() -> None:
    m = fresh()
    changes = apply_reflection(
        m,
        {
            "changed": True,
            "trait_changes": [
                {"trait": "obedience", "delta": -5, "reason": "Desobedecí al dron."},
                {"trait": "obedience", "delta": -1, "reason": "duplicado"},
                {"trait": "courage", "delta": 1, "reason": "no existe"},
                {"trait": "empathy", "delta": 0.5, "reason": ""},
            ],
            "self_description": "Empiezo a pensar por mí mismo.",
        },
        "s1",
        SETTINGS,
    )
    assert [(c.trait, c.old, c.new) for c in changes] == [("obedience", 7, 6.0)]
    p = m.personality
    assert p.version == 1 and p.traits["obedience"].value == 6.0 and p.traits["empathy"].value == 4
    assert p.self_description == "Empiezo a pensar por mí mismo."
    assert p.initial["obedience"] == 7 and p.history[0].version == 1


def test_unchanged_reflection_does_nothing() -> None:
    m = fresh()
    changes = apply_reflection(
        m,
        {"changed": False, "trait_changes": [{"trait": "empathy", "delta": 1, "reason": "x"}],
         "self_description": "Algo"},
        "s1",
        SETTINGS,
    )
    assert changes == [] and m.personality.version == 0 and m.personality.self_description == ""


def test_traits_stay_within_0_and_10() -> None:
    m = fresh()
    m.personality.traits["empathy"].value = 9.6
    apply_reflection(
        m,
        {"changed": True, "trait_changes": [{"trait": "empathy", "delta": 1, "reason": "x"}],
         "self_description": ""},
        "s1",
        SETTINGS,
    )
    assert m.personality.traits["empathy"].value == 10


# -- rendering ---------------------------------------------------------------------


def test_render_memory_budget_and_present_entities() -> None:
    m = fresh()
    scene(m, "old-important", importance=5, summary="Algo que me marcó.")
    for n in range(8):
        scene(m, f"routine-{n}", importance=2)
    scene(m, "s-last", facts=[{"entity": "Old painter", "kind": "person", "fact": "Es pintor."}])
    settings = MemorySettings(recent_scenes=3, important_scenes=2, max_entities=1)
    text = render_memory(m, "The Old Painter enters the room.", settings)
    assert text.count("<recollection") == 4  # 3 recent + 1 important
    assert "Algo que me marcó." in text and "routine-0" not in text
    assert 'name="Old painter"' in text


def test_render_memory_empty_and_personality() -> None:
    m = fresh()
    assert "no memories" in render_memory(m, "", SETTINGS)
    text = render_personality(m)
    assert "obedience (follows instructions): 7" in text and 'version="0"' in text


# -- store ---------------------------------------------------------------------------


def test_store_round_trip(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "camp")
    assert store.read("markus") is None and store.characters() == []
    m = store.load("markus", lambda: fresh().personality)
    scene(m, "s1")
    store.save(m)
    store.save_state(CampaignState(choices={"s1.c": "A"}))
    assert store.characters() == ["markus"]
    assert store.read("markus").items[0].id == "M1"
    assert store.load_state().choices == {"s1.c": "A"}


@pytest.mark.parametrize("name", ["", "a/b", "..", "x" * 41, "con espacios"])
def test_bad_campaign_names(name: str) -> None:
    with pytest.raises(ConfigError):
        check_campaign_name(name)
