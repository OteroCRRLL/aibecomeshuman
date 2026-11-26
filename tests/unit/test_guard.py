from __future__ import annotations

import json

from aibecomeshuman.cognition.guard import Judge, Lexicon, Statement
from aibecomeshuman.config import ModelProfile, ProjectConfig
from aibecomeshuman.core.events import read_events
from aibecomeshuman.env.scenario import load_scenario
from aibecomeshuman.evals import run_suite
from aibecomeshuman.llm.fake import FakeProvider
from aibecomeshuman.play import ScenarioRunner
from aibecomeshuman.run.session import RunSession

from ..conftest import REPO_ROOT

FAKE = ModelProfile(name="fake", provider="fake", model="fake-1")


def test_lexicon_flags_only_words_never_met() -> None:
    lexicon = Lexicon(story=["Hank", "Leo"], meta=["final bueno", "QTE"])
    assert lexicon.unexplained("Creo que HANK me ayudará", known="") == ["Hank"]
    assert lexicon.unexplained("Hank está enfadado", known="Hank: ¿qué quieres?") == []
    assert lexicon.unexplained("Leonardo pinta", known="") == []  # whole words only
    assert lexicon.unexplained("Así llegaré al final bueno", known="") == ["final bueno"]


def test_repo_lexicon_loads() -> None:
    lexicon = Lexicon.load(REPO_ROOT / "config" / "guard.toml")
    assert "Hank" in lexicon.story and "QTE" in lexicon.meta


def _judge(claims: list[dict]) -> Judge:
    answer = {"statements": [{"id": "s1", "claims": claims}]}
    return Judge(FakeProvider(FAKE, responses=[json.dumps(answer)]), "system")


def test_judge_flags_only_missing_or_contradicted_facts() -> None:
    statement = [Statement(id="s1", kind="stance", text="...")]
    ok = _judge([{"claim": "he seems angry", "status": "interpretation"}]).review("r", statement)
    assert ok[0].grounded
    bad = _judge(
        [
            {"claim": "he seems angry", "status": "interpretation"},
            {"claim": "I lied to him", "status": "contradicts_record"},
        ]
    ).review("r", statement)
    assert not bad[0].grounded and bad[0].unsupported == ["I lied to him (contradicts record)"]


def test_ungrounded_personality_change_is_rejected(project: ProjectConfig) -> None:
    scenario = load_scenario(REPO_ROOT / "scenarios" / "tram-stop.toml")
    with RunSession(project, "offline") as session:
        runner = ScenarioRunner(session, scenario)
        verdict = {
            "statements": [
                {"id": "trait-empathy", "claims": [{"claim": "x", "status": "not_in_record"}]},
                {"id": "trait-obedience", "claims": [{"claim": "y", "status": "in_record"}]},
            ]
        }
        runner.judge = Judge(FakeProvider(FAKE, responses=[json.dumps(verdict)]), "system")
        proposal = {
            "changed": True,
            "trait_changes": [
                {"trait": "empathy", "delta": 1, "reason": "algo que no pasó"},
                {"trait": "obedience", "delta": -1, "reason": "desobedecí al dron"},
            ],
            "self_description": "Recuerdo a Hank.",  # a name never met
        }
        reviewed = runner._review_reflection(proposal)
    assert [c["trait"] for c in reviewed["trait_changes"]] == ["obedience"]
    assert reviewed["self_description"] == ""
    types = [e.type for e in read_events(session.dir / "events.jsonl")]
    assert "personality.rejected" in types and "guard.flag" in types


def test_eval_suite_runs_offline(project: ProjectConfig) -> None:
    with RunSession(project, "offline") as session:
        results = run_suite(session, REPO_ROOT / "evals" / "suite.toml")
    assert [r.scenario for r in results] == ["canary-rooftop", "canary-kitchen", "canary-studio"]
    assert all(r.decisions >= 1 for r in results)
    assert all(r.clean for r in results)  # the fake model never uses outside words
    types = [e.type for e in read_events(session.dir / "events.jsonl")]
    assert types.count("eval.case") == 3 and "guard.audit" in types
