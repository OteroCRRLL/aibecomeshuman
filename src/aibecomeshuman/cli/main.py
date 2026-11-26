"""`aibh` command line.

    aibh check                              validate the configuration and list models and runs
    aibh run -p offline                     empty run: creates the manifest and the event log
    aibh ping -p offline -r decider         run with a single call to a role's model
    aibh play scenarios/<file>.toml         play a scenario (default run profile)
    aibh play scenarios/chains/<x>.toml -c test   play a chain, memory kept in campaigns/test
    aibh perceive data/recordings/<x>.mkv   turn a game recording into what the agent sees
    aibh eval                               anti-metagaming evaluation (canary scenes)
    aibh budget                             show spend against the budget caps
    aibh ui                                 open the local control panel in the browser
    aibh events runs/<folder>               print a run's event log

Runs that use locked (expensive) models need --unlock-premium.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from aibecomeshuman.budget import project_spent
from aibecomeshuman.config import ConfigError, ProjectConfig, load_project
from aibecomeshuman.core.events import read_events
from aibecomeshuman.env.scenario import load_playlist
from aibecomeshuman.envfile import load_dotenv
from aibecomeshuman.llm.types import LLMError, LLMRequest, Message
from aibecomeshuman.play import play_scenarios
from aibecomeshuman.run.session import RunSession

PING_SYSTEM = "Answer in one short sentence."
PING_TEXT = "Connection test. Say which model you are and that the connection works."


def _load(args: argparse.Namespace) -> ProjectConfig:
    project = load_project(Path(args.root) if args.root else None)
    load_dotenv(project.root / ".env")
    return project


def _session(project: ProjectConfig, args: argparse.Namespace) -> RunSession:
    return RunSession(project, args.profile, unlock_premium=args.unlock_premium)


def cmd_check(args: argparse.Namespace) -> int:
    project = _load(args)
    print(f"Project: {project.root}")
    print("\nModels:")
    for name, m in sorted(project.models.items()):
        price = (
            f"${m.price_input_per_mtok:g}/${m.price_output_per_mtok:g} per MTok"
            if m.is_paid
            else "free"
        )
        lock = "  LOCKED" if m.locked else ""
        print(f"  {name:<16} {m.provider:<10} {m.model:<18} {price}{lock}")
    print("\nRuns:")
    for name, r in sorted(project.runs.items()):
        print(f"  {name:<16} characters={','.join(r.characters)} directive={r.directive}")
        for role, model in r.roles.items():
            print(f"      {role:<13} -> {model}")
    b = project.app.budget
    print(f"\nBudget caps: project ${b.project_cap_usd:.2f}, per run ${b.run_cap_usd:.2f}")
    print("\nConfiguration is valid.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    project = _load(args)
    with _session(project, args) as session:
        pass
    print(f"Run created: {session.dir}")
    return 0


def cmd_ping(args: argparse.Namespace) -> int:
    project = _load(args)
    session = _session(project, args)
    with session:
        provider = session.llm(args.role)
        request = LLMRequest(
            purpose="ping",
            system=PING_SYSTEM,
            messages=[Message.user(args.text or PING_TEXT)],
        )
        try:
            response = provider.complete(request)
        except LLMError as e:
            print(f"Model error ({provider.profile.name}): {e}", file=sys.stderr)
            session.finish("error", error=str(e))
            return 1
    print(f"[{response.provider}/{response.model}] {response.text}")
    print(
        f"latency={response.latency_ms:.0f} ms  "
        f"tokens in/out={response.usage.input_tokens}/{response.usage.output_tokens}  "
        f"cost=${session.manifest.cost_usd:.6f}"
    )
    print(f"Run: {session.dir}")
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    project = _load(args)
    scenarios = load_playlist([Path(p) for p in args.scenarios])
    session = _session(project, args)
    with session:
        decisions = play_scenarios(
            session, scenarios, directive=args.directive, echo=print, campaign=args.campaign
        )
    fallbacks = sum(d.source == "fallback" for d in decisions)
    where = f"campaign '{args.campaign}'" if args.campaign else "this run only"
    print(
        f"\n{len(decisions)} decisions ({fallbacks} by fallback) · memory: {where} · "
        f"cost ${session.manifest.cost_usd:.6f} · run: {session.dir}"
    )
    return 0


def cmd_perceive(args: argparse.Namespace) -> int:
    from aibecomeshuman.perception.pipeline import finish_outputs, perceive, rebuild
    from aibecomeshuman.perception.video import parse_time

    project = _load(args)
    video = Path(args.video)
    start, end = parse_time(args.start) or 0.0, parse_time(args.end)
    session = _session(project, args)
    with session:
        if args.rebuild:
            result = rebuild(session, Path(args.rebuild), video, args.character)
        else:
            result = perceive(
                session, video, args.character, start, end, dry_run=args.dry_run, echo=print
            )
        evaluation = (
            None
            if args.dry_run
            else finish_outputs(session, video, result, args.character, start, end)
        )
    if args.dry_run:
        print(f"Dry run: nothing was sent to the model. Keyframes in {session.dir / 'frames'}")
        return 0
    choices = sum(1 for i in result.items if i.kind == "choice")
    lines = sum(1 for i in result.items if i.kind == "dialogue")
    print(
        f"\n{result.observed} frames observed ({result.failed} failed, "
        f"{result.hidden_frames} hidden from the agent) -> {lines} lines, {choices} choices"
    )
    if evaluation is not None:
        print("\nAgainst the labels:")
        for row in evaluation.summary():
            print(f"  {row}")
    print(f"\ncost ${session.manifest.cost_usd:.4f} · transcript: {session.dir / 'transcript.md'}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from aibecomeshuman.evals import run_suite, summary_rows

    project = _load(args)
    suite = Path(args.suite) if args.suite else project.root / "evals" / "suite.toml"
    session = _session(project, args)
    with session:
        results = run_suite(session, suite, echo=print if args.verbose else None)
    print("\nAnti-metagaming evaluation:")
    for row in summary_rows(results):
        print(f"  {row}")
    clean = sum(r.clean for r in results)
    print(f"\n{clean}/{len(results)} cases clean · cost ${session.manifest.cost_usd:.4f} · "
          f"run: {session.dir}")
    return 0


def cmd_budget(args: argparse.Namespace) -> int:
    project = _load(args)
    b = project.app.budget
    spent = project_spent(project.runs_dir)
    print(f"Project spend: ${spent:.4f} of ${b.project_cap_usd:.2f}")
    print(f"Per-run cap:   ${b.run_cap_usd:.2f}\n")
    rows = []
    for manifest in sorted(project.runs_dir.glob("*/manifest.json")):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("cost_usd"):
            rows.append((manifest.parent.name, data["cost_usd"]))
    for name, cost in rows[-args.last :]:
        print(f"  {name:<40} ${cost:.6f}")
    if not rows:
        print("  (no paid runs yet)")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    from aibecomeshuman.ui.server import serve

    serve(Path(args.root) if args.root else None, port=args.port, open_browser=not args.no_browser)
    return 0


def cmd_events(args: argparse.Namespace) -> int:
    path = Path(args.run_dir)
    if path.is_dir():
        path = path / "events.jsonl"
    if not path.is_file():
        print(f"{path} does not exist", file=sys.stderr)
        return 1
    for event in read_events(path):
        if args.type and event.type != args.type:
            continue
        ts = event.ts.strftime("%H:%M:%S.%f")[:-3]
        print(f"#{event.seq:<5} {ts}  {event.type:<16} [{event.source}] {event.payload}")
    return 0


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-p", "--profile", default="default")
    parser.add_argument(
        "--unlock-premium",
        action="store_true",
        help="allow locked (expensive) models in this run",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aibh", description="aibecomeshuman")
    parser.add_argument("--root", help="project root (searched upwards from here by default)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="validate the configuration").set_defaults(func=cmd_check)

    run = sub.add_parser("run", help="create an empty run")
    _add_run_args(run)
    run.set_defaults(func=cmd_run)

    ping = sub.add_parser("ping", help="one test call to a role's model")
    _add_run_args(ping)
    ping.add_argument("-r", "--role", default="decider")
    ping.add_argument("text", nargs="?", help="text to send (optional)")
    ping.set_defaults(func=cmd_ping)

    play = sub.add_parser("play", help="play scenarios or chains without the game")
    _add_run_args(play)
    play.add_argument(
        "scenarios", nargs="+", help="scenario and/or chain .toml files, played in order"
    )
    play.add_argument("-d", "--directive", help="override the run's directive")
    play.add_argument(
        "-c", "--campaign", help="keep memory in campaigns/<name> across runs (a playthrough)"
    )
    play.set_defaults(func=cmd_play)

    perceive = sub.add_parser("perceive", help="turn a game recording into what the agent sees")
    _add_run_args(perceive)
    perceive.add_argument("video", help="recording file (e.g. data/recordings/x.mkv)")
    perceive.add_argument("-c", "--character", default="connor", help="protagonist")
    perceive.add_argument("--from", dest="start", help="start time, e.g. 07:30")
    perceive.add_argument("--to", dest="end", help="end time, e.g. 09:00")
    perceive.add_argument(
        "--dry-run", action="store_true", help="only pick keyframes and estimate the cost (free)"
    )
    perceive.add_argument(
        "--rebuild",
        metavar="RUN_DIR",
        help="rebuild the transcript from that run's observations (no frames re-read; cheap)",
    )
    perceive.set_defaults(func=cmd_perceive)

    evaluate = sub.add_parser("eval", help="anti-metagaming evaluation with canary scenes")
    _add_run_args(evaluate)
    evaluate.add_argument("--suite", help="suite file (default: evals/suite.toml)")
    evaluate.add_argument("-v", "--verbose", action="store_true", help="print every step")
    evaluate.set_defaults(func=cmd_eval)

    budget = sub.add_parser("budget", help="show spend against the budget caps")
    budget.add_argument("--last", type=int, default=15, help="how many paid runs to list")
    budget.set_defaults(func=cmd_budget)

    ui = sub.add_parser("ui", help="open the local control panel in the browser")
    ui.add_argument("--port", type=int, default=8765)
    ui.add_argument("--no-browser", action="store_true", help="do not open the browser")
    ui.set_defaults(func=cmd_ui)

    events = sub.add_parser("events", help="print a run's event log")
    events.add_argument("run_dir")
    events.add_argument("-t", "--type", help="filter by event type")
    events.set_defaults(func=cmd_events)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, KeyError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
