"""`aibh` command line.

    aibh check                          validate the configuration and list models and runs
    aibh run -p offline                 empty run: creates the manifest and the event log
    aibh ping -p offline -r decider     run with a single call to a role's model
    aibh events runs/<folder>           print a run's event log
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from aibecomeshuman.config import ConfigError, ProjectConfig, load_project
from aibecomeshuman.core.events import read_events
from aibecomeshuman.llm.types import LLMError, LLMRequest, Message
from aibecomeshuman.run.session import RunSession

PING_SYSTEM = "Answer in one short sentence."
PING_TEXT = "Connection test. Say which model you are and that the connection works."


def _load(args: argparse.Namespace) -> ProjectConfig:
    return load_project(Path(args.root) if args.root else None)


def cmd_check(args: argparse.Namespace) -> int:
    project = _load(args)
    print(f"Project: {project.root}")
    print("\nModels:")
    for name, m in sorted(project.models.items()):
        print(f"  {name:<16} {m.provider:<10} {m.model}  (effort={m.effort or '-'})")
    print("\nRuns:")
    for name, r in sorted(project.runs.items()):
        print(f"  {name:<16} character={r.character} directive={r.directive}")
        for role, model in r.roles.items():
            print(f"      {role:<13} -> {model}")
    print("\nConfiguration is valid.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    project = _load(args)
    with RunSession(project, args.profile) as session:
        pass
    print(f"Run created: {session.dir}")
    return 0


def cmd_ping(args: argparse.Namespace) -> int:
    project = _load(args)
    session = RunSession(project, args.profile)
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
        f"tokens in/out={response.usage.input_tokens}/{response.usage.output_tokens}"
    )
    print(f"Run: {session.dir}")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aibh", description="aibecomeshuman")
    parser.add_argument("--root", help="project root (searched upwards from here by default)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="validate the configuration").set_defaults(func=cmd_check)

    run = sub.add_parser("run", help="create an empty run")
    run.add_argument("-p", "--profile", default="default")
    run.set_defaults(func=cmd_run)

    ping = sub.add_parser("ping", help="one test call to a role's model")
    ping.add_argument("-p", "--profile", default="default")
    ping.add_argument("-r", "--role", default="decider")
    ping.add_argument("text", nargs="?", help="text to send (optional)")
    ping.set_defaults(func=cmd_ping)

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
