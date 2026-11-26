"""Línea de comandos `aibh`.

    aibh check                          valida la configuración y lista modelos y runs
    aibh run -p offline                 run vacía: crea manifiesto y registro de eventos
    aibh ping -p offline -r decider     run con una sola llamada al modelo de un rol
    aibh events runs/<carpeta>          muestra el registro de eventos de una run
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

PING_SYSTEM = "Responde en una sola frase corta, en español."
PING_TEXT = "Prueba de conexión. Di qué modelo eres y que la conexión funciona."


def _load(args: argparse.Namespace) -> ProjectConfig:
    return load_project(Path(args.root) if args.root else None)


def cmd_check(args: argparse.Namespace) -> int:
    project = _load(args)
    print(f"Proyecto: {project.root}")
    print("\nModelos:")
    for name, m in sorted(project.models.items()):
        print(f"  {name:<16} {m.provider:<10} {m.model}  (effort={m.effort or '-'})")
    print("\nRuns:")
    for name, r in sorted(project.runs.items()):
        print(f"  {name:<16} personaje={r.character} directiva={r.directive}")
        for role, model in r.roles.items():
            print(f"      {role:<13} -> {model}")
    print("\nConfiguración válida.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    project = _load(args)
    with RunSession(project, args.profile) as session:
        pass
    print(f"Run creada: {session.dir}")
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
            print(f"Error del modelo ({provider.profile.name}): {e}", file=sys.stderr)
            session.finish("error", error=str(e))
            return 1
    print(f"[{response.provider}/{response.model}] {response.text}")
    print(
        f"latencia={response.latency_ms:.0f} ms  "
        f"tokens in/out={response.usage.input_tokens}/{response.usage.output_tokens}"
    )
    print(f"Run: {session.dir}")
    return 0


def cmd_events(args: argparse.Namespace) -> int:
    path = Path(args.run_dir)
    if path.is_dir():
        path = path / "events.jsonl"
    if not path.is_file():
        print(f"No existe {path}", file=sys.stderr)
        return 1
    for event in read_events(path):
        if args.type and event.type != args.type:
            continue
        ts = event.ts.strftime("%H:%M:%S.%f")[:-3]
        print(f"#{event.seq:<5} {ts}  {event.type:<16} [{event.source}] {event.payload}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aibh", description="aibecomeshuman")
    parser.add_argument("--root", help="raíz del proyecto (por defecto, se busca desde aquí)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="valida la configuración").set_defaults(func=cmd_check)

    run = sub.add_parser("run", help="crea una run vacía")
    run.add_argument("-p", "--profile", default="default")
    run.set_defaults(func=cmd_run)

    ping = sub.add_parser("ping", help="una llamada de prueba al modelo de un rol")
    ping.add_argument("-p", "--profile", default="default")
    ping.add_argument("-r", "--role", default="decider")
    ping.add_argument("text", nargs="?", help="texto a enviar (opcional)")
    ping.set_defaults(func=cmd_ping)

    events = sub.add_parser("events", help="muestra el registro de eventos de una run")
    events.add_argument("run_dir")
    events.add_argument("-t", "--type", help="filtra por tipo de evento")
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
