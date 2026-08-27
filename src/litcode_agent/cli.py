"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from litcode_agent import __version__
from litcode_agent.agent import Agent, AgentEvent
from litcode_agent.config import ConfigurationError, Settings
from litcode_agent.model import ModelError, OpenAIChatModel
from litcode_agent.tools import build_default_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="litcode",
        description="A small, transparent coding agent.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser(
        "doctor", help="validate configuration without revealing secrets"
    )
    doctor.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="workspace the agent may access (default: current directory)",
    )

    run = subparsers.add_parser("run", help="run the coding agent on a task")
    run.add_argument("task", help="programming task for the agent")
    run.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="workspace the agent may access (default: current directory)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        try:
            settings = Settings.from_env(args.workspace)
        except ConfigurationError as error:
            build_parser().error(str(error))
        print(json.dumps(settings.safe_summary(), indent=2, ensure_ascii=False))
        return 0
    if args.command == "run":
        try:
            settings = Settings.from_env(args.workspace)
        except ConfigurationError as error:
            build_parser().error(str(error))
        agent = Agent(
            OpenAIChatModel(settings),
            build_default_registry(settings, _confirm_command),
            settings.max_iterations,
            _print_event,
        )
        try:
            result = agent.run(args.task)
        except ModelError as error:
            print(f"model error: {error}", file=sys.stderr)
            return 1
        print(result.output)
        if not result.succeeded:
            print(f"termination: {result.reason}", file=sys.stderr)
            return 1
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


def _confirm_command(command: str) -> bool:
    print(f"\nDangerous command requested:\n  {command}", file=sys.stderr)
    try:
        answer = input("Allow this command? [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in {"y", "yes"}


def _print_event(event: AgentEvent) -> None:
    if event.kind == "model_start":
        print(f"[iteration {event.iteration}] asking model", file=sys.stderr)
        return
    assert event.tool_call is not None
    if event.kind == "tool_start":
        print(
            f"[iteration {event.iteration}] tool {event.tool_call.name} "
            f"{event.tool_call.arguments}",
            file=sys.stderr,
        )
        return
    status = "error" if event.is_error else "ok"
    print(f"[iteration {event.iteration}] tool result: {status}", file=sys.stderr)
