"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from litcode_agent import __version__
from litcode_agent.config import ConfigurationError, Settings


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
    raise AssertionError(f"unhandled command: {args.command}")
