"""LitCode Agent 命令行入口。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from litcode_agent import __version__
from litcode_agent.agent import Agent
from litcode_agent.config import ConfigurationError, Settings
from litcode_agent.hooks import HookRunner
from litcode_agent.model import ModelError, OpenAIChatModel
from litcode_agent.prompt import PromptBuilder
from litcode_agent.skills import SkillCatalog
from litcode_agent.tools import build_default_registry
from litcode_agent.tui import run_tui
from litcode_agent.ui import TerminalUI

COMMANDS = {"doctor", "models", "run", "chat"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="litcode",
        usage="litcode [PATH] | litcode {doctor,models,run,chat} ...",
        description="一个透明、可解释的本地编程智能体。",
        epilog="不带参数时打开当前目录；传入路径时打开该目录的全屏 TUI。",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="校验配置且不显示密钥")
    _add_common_options(doctor)

    models = subparsers.add_parser("models", help="查询当前 API 的可用模型")
    _add_common_options(models)

    run = subparsers.add_parser("run", help="执行一次编程任务")
    run.add_argument("task", help="交给 Agent 的编程任务")
    _add_common_options(run, allow_model_override=True)

    chat = subparsers.add_parser("chat", help="启动全屏 TUI 会话")
    chat.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="要打开的工作区（默认：当前目录）",
    )
    _add_common_options(
        chat,
        allow_model_override=True,
        workspace_default=None,
    )
    return parser


def _add_common_options(
    parser: argparse.ArgumentParser,
    *,
    allow_model_override: bool = False,
    workspace_default: Path | None = Path.cwd(),
) -> None:
    parser.add_argument(
        "--workspace",
        type=Path,
        default=workspace_default,
        help="Agent 可以访问的工作区（默认：当前目录）",
    )
    parser.add_argument(
        "--profile",
        help="覆盖 settings.json 中的 defaultModel 配置档",
    )
    if allow_model_override:
        parser.add_argument("--model", help="覆盖配置档中的具体模型 ID")


def main(
    argv: Sequence[str] | None = None,
    ui: TerminalUI | None = None,
) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(_normalize_args(raw_args))
    if args.command == "chat":
        args.workspace = args.path or args.workspace or Path.cwd()
    settings = _load_settings(args)
    terminal = ui or TerminalUI()

    if args.command == "doctor":
        terminal.console.print_json(
            json.dumps(settings.safe_summary(), ensure_ascii=False)
        )
        return 0

    model = OpenAIChatModel(settings)
    skills = SkillCatalog.discover(settings.workspace)
    if getattr(args, "model", None):
        model.select_model(args.model)

    if args.command == "models":
        try:
            terminal.show_models(model.list_models(), model.model)
        except ModelError as error:
            terminal.show_error(str(error))
            return 1
        return 0

    if args.command == "chat" and ui is None:
        return run_tui(settings, model)

    agent = Agent(
        model,
        build_default_registry(settings, terminal.confirm_command, skills),
        settings.max_iterations,
        terminal.handle_event,
        HookRunner(settings.workspace, settings.hooks),
        PromptBuilder(
            settings.workspace, settings.max_iterations, skills.metadata()
        ).build(),
    )
    if args.command == "run":
        terminal.show_user(args.task)
        try:
            result = agent.run(args.task)
        except ModelError as error:
            terminal.show_error(str(error))
            return 1
        terminal.show_assistant(result.output)
        if not result.succeeded:
            terminal.show_error(f"终止原因：{result.reason}")
            return 1
        return 0
    if args.command == "chat":
        return _chat(agent, model, settings, terminal)
    raise AssertionError(f"unhandled command: {args.command}")


def _load_settings(args: argparse.Namespace) -> Settings:
    environ = dict(os.environ)
    if args.profile:
        environ["LITCODE_DEFAULT_MODEL"] = args.profile
    try:
        return Settings.load(args.workspace, environ)
    except ConfigurationError as error:
        build_parser().error(str(error))


def _normalize_args(argv: list[str]) -> list[str]:
    """让 litcode [路径] 成为 chat 子命令的便捷入口。"""

    if not argv:
        return ["chat"]
    first = argv[0]
    if first in COMMANDS or first in {"-h", "--help", "--version"}:
        return argv
    return ["chat", *argv]


def _chat(
    agent: Agent,
    model: OpenAIChatModel,
    settings: Settings,
    ui: TerminalUI,
) -> int:
    ui.show_banner(settings, model.model)
    session = agent.start_session()
    while True:
        try:
            user_input = ui.prompt().strip()
        except (EOFError, KeyboardInterrupt):
            ui.console.print()
            session.close()
            ui.show_info("会话已结束。")
            return 0
        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            session.close()
            ui.show_info("会话已结束。")
            return 0
        if user_input == "/help":
            ui.show_help()
            continue
        if user_input in {"/model", "/models"}:
            try:
                selected = ui.choose_model(model.list_models(), model.model)
            except ModelError as error:
                ui.show_error(str(error))
                continue
            if selected != model.model:
                model.select_model(selected)
                ui.show_info(f"已切换到模型 {selected}；现有对话上下文保持不变。")
            continue
        if user_input == "/clear":
            session.close("user_clear", "")
            session = agent.start_session()
            ui.show_info("对话上下文已清空。")
            continue
        if user_input.startswith("/"):
            ui.show_error("未知命令；输入 /help 查看帮助。")
            continue

        ui.show_user(user_input)
        try:
            result = session.ask(user_input)
        except ModelError as error:
            ui.show_error(str(error))
            continue
        ui.show_assistant(result.output)
        if not result.succeeded:
            ui.show_error(f"本轮终止原因：{result.reason}")
