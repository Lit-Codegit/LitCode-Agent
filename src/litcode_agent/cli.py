"""LitCode Agent 命令行入口。"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

from litcode_agent import __version__
from litcode_agent.agent import Agent
from litcode_agent.config import ConfigurationError, Settings
from litcode_agent.credentials import (
    CredentialError,
    save_api_key,
    validate_credential_name,
)
from litcode_agent.hooks import HookRunner
from litcode_agent.model import ModelError, OpenAIChatModel
from litcode_agent.prompt import PromptBuilder
from litcode_agent.scheduler import Scheduler, describe_task
from litcode_agent.session_runtime import SessionRuntime, SessionRuntimeError
from litcode_agent.session_store import SessionStore
from litcode_agent.skill_manager import SkillManagementError, SkillManager
from litcode_agent.skills import SkillCatalog
from litcode_agent.tools import build_default_registry
from litcode_agent.tools.base import ToolExecutionContext
from litcode_agent.tui import run_tui
from litcode_agent.ui import TerminalUI

COMMANDS = {"auth", "doctor", "models", "run", "chat", "schedule", "skill"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="litcode",
        usage="litcode [PATH] | litcode {auth,doctor,models,run,chat,schedule,skill} ...",
        description="一个透明、可解释的本地编程智能体。",
        epilog="不带参数时打开当前目录；传入路径时打开该目录的全屏 TUI。",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    auth = subparsers.add_parser("auth", help="管理用户级 API 凭据")
    auth_commands = auth.add_subparsers(dest="auth_command", required=True)
    auth_login = auth_commands.add_parser("login", help="安全保存 API Key")
    auth_login.add_argument(
        "credential",
        nargs="?",
        help="与模型配置 apiKeyEnv 相同的凭据名称",
    )
    auth_login.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="用于自动识别 apiKeyEnv 的工作区（默认：当前目录）",
    )

    skill = subparsers.add_parser("skill", help="管理标准 Agent Skills")
    skill.add_argument(
        "--workspace", type=Path, default=Path.cwd(), help="项目 Skill 所属工作区"
    )
    skill_commands = skill.add_subparsers(dest="skill_command", required=True)
    skill_list = skill_commands.add_parser("list", help="列出已发现的 Skill")
    skill_list.add_argument(
        "--scope", choices=("all", "project", "user"), default="all"
    )
    skill_create = skill_commands.add_parser("create", help="创建通用 Skill 骨架")
    skill_create.add_argument("name")
    skill_create.add_argument("--description", required=True)
    skill_create.add_argument(
        "--scope", choices=("project", "user"), default="project"
    )
    skill_create.add_argument(
        "--resources",
        action="append",
        choices=("scripts", "references", "assets"),
        default=[],
    )
    skill_install = skill_commands.add_parser("install", help="从目录或 Git 仓库安装 Skill")
    skill_install.add_argument("source")
    skill_install.add_argument("--name", help="来源包含多个 Skill 时选择名称")
    skill_install.add_argument(
        "--scope", choices=("project", "user"), default="project"
    )
    skill_validate = skill_commands.add_parser("validate", help="校验 Skill 目录和元数据")
    skill_validate.add_argument("target")
    skill_validate.add_argument(
        "--scope", choices=("all", "project", "user"), default="all"
    )
    skill_sync = skill_commands.add_parser("sync", help="同步到本机其他 Agent")
    skill_sync.add_argument("names", nargs="*")
    skill_sync.add_argument(
        "--scope", choices=("project", "user"), default="project"
    )
    skill_sync.add_argument(
        "--agent",
        action="append",
        choices=(
            "codex",
            "claude-code",
            "opencode",
            "cursor",
            "gemini-cli",
            "github-copilot",
        ),
        default=[],
    )

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
    schedule = subparsers.add_parser("schedule", help="运行或查看定时 Agent 任务")
    schedule_commands = schedule.add_subparsers(
        dest="schedule_command", required=True
    )
    schedule_tick = schedule_commands.add_parser(
        "tick", help="触发到期任务，供 launchd/systemd 调用"
    )
    schedule_tick.add_argument(
        "--wait-seconds", type=float, default=3600.0, help="等待 Agent 完成的上限"
    )
    _add_common_options(schedule_tick)
    schedule_serve = schedule_commands.add_parser(
        "serve", help="常驻监视到期任务，供操作系统保活"
    )
    schedule_serve.add_argument(
        "--poll-seconds",
        type=float,
        default=0.25,
        help="到期检查间隔（默认 0.25 秒）",
    )
    schedule_serve.add_argument(
        "--wait-seconds", type=float, default=3600.0, help="单批 Agent 完成等待上限"
    )
    _add_common_options(schedule_serve)
    schedule_list = schedule_commands.add_parser("list", help="列出持久化定时任务")
    schedule_list.add_argument("--all", action="store_true", help="包含已完成和已取消任务")
    _add_common_options(schedule_list)
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
    # Windows 默认把重定向输出按本地区码（cp1252 等）编码，中文 help 和
    # 状态文案会抛 UnicodeEncodeError；CLI 统一 UTF-8，不再随终端环境漂移。
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(_normalize_args(raw_args))
    terminal = ui or TerminalUI()
    if args.command == "auth":
        return _auth_login(args, terminal)
    if args.command == "skill":
        return _skill_command(args, terminal)
    if args.command == "chat":
        args.workspace = args.path or args.workspace or Path.cwd()
    settings = _load_settings(
        args,
        tui_mode=(
            args.command == "chat"
            or (args.command == "schedule" and args.schedule_command == "list")
        ),
    )

    if args.command == "schedule" and args.schedule_command == "list":
        assert settings.session_database is not None
        store = SessionStore(settings.session_database)
        try:
            tasks = store.scheduled_tasks(
                settings.workspace, include_inactive=args.all
            )
            if not tasks:
                terminal.show_info("当前工作区没有定时任务。")
            else:
                for task in tasks:
                    terminal.show_info(describe_task(task))
        finally:
            store.close()
        return 0

    if args.command == "doctor":
        terminal.console.print_json(
            json.dumps(settings.safe_summary(), ensure_ascii=False)
        )
        return 0

    model = OpenAIChatModel(settings)
    skills = SkillCatalog.discover(settings.workspace, settings.user_skill_root)
    if getattr(args, "model", None):
        model.select_model(args.model)

    if args.command == "models":
        try:
            terminal.show_models(model.list_models(), model.model)
        except ModelError as error:
            terminal.show_error(str(error))
            return 1
        return 0

    if args.command == "schedule" and args.schedule_command == "tick":
        return _schedule_tick(settings, model, skills, terminal, args.wait_seconds)
    if args.command == "schedule" and args.schedule_command == "serve":
        return _schedule_serve(
            settings,
            model,
            skills,
            terminal,
            args.poll_seconds,
            args.wait_seconds,
        )

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
        auto_compact_chars=settings.auto_compact_chars,
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


def _auth_login(args: argparse.Namespace, terminal: TerminalUI) -> int:
    try:
        credential = (
            validate_credential_name(args.credential)
            if args.credential
            else Settings.configured_api_key_name(args.workspace, os.environ)
        )
        key = getpass.getpass(f"输入 {credential}（不会回显）：")
        path = save_api_key(credential, key, os.environ)
    except (
        ConfigurationError,
        CredentialError,
        EOFError,
        KeyboardInterrupt,
    ) as error:
        terminal.show_error(str(error) or "凭据输入已取消")
        return 1
    terminal.show_info(f"凭据 {credential} 已保存到 {path}（权限 0600）。")
    return 0


def _skill_command(args: argparse.Namespace, terminal: TerminalUI) -> int:
    manager = SkillManager(args.workspace)
    try:
        if args.skill_command == "list":
            items = manager.list(args.scope)
            if not items:
                terminal.show_info("没有发现 Skill。")
            for item in items:
                terminal.show_info(
                    f"{item.skill.name} · {item.scope} · {item.skill.description}"
                )
            return 0
        if args.skill_command == "create":
            skill = manager.create(
                args.name,
                args.description,
                scope=args.scope,
                resources=args.resources,
            )
            terminal.show_info(f"已创建 Skill：{skill.root}")
            return 0
        if args.skill_command == "install":
            skill = manager.install(args.source, name=args.name, scope=args.scope)
            terminal.show_info(f"已安装 Skill：{skill.root}")
            return 0
        if args.skill_command == "validate":
            skill = manager.validate(args.target, args.scope)
            terminal.show_info(f"Skill 校验通过：{skill.name} · {skill.root}")
            return 0
        if args.skill_command == "sync":
            links = manager.sync(
                args.names, scope=args.scope, agents=args.agent
            )
            if not links:
                terminal.show_info("没有检测到需要同步的 Agent 目录。")
            for link in links:
                status = "已创建" if link.created else "已存在"
                terminal.show_info(
                    f"{status}：{link.agent}/{link.skill} -> {link.destination}"
                )
            return 0
        raise AssertionError(f"unhandled skill command: {args.skill_command}")
    except (OSError, SkillManagementError) as error:
        terminal.show_error(str(error))
        return 1


def _load_settings(
    args: argparse.Namespace, *, tui_mode: bool = False
) -> Settings:
    environ = dict(os.environ)
    if args.profile:
        environ["LITCODE_DEFAULT_MODEL"] = args.profile
    loader = Settings.load_tui if tui_mode else Settings.load
    try:
        return loader(args.workspace, environ)
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


def _schedule_tick(
    settings: Settings,
    model: OpenAIChatModel,
    skills: SkillCatalog,
    terminal: TerminalUI,
    wait_seconds: float,
    *,
    quiet_lock: bool = False,
) -> int:
    """Run one durable due batch; an active TUI owns and dispatches it instead."""

    if wait_seconds <= 0:
        terminal.show_error("--wait-seconds must be positive")
        return 1
    assert settings.session_database is not None
    store = SessionStore(settings.session_database)
    system_prompt = PromptBuilder(
        settings.workspace, settings.max_iterations, skills.metadata()
    ).build()
    try:
        runtime = SessionRuntime(store, settings.workspace, system_prompt=system_prompt)
    except SessionRuntimeError as error:
        store.close()
        if "另一个 LitCode 进程" in str(error):
            if not quiet_lock:
                terminal.show_info("工作区已由 TUI 运行；进程内调度器会负责触发。")
            return 0
        raise
    scheduler = Scheduler(store, runtime, settings.workspace)
    registry = build_default_registry(
        settings,
        skills=skills,
        store=store,
        runtime=runtime,
        scheduler=scheduler,
    )

    def session_factory(session_id: str, profile: str):
        info = store.session_info(session_id)
        selected_model = model.clone_for_model(info.model)

        def tool_context(current_id: str) -> ToolExecutionContext:
            current = store.session_info(current_id)
            return ToolExecutionContext(
                current_id,
                settings.workspace.resolve(),
                profile=current.profile,
                turn_id=current.active_turn_id,
                runtime=runtime,
            )

        agent = Agent(
            selected_model,
            registry,
            settings.max_iterations,
            hooks=HookRunner(settings.workspace, settings.hooks),
            system_prompt=system_prompt,
            store=store,
            model_name=info.model,
            workspace=settings.workspace,
            tool_context=tool_context,
            auto_compact_chars=settings.auto_compact_chars,
        )
        return agent.start_session(session_id)

    runtime.session_factory = session_factory
    try:
        targets = scheduler.dispatch_due()
        succeeded = all(runtime.wait_for_idle(target, wait_seconds) for target in targets)
        if targets:
            terminal.show_info(f"已触发 {len(targets)} 个定时 Agent 任务。")
        return 0 if succeeded else 1
    finally:
        scheduler.close()
        runtime.close()
        store.close()


def _schedule_serve(
    settings: Settings,
    model: OpenAIChatModel,
    skills: SkillCatalog,
    terminal: TerminalUI,
    poll_seconds: float,
    wait_seconds: float,
) -> int:
    """Stay idle without the workspace lock; acquire it only for a due batch."""

    if poll_seconds <= 0:
        terminal.show_error("--poll-seconds must be positive")
        return 1
    assert settings.session_database is not None
    terminal.show_info(
        f"定时 Agent 守护进程已启动，检查间隔 {poll_seconds:g} 秒。"
    )
    try:
        while True:
            store = SessionStore(settings.session_database)
            try:
                due = bool(
                    store.due_scheduled_tasks(settings.workspace, time.time(), limit=1)
                )
            finally:
                store.close()
            if due:
                result = _schedule_tick(
                    settings,
                    model,
                    skills,
                    terminal,
                    wait_seconds,
                    quiet_lock=True,
                )
                if result != 0:
                    return result
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        terminal.show_info("定时 Agent 守护进程已停止。")
        return 0
