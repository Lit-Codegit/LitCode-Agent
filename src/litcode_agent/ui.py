"""基于 Rich 的轻量终端交互层。"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from litcode_agent.agent import AgentEvent
from litcode_agent.config import Settings


class TerminalUI:
    def __init__(
        self,
        console: Console | None = None,
        input_fn: Callable[[str], str] = input,
    ) -> None:
        self.console = console or Console()
        self.input_fn = input_fn

    def show_banner(self, settings: Settings, model: str) -> None:
        content = Text()
        content.append("工作区  ", style="bold")
        content.append(str(settings.workspace))
        content.append("\n配置档  ", style="bold")
        content.append(settings.model_profile)
        content.append("\n模型    ", style="bold")
        content.append(model)
        content.append("\n命令    ", style="bold")
        content.append("/help  /model  /clear  /exit", style="cyan")
        self.console.print(
            Panel(content, title="[bold cyan]LitCode Agent[/bold cyan]", border_style="cyan")
        )

    def show_help(self) -> None:
        table = Table(title="交互命令", show_header=True)
        table.add_column("命令", style="cyan", no_wrap=True)
        table.add_column("作用")
        table.add_row("/help", "显示帮助")
        table.add_row("/model", "查询 API 并选择当前模型")
        table.add_row("/clear", "结束当前上下文并开始新会话")
        table.add_row("/exit", "退出")
        self.console.print(table)

    def prompt(self) -> str:
        return self.input_fn("你 > ")

    def confirm_command(self, command: str) -> bool:
        self.console.print(
            Panel(command, title="[bold yellow]危险命令请求[/bold yellow]", border_style="yellow")
        )
        try:
            answer = self.input_fn("允许执行？[y/N] ")
        except EOFError:
            return False
        return answer.strip().lower() in {"y", "yes"}

    def show_assistant(self, content: str) -> None:
        self.console.print(
            Panel(Markdown(content), title="[bold green]LitCode[/bold green]", border_style="green")
        )

    def show_user(self, content: str) -> None:
        self.console.print(
            Panel(content, title="[bold blue]你[/bold blue]", border_style="blue")
        )

    def show_info(self, message: str) -> None:
        self.console.print(f"[cyan]•[/cyan] {message}")

    def show_error(self, message: str) -> None:
        self.console.print(f"[bold red]错误：[/bold red]{message}")

    def show_models(self, models: Sequence[str], current: str) -> None:
        table = Table(title="API 可用模型", show_lines=False)
        table.add_column("序号", justify="right", style="dim")
        table.add_column("模型 ID")
        table.add_column("当前", justify="center")
        for index, model in enumerate(models, start=1):
            table.add_row(str(index), model, "✓" if model == current else "")
        self.console.print(table)

    def choose_model(self, models: Sequence[str], current: str) -> str:
        if not models:
            self.show_info("API 没有返回可选模型，继续使用当前模型。")
            return current
        self.show_models(models, current)
        try:
            answer = self.input_fn("选择序号（直接回车保持当前模型）：").strip()
        except EOFError:
            return current
        if not answer:
            return current
        try:
            index = int(answer)
        except ValueError:
            self.show_error("请输入模型序号。")
            return current
        if not 1 <= index <= len(models):
            self.show_error("模型序号超出范围。")
            return current
        return models[index - 1]

    def handle_event(self, event: AgentEvent) -> None:
        if event.kind in {"compaction_start", "compaction_end"}:
            if event.is_error:
                self.show_error(event.content or "自动上下文压缩失败")
            else:
                self.show_info(event.content or "自动上下文压缩完成")
            return
        if event.kind == "model_start":
            self.console.print(
                f"[dim]第 {event.iteration} 轮 · 正在请求模型…[/dim]"
            )
            return
        if event.kind in {"model_delta", "model_end"}:
            return
        if event.kind == "hook_result":
            self._show_hook(event)
            return
        assert event.tool_call is not None
        if event.kind == "tool_start":
            arguments = _pretty_json(event.tool_call.arguments)
            self.console.print(
                Panel(
                    arguments,
                    title=f"[bold magenta]工具 · {event.tool_call.name}[/bold magenta]",
                    border_style="magenta",
                )
            )
            return
        status = "失败" if event.is_error else "完成"
        color = "red" if event.is_error else "green"
        self.console.print(
            Panel(
                _preview(event.content or ""),
                title=f"[{color}]工具结果 · {status}[/{color}]",
                border_style=color,
            )
        )

    def _show_hook(self, event: AgentEvent) -> None:
        assert event.hook_execution is not None
        execution = event.hook_execution
        if execution.return_code == 0 and not execution.timed_out:
            self.console.print(
                f"[dim]hook {execution.event} · 完成 · {execution.command}[/dim]"
            )
            return
        status = "超时" if execution.timed_out else f"退出码 {execution.return_code}"
        self.console.print(
            Panel(
                _preview(execution.stderr or execution.stdout),
                title=f"[red]hook {execution.event} · {status}[/red]",
                border_style="red",
            )
        )


def _pretty_json(raw: str) -> str:
    try:
        return json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return raw


def _preview(content: str, limit: int = 1_600) -> str:
    if len(content) <= limit:
        return content or "（无输出）"
    return f"{content[:limit]}\n… 已省略 {len(content) - limit} 个字符"
