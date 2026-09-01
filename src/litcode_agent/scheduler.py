"""Deterministic same-terminal actions for bounded orchestration runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, AbstractSet

from litcode_agent.orchestration import OrchestrationService, OrchestrationTask


ActionKind = Literal["wake_task", "resume_coordinator"]


@dataclass(frozen=True, slots=True)
class SchedulerAction:
    kind: ActionKind
    session_id: str
    pane_slot: int
    prompt: str
    task_id: str
    run_id: str


class LocalScheduler:
    """Choose explicit actions; the UI only executes returned wake requests."""

    def __init__(
        self,
        service: OrchestrationService,
        *,
        max_parallel_read_tasks: int = 2,
    ) -> None:
        self.service = service
        self.max_parallel_read_tasks = max_parallel_read_tasks

    def next_action(
        self,
        *,
        mounted: Mapping[str, int],
        busy: AbstractSet[str],
    ) -> SchedulerAction | None:
        runs = [run for run in self.service.active_runs() if run.status == "running"]
        tasks = [task for run in runs for task in self.service.tasks(run.id)]
        resumed = {
            event.task_id
            for run in runs
            for event in self.service.ledger(run.id)
            if event.kind == "coordinator_resumed" and event.task_id is not None
        }

        for task in tasks:
            if task.status not in {"completed", "blocked", "failed"}:
                continue
            if task.id in resumed:
                continue
            run = self.service.get_run(task.run_id)
            pane_slot = mounted.get(run.coordinator_session_id)
            if pane_slot is None or run.coordinator_session_id in busy:
                continue
            self.service.record_event(
                run.id,
                task_id=task.id,
                kind="coordinator_resumed",
                actor_session_id=None,
                source_session_id=task.target_session_id,
                target_session_id=run.coordinator_session_id,
                summary=f"任务 {task.id} 已报告，恢复协调者",
            )
            return SchedulerAction(
                "resume_coordinator",
                run.coordinator_session_id,
                pane_slot,
                _coordinator_prompt(task),
                task.id,
                run.id,
            )

        running = [task for task in tasks if task.status == "running"]
        write_running = any(task.write_policy == "workspace-write" for task in running)
        read_running = sum(task.write_policy == "none" for task in running)
        for task in tasks:
            if task.status != "queued":
                continue
            pane_slot = mounted.get(task.target_session_id)
            if pane_slot is None or task.target_session_id in busy:
                continue
            if task.write_policy == "workspace-write" and write_running:
                continue
            if task.write_policy == "none" and read_running >= self.max_parallel_read_tasks:
                continue
            started = self.service.start_task(task.id, task.target_session_id)
            return SchedulerAction(
                "wake_task",
                task.target_session_id,
                pane_slot,
                _task_prompt(started),
                task.id,
                task.run_id,
            )
        return None


def _task_prompt(task: OrchestrationTask) -> str:
    acceptance = "\n".join(f"- {item}" for item in task.acceptance) or "- 完成目标"
    paths = "、".join(task.allowed_paths) or "只读，不允许修改文件"
    return (
        f"你正在执行 LitCode 编排任务 {task.id}（角色：{task.role}）。\n"
        f"目标：{task.objective}\n验收条件：\n{acceptance}\n"
        f"允许路径：{paths}\n写策略：{task.write_policy}\n"
        "只按需读取其他会话上下文。完成、失败或受阻时，必须调用 "
        f"report_task(task_id=\"{task.id}\", ...) 提交结构化结果。"
    )


def _coordinator_prompt(task: OrchestrationTask) -> str:
    evidence = "\n".join(f"- {item}" for item in task.evidence) or "- 未提供"
    files = "、".join(task.changed_files) or "无"
    next_step = (
        "该结果来自 implementer；必须把相关规格、实现和测试路径委派给一个独立 "
        "reviewer，只读核验后才能结束编排。"
        if task.role == "implementer" and task.status == "completed"
        else "根据审查状态与证据决定结束编排或向用户报告阻塞。"
    )
    return (
        f"编排任务 {task.id} 已返回，状态：{task.status}。\n"
        f"摘要：{task.summary}\n证据：\n{evidence}\n变更文件：{files}\n"
        f"请作为协调者判断下一步：{next_step} 不要重复委派同一任务。"
    )
