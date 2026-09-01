"""Bounded, explicit orchestration state above independent Agent sessions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, TYPE_CHECKING
import time
import uuid

if TYPE_CHECKING:
    from litcode_agent.session_store import SessionStore


RunStatus = Literal[
    "proposed", "running", "paused", "completed", "failed", "cancelled"
]
TaskStatus = Literal[
    "queued", "running", "completed", "blocked", "failed", "cancelled", "interrupted"
]
TaskRole = Literal["implementer", "reviewer"]
WritePolicy = Literal["none", "workspace-write"]
ReportStatus = Literal["completed", "blocked", "failed"]
FinishStatus = Literal["completed", "failed", "cancelled"]


class OrchestrationError(ValueError):
    """A visible protocol or budget violation."""


@dataclass(frozen=True, slots=True)
class OrchestrationRun:
    id: str
    workspace: str
    coordinator_session_id: str
    objective: str
    status: RunStatus
    max_tasks: int
    model_requests: int
    max_model_requests: int
    deadline: float
    created_at: float
    updated_at: float


@dataclass(frozen=True, slots=True)
class OrchestrationTask:
    id: str
    run_id: str
    parent_task_id: str | None
    source_session_id: str
    target_session_id: str
    role: TaskRole
    objective: str
    acceptance: tuple[str, ...]
    allowed_paths: tuple[str, ...]
    write_policy: WritePolicy
    status: TaskStatus
    attempt: int
    hop: int
    summary: str
    evidence: tuple[str, ...]
    changed_files: tuple[str, ...]
    created_at: float
    updated_at: float


@dataclass(frozen=True, slots=True)
class OrchestrationEvent:
    id: int
    run_id: str
    task_id: str | None
    kind: str
    actor_session_id: str | None
    source_session_id: str | None
    target_session_id: str | None
    summary: str
    created_at: float


class OrchestrationService:
    """Validate transitions; persistence never decides what an Agent should do."""

    def __init__(
        self,
        store: SessionStore,
        workspace: Path,
        *,
        max_tasks: int = 8,
        max_model_requests: int = 12,
        timeout_seconds: float = 600.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not 1 <= max_tasks <= 8:
            raise ValueError("max_tasks must be between 1 and 8")
        self.store = store
        self.workspace = workspace.resolve()
        self.max_tasks = max_tasks
        if not 1 <= max_model_requests <= 100:
            raise ValueError("max_model_requests must be between 1 and 100")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.max_model_requests = max_model_requests
        self.timeout_seconds = timeout_seconds
        self.clock = clock
        self.store.recover_interrupted_orchestrations(self.workspace)

    def start_run(
        self, coordinator_session_id: str, objective: str
    ) -> OrchestrationRun:
        objective = _bounded_text(objective, "objective", 4_000)
        self._require_session(coordinator_session_id)
        active = self.store.active_orchestration_run(
            self.workspace, coordinator_session_id
        )
        if active is not None:
            raise OrchestrationError("协调会话已经有未结束的编排")
        now = self.clock()
        run = OrchestrationRun(
            id=f"R-{uuid.uuid4().hex[:8].upper()}",
            workspace=str(self.workspace),
            coordinator_session_id=coordinator_session_id,
            objective=objective,
            status="proposed",
            max_tasks=self.max_tasks,
            model_requests=0,
            max_model_requests=self.max_model_requests,
            deadline=now + self.timeout_seconds,
            created_at=now,
            updated_at=now,
        )
        self.store.create_orchestration_run(run)
        self.store.add_orchestration_event(
            run.id,
            kind="run_proposed",
            actor_session_id=coordinator_session_id,
            source_session_id=coordinator_session_id,
            target_session_id=None,
            summary=objective,
        )
        return run

    def approve_run(self, run_id: str, actor_session_id: str) -> OrchestrationRun:
        run = self.get_run(run_id)
        self._require_coordinator(run, actor_session_id)
        if run.status != "proposed":
            raise OrchestrationError("只有 proposed 编排可以批准")
        updated = self.store.update_orchestration_run_status(run.id, "running")
        self.store.add_orchestration_event(
            run.id,
            kind="run_approved",
            actor_session_id=actor_session_id,
            source_session_id=actor_session_id,
            target_session_id=None,
            summary="用户已批准受限编排",
        )
        return updated

    def before_model_request(self, session_id: str) -> None:
        task = self.store.running_task_for_session(session_id)
        run = (
            self.get_run(task.run_id)
            if task is not None
            else self.store.active_orchestration_run(self.workspace, session_id)
        )
        if run is None or run.status != "running":
            return
        now = self.clock()
        if now > run.deadline:
            reason = "编排已超过截止时间，run 已暂停"
        elif run.model_requests >= run.max_model_requests:
            reason = "编排已达到模型请求上限，run 已暂停"
        else:
            self.store.consume_orchestration_model_request(run.id, now)
            return
        self.store.update_orchestration_run_status(run.id, "paused")
        self.store.add_orchestration_event(
            run.id,
            kind="run_paused",
            actor_session_id=None,
            source_session_id=session_id,
            target_session_id=run.coordinator_session_id,
            summary=reason,
        )
        raise OrchestrationError(reason)

    def delegate(
        self,
        run_id: str,
        source_session_id: str,
        target_alias: str,
        *,
        role: TaskRole,
        objective: str,
        acceptance: tuple[str, ...],
        allowed_paths: tuple[str, ...],
        write_policy: WritePolicy,
    ) -> OrchestrationTask:
        run = self.get_run(run_id)
        self._require_coordinator(run, source_session_id)
        if run.status != "running":
            raise OrchestrationError("编排未处于 running 状态")
        tasks = self.store.orchestration_tasks(run.id)
        if len(tasks) >= run.max_tasks:
            self.store.update_orchestration_run_status(run.id, "paused")
            self.store.add_orchestration_event(
                run.id,
                kind="run_paused",
                actor_session_id=source_session_id,
                source_session_id=source_session_id,
                target_session_id=None,
                summary="达到任务上限，需要用户决定是否继续",
            )
            raise OrchestrationError("已达到编排任务上限，run 已暂停")
        if role not in {"implementer", "reviewer"}:
            raise OrchestrationError("role 必须是 implementer 或 reviewer")
        if write_policy not in {"none", "workspace-write"}:
            raise OrchestrationError("未知 write_policy")
        if role == "reviewer" and write_policy != "none":
            raise OrchestrationError("reviewer 必须使用只读 write_policy")
        target_session_id = self.store.session_id_for_alias(
            self.workspace, target_alias
        )
        if target_session_id == source_session_id:
            raise OrchestrationError("不能把任务委派给协调会话自身")
        objective = _bounded_text(objective, "objective", 4_000)
        acceptance = _bounded_items(acceptance, "acceptance", 8, 500)
        allowed_paths = _bounded_items(allowed_paths, "allowed_paths", 16, 500)
        now = time.time()
        task = OrchestrationTask(
            id=f"T-{uuid.uuid4().hex[:8].upper()}",
            run_id=run.id,
            parent_task_id=None,
            source_session_id=source_session_id,
            target_session_id=target_session_id,
            role=role,
            objective=objective,
            acceptance=acceptance,
            allowed_paths=allowed_paths,
            write_policy=write_policy,
            status="queued",
            attempt=1,
            hop=len(tasks) + 1,
            summary="",
            evidence=(),
            changed_files=(),
            created_at=now,
            updated_at=now,
        )
        self.store.create_orchestration_task(task)
        self.store.add_orchestration_event(
            run.id,
            task_id=task.id,
            kind="task_queued",
            actor_session_id=source_session_id,
            source_session_id=source_session_id,
            target_session_id=target_session_id,
            summary=objective,
        )
        return task

    def start_task(
        self, task_id: str, actor_session_id: str
    ) -> OrchestrationTask:
        task = self.get_task(task_id)
        run = self.get_run(task.run_id)
        if run.status != "running" or task.status != "queued":
            raise OrchestrationError("只有 running run 中的 queued task 可以启动")
        if actor_session_id != task.target_session_id:
            raise OrchestrationError("只有目标会话可以启动任务")
        if self.store.running_task_for_session(actor_session_id) is not None:
            raise OrchestrationError("目标会话已有运行中的编排任务")
        updated = self.store.update_orchestration_task_status(task.id, "running")
        self.store.add_orchestration_event(
            run.id,
            task_id=task.id,
            kind="task_started",
            actor_session_id=actor_session_id,
            source_session_id=task.source_session_id,
            target_session_id=task.target_session_id,
            summary=task.objective,
        )
        return updated

    def report_task(
        self,
        task_id: str,
        actor_session_id: str,
        *,
        status: ReportStatus,
        summary: str,
        evidence: tuple[str, ...],
        changed_files: tuple[str, ...],
    ) -> OrchestrationTask:
        task = self.get_task(task_id)
        run = self.get_run(task.run_id)
        if actor_session_id != task.target_session_id:
            raise OrchestrationError("只有目标会话可以报告任务")
        if task.status != "running":
            raise OrchestrationError("只有 running task 可以报告结果")
        if status not in {"completed", "blocked", "failed"}:
            raise OrchestrationError("无效的任务报告状态")
        summary = _bounded_text(summary, "summary", 4_000)
        evidence = _bounded_items(evidence, "evidence", 8, 1_000)
        changed_files = _bounded_items(changed_files, "changed_files", 32, 500)
        if task.write_policy == "none" and changed_files:
            raise OrchestrationError("只读任务不能报告文件修改")
        updated = self.store.complete_orchestration_task(
            task.id,
            status,
            summary=summary,
            evidence=evidence,
            changed_files=changed_files,
        )
        self.store.add_orchestration_event(
            run.id,
            task_id=task.id,
            kind=f"task_{status}",
            actor_session_id=actor_session_id,
            source_session_id=actor_session_id,
            target_session_id=run.coordinator_session_id,
            summary=summary,
        )
        return updated

    def interrupt_task(self, task_id: str, reason: str) -> OrchestrationTask:
        task = self.get_task(task_id)
        if task.status != "running":
            return task
        reason = _bounded_text(reason, "reason", 4_000)
        updated = self.store.complete_orchestration_task(
            task.id,
            "blocked",
            summary=reason,
            evidence=(),
            changed_files=(),
        )
        run = self.get_run(task.run_id)
        self.store.add_orchestration_event(
            run.id,
            task_id=task.id,
            kind="task_blocked",
            actor_session_id=None,
            source_session_id=task.target_session_id,
            target_session_id=run.coordinator_session_id,
            summary=reason,
        )
        return updated

    def finish_run(
        self,
        run_id: str,
        actor_session_id: str,
        *,
        status: FinishStatus,
        summary: str,
    ) -> OrchestrationRun:
        run = self.get_run(run_id)
        self._require_coordinator(run, actor_session_id)
        if run.status not in {"running", "paused"}:
            raise OrchestrationError("只有 running 或 paused 编排可以结束")
        if status not in {"completed", "failed", "cancelled"}:
            raise OrchestrationError("无效的编排终止状态")
        unfinished = [
            task
            for task in self.store.orchestration_tasks(run.id)
            if task.status in {"queued", "running"}
        ]
        if unfinished:
            raise OrchestrationError("仍有 queued 或 running 任务，不能结束编排")
        summary = _bounded_text(summary, "summary", 4_000)
        updated = self.store.update_orchestration_run_status(run.id, status)
        self.store.add_orchestration_event(
            run.id,
            kind=f"run_{status}",
            actor_session_id=actor_session_id,
            source_session_id=actor_session_id,
            target_session_id=None,
            summary=summary,
        )
        return updated

    def pause_run(
        self, run_id: str, actor_session_id: str
    ) -> OrchestrationRun:
        run = self.get_run(run_id)
        self._require_coordinator(run, actor_session_id)
        if run.status != "running":
            raise OrchestrationError("只有 running 编排可以暂停")
        updated = self.store.update_orchestration_run_status(run.id, "paused")
        self.store.add_orchestration_event(
            run.id,
            kind="run_paused",
            actor_session_id=actor_session_id,
            source_session_id=actor_session_id,
            target_session_id=None,
            summary="用户暂停编排",
        )
        return updated

    def resume_run(
        self, run_id: str, actor_session_id: str
    ) -> OrchestrationRun:
        run = self.get_run(run_id)
        self._require_coordinator(run, actor_session_id)
        if run.status != "paused":
            raise OrchestrationError("只有 paused 编排可以恢复")
        if any(
            task.status == "interrupted"
            for task in self.store.orchestration_tasks(run.id)
        ):
            raise OrchestrationError("存在 interrupted 任务，需要重新委派后再恢复")
        updated = self.store.update_orchestration_run_status(run.id, "running")
        self.store.add_orchestration_event(
            run.id,
            kind="run_resumed",
            actor_session_id=actor_session_id,
            source_session_id=actor_session_id,
            target_session_id=None,
            summary="用户恢复编排",
        )
        return updated

    def cancel_run(
        self, run_id: str, actor_session_id: str, reason: str
    ) -> OrchestrationRun:
        run = self.get_run(run_id)
        self._require_coordinator(run, actor_session_id)
        if run.status not in {"proposed", "running", "paused"}:
            raise OrchestrationError("编排已经结束")
        reason = _bounded_text(reason, "reason", 4_000)
        self.store.cancel_orchestration_tasks(run.id)
        updated = self.store.update_orchestration_run_status(run.id, "cancelled")
        self.store.add_orchestration_event(
            run.id,
            kind="run_cancelled",
            actor_session_id=actor_session_id,
            source_session_id=actor_session_id,
            target_session_id=None,
            summary=reason,
        )
        return updated

    def get_run(self, run_id: str) -> OrchestrationRun:
        run = self.store.orchestration_run(run_id)
        if run.workspace != str(self.workspace):
            raise OrchestrationError("编排不属于当前工作区")
        return run

    def get_task(self, task_id: str) -> OrchestrationTask:
        task = self.store.orchestration_task(task_id)
        self.get_run(task.run_id)
        return task

    def ledger(self, run_id: str) -> tuple[OrchestrationEvent, ...]:
        self.get_run(run_id)
        return self.store.orchestration_events(run_id)

    def active_runs(self) -> tuple[OrchestrationRun, ...]:
        return self.store.active_orchestration_runs(self.workspace)

    def runs_for_session(
        self, session_id: str
    ) -> tuple[OrchestrationRun, ...]:
        self._require_session(session_id)
        return self.store.orchestration_runs_for_session(
            self.workspace, session_id
        )

    def tasks(self, run_id: str) -> tuple[OrchestrationTask, ...]:
        self.get_run(run_id)
        return self.store.orchestration_tasks(run_id)

    def record_event(
        self,
        run_id: str,
        *,
        kind: str,
        summary: str,
        task_id: str | None = None,
        actor_session_id: str | None = None,
        source_session_id: str | None = None,
        target_session_id: str | None = None,
    ) -> OrchestrationEvent:
        self.get_run(run_id)
        return self.store.add_orchestration_event(
            run_id,
            task_id=task_id,
            kind=kind,
            actor_session_id=actor_session_id,
            source_session_id=source_session_id,
            target_session_id=target_session_id,
            summary=_bounded_text(summary, "event summary", 4_000),
        )

    def _require_session(self, session_id: str) -> None:
        if not self.store.session_belongs_to(self.workspace, session_id):
            raise OrchestrationError("会话不属于当前工作区")

    @staticmethod
    def _require_coordinator(
        run: OrchestrationRun, actor_session_id: str
    ) -> None:
        if actor_session_id != run.coordinator_session_id:
            raise OrchestrationError("只有协调者会话可以执行此操作")


def _bounded_text(value: str, name: str, max_chars: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OrchestrationError(f"{name} 不能为空")
    result = value.strip()
    if len(result) > max_chars:
        raise OrchestrationError(f"{name} 超过 {max_chars} 字符")
    return result


def _bounded_items(
    values: tuple[str, ...], name: str, max_items: int, max_chars: int
) -> tuple[str, ...]:
    if len(values) > max_items:
        raise OrchestrationError(f"{name} 最多 {max_items} 项")
    return tuple(_bounded_text(value, name, max_chars) for value in values)
