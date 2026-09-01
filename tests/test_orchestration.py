from pathlib import Path

import pytest

from litcode_agent.orchestration import (
    OrchestrationError,
    OrchestrationService,
)
from litcode_agent.session_store import SessionStore


def _sessions(tmp_path: Path) -> tuple[SessionStore, str, str, str]:
    store = SessionStore(tmp_path / "sessions.db")
    coordinator = store.create(tmp_path, "model", [])
    implementer = store.create(tmp_path, "model", [])
    reviewer = store.create(tmp_path, "model", [])
    return store, coordinator, implementer, reviewer


def test_task_lifecycle_is_causal_and_visible_in_ledger(tmp_path: Path) -> None:
    store, coordinator, implementer, _ = _sessions(tmp_path)
    service = OrchestrationService(store, tmp_path)
    run = service.start_run(coordinator, "实现并验证标签规范化")
    service.approve_run(run.id, coordinator)
    target_alias = store.session_info(implementer).alias

    task = service.delegate(
        run.id,
        coordinator,
        target_alias,
        role="implementer",
        objective="实现 normalize_tags",
        acceptance=("去重", "保持首次出现顺序", "测试通过"),
        allowed_paths=("src/tags.py", "tests/test_tags.py"),
        write_policy="workspace-write",
    )
    running = service.start_task(task.id, implementer)
    completed = service.report_task(
        task.id,
        implementer,
        status="completed",
        summary="实现与测试完成",
        evidence=("pytest: 8 passed",),
        changed_files=("src/tags.py", "tests/test_tags.py"),
    )
    finished = service.finish_run(
        run.id, coordinator, status="completed", summary="实现和审查证据充分"
    )

    assert running.status == "running"
    assert completed.status == "completed"
    assert finished.status == "completed"
    events = service.ledger(run.id)
    assert [event.kind for event in events] == [
        "run_proposed",
        "run_approved",
        "task_queued",
        "task_started",
        "task_completed",
        "run_completed",
    ]
    assert events[-2].task_id == task.id
    assert events[-2].source_session_id == implementer
    assert events[-2].target_session_id == coordinator


def test_only_coordinator_can_delegate_and_only_target_can_report(
    tmp_path: Path,
) -> None:
    store, coordinator, implementer, reviewer = _sessions(tmp_path)
    service = OrchestrationService(store, tmp_path)
    run = service.start_run(coordinator, "受限协作")
    service.approve_run(run.id, coordinator)

    with pytest.raises(OrchestrationError, match="协调者"):
        service.delegate(
            run.id,
            implementer,
            store.session_info(reviewer).alias,
            role="reviewer",
            objective="绕过协调者",
            acceptance=("不允许",),
            allowed_paths=(),
            write_policy="none",
        )

    task = service.delegate(
        run.id,
        coordinator,
        store.session_info(implementer).alias,
        role="implementer",
        objective="实现",
        acceptance=("完成",),
        allowed_paths=("src/feature.py",),
        write_policy="workspace-write",
    )
    service.start_task(task.id, implementer)

    with pytest.raises(OrchestrationError, match="目标会话"):
        service.report_task(
            task.id,
            reviewer,
            status="completed",
            summary="伪造完成",
            evidence=(),
            changed_files=(),
        )


def test_run_enforces_task_limit_without_silently_expanding_budget(
    tmp_path: Path,
) -> None:
    store, coordinator, implementer, _ = _sessions(tmp_path)
    service = OrchestrationService(store, tmp_path, max_tasks=1)
    run = service.start_run(coordinator, "限制测试")
    service.approve_run(run.id, coordinator)
    alias = store.session_info(implementer).alias
    service.delegate(
        run.id,
        coordinator,
        alias,
        role="implementer",
        objective="第一项",
        acceptance=("完成",),
        allowed_paths=(),
        write_policy="none",
    )

    with pytest.raises(OrchestrationError, match="任务上限"):
        service.delegate(
            run.id,
            coordinator,
            alias,
            role="reviewer",
            objective="第二项",
            acceptance=("完成",),
            allowed_paths=(),
            write_policy="none",
        )

    assert service.get_run(run.id).status == "paused"
    assert service.ledger(run.id)[-1].kind == "run_paused"


def test_running_task_is_paused_after_store_reopens(tmp_path: Path) -> None:
    store, coordinator, implementer, _ = _sessions(tmp_path)
    service = OrchestrationService(store, tmp_path)
    run = service.start_run(coordinator, "崩溃恢复")
    service.approve_run(run.id, coordinator)
    task = service.delegate(
        run.id,
        coordinator,
        store.session_info(implementer).alias,
        role="implementer",
        objective="写文件",
        acceptance=("完成",),
        allowed_paths=("src/a.py",),
        write_policy="workspace-write",
    )
    service.start_task(task.id, implementer)
    store.close()

    reopened = SessionStore(tmp_path / "sessions.db")
    recovered = OrchestrationService(reopened, tmp_path)

    assert recovered.get_run(run.id).status == "paused"
    assert recovered.get_task(task.id).status == "interrupted"
    assert recovered.ledger(run.id)[-1].kind == "run_interrupted"


def test_user_can_pause_resume_and_cancel_without_hidden_work(
    tmp_path: Path,
) -> None:
    store, coordinator, implementer, _ = _sessions(tmp_path)
    service = OrchestrationService(store, tmp_path)
    run = service.start_run(coordinator, "用户控制")
    service.approve_run(run.id, coordinator)
    task = service.delegate(
        run.id,
        coordinator,
        store.session_info(implementer).alias,
        role="implementer",
        objective="等待执行",
        acceptance=("完成",),
        allowed_paths=("src/a.py",),
        write_policy="workspace-write",
    )

    assert service.pause_run(run.id, coordinator).status == "paused"
    assert service.resume_run(run.id, coordinator).status == "running"
    assert service.cancel_run(run.id, coordinator, "用户取消").status == "cancelled"
    assert service.get_task(task.id).status == "cancelled"
    assert service.ledger(run.id)[-1].kind == "run_cancelled"


def test_model_request_budget_and_deadline_pause_the_run(tmp_path: Path) -> None:
    store, coordinator, _, _ = _sessions(tmp_path)
    now = [100.0]
    service = OrchestrationService(
        store,
        tmp_path,
        max_model_requests=1,
        timeout_seconds=10,
        clock=lambda: now[0],
    )
    run = service.start_run(coordinator, "预算测试")
    service.approve_run(run.id, coordinator)

    service.before_model_request(coordinator)
    with pytest.raises(OrchestrationError, match="模型请求上限"):
        service.before_model_request(coordinator)
    assert service.get_run(run.id).status == "paused"

    service.cancel_run(run.id, coordinator, "结束第一轮")
    second = service.start_run(coordinator, "超时测试")
    service.approve_run(second.id, coordinator)
    now[0] = 111.0
    with pytest.raises(OrchestrationError, match="截止时间"):
        service.before_model_request(coordinator)
    assert service.get_run(second.id).status == "paused"
