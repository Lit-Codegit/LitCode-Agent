from pathlib import Path

from litcode_agent.orchestration import OrchestrationService
from litcode_agent.scheduler import LocalScheduler
from litcode_agent.session_store import SessionStore


def _queued_task(tmp_path: Path, *, write: bool = True):
    store = SessionStore(tmp_path / "sessions.db")
    coordinator = store.create(tmp_path, "model", [])
    worker = store.create(tmp_path, "model", [])
    service = OrchestrationService(store, tmp_path)
    run = service.start_run(coordinator, "完成特性")
    service.approve_run(run.id, coordinator)
    task = service.delegate(
        run.id,
        coordinator,
        store.session_info(worker).alias,
        role="implementer" if write else "reviewer",
        objective="实现解析器" if write else "审查解析器",
        acceptance=("测试通过",),
        allowed_paths=("src/parser.py",) if write else (),
        write_policy="workspace-write" if write else "none",
    )
    return store, service, run, task, coordinator, worker


def test_scheduler_wakes_only_an_idle_mounted_target(tmp_path: Path) -> None:
    _, service, _, task, coordinator, worker = _queued_task(tmp_path)
    scheduler = LocalScheduler(service)

    assert scheduler.next_action(mounted={coordinator: 1}, busy=set()) is None
    action = scheduler.next_action(
        mounted={coordinator: 1, worker: 2}, busy={worker}
    )
    assert action is None
    action = scheduler.next_action(
        mounted={coordinator: 1, worker: 2}, busy=set()
    )

    assert action is not None
    assert action.kind == "wake_task"
    assert action.session_id == worker
    assert action.pane_slot == 2
    assert action.task_id == task.id
    assert "report_task" in action.prompt
    assert service.get_task(task.id).status == "running"


def test_scheduler_resumes_coordinator_once_with_bounded_result(
    tmp_path: Path,
) -> None:
    _, service, _, task, coordinator, worker = _queued_task(tmp_path)
    scheduler = LocalScheduler(service)
    scheduler.next_action(
        mounted={coordinator: 1, worker: 2}, busy=set()
    )
    service.report_task(
        task.id,
        worker,
        status="completed",
        summary="parser 已完成",
        evidence=("pytest: 12 passed",),
        changed_files=("src/parser.py",),
    )

    action = scheduler.next_action(
        mounted={coordinator: 1, worker: 2}, busy=set()
    )

    assert action is not None
    assert action.kind == "resume_coordinator"
    assert action.session_id == coordinator
    assert "parser 已完成" in action.prompt
    assert "pytest: 12 passed" in action.prompt
    assert scheduler.next_action(
        mounted={coordinator: 1, worker: 2}, busy=set()
    ) is None


def test_scheduler_serializes_write_tasks_for_the_workspace(tmp_path: Path) -> None:
    store, service, run, first, coordinator, first_worker = _queued_task(tmp_path)
    second_worker = store.create(tmp_path, "model", [])
    second = service.delegate(
        run.id,
        coordinator,
        store.session_info(second_worker).alias,
        role="implementer",
        objective="实现 formatter",
        acceptance=("测试通过",),
        allowed_paths=("src/formatter.py",),
        write_policy="workspace-write",
    )
    scheduler = LocalScheduler(service)
    mounted = {coordinator: 1, first_worker: 2, second_worker: 3}

    first_action = scheduler.next_action(mounted=mounted, busy=set())
    second_action = scheduler.next_action(
        mounted=mounted, busy={first_worker}
    )

    assert first_action is not None and first_action.task_id == first.id
    assert second_action is None
    assert service.get_task(second.id).status == "queued"
