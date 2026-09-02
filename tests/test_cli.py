from pathlib import Path
from io import StringIO
import json
import os
import stat

import litcode_agent.cli as cli
from litcode_agent.cli import main
from litcode_agent.model import AssistantTurn
from litcode_agent.session_store import SessionStore
from litcode_agent.ui import TerminalUI
from rich.console import Console


def make_ui(*answers: str) -> tuple[TerminalUI, StringIO]:
    output = StringIO()
    inputs = iter(answers)
    ui = TerminalUI(
        Console(
            file=output,
            color_system=None,
            force_terminal=False,
            width=100,
        ),
        lambda prompt: next(inputs),
    )
    return ui, output


def test_doctor_prints_safe_configuration(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-print")
    monkeypatch.setenv("LITCODE_MODEL", "example-model")

    ui, output = make_ui()

    assert main(["doctor", "--workspace", str(tmp_path)], ui) == 0

    assert "example-model" in output.getvalue()
    assert "do-not-print" not in output.getvalue()


def test_auth_login_stores_key_in_private_user_file(
    tmp_path: Path, monkeypatch
) -> None:
    user_home = tmp_path / "user-home"
    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt: "stored-secret")
    ui, output = make_ui()

    assert main(["auth", "login", "PRIMARY_KEY"], ui) == 0

    auth_file = user_home / ".local" / "share" / "litcode" / "auth.json"
    stored = json.loads(auth_file.read_text(encoding="utf-8"))
    assert stored["credentials"]["PRIMARY_KEY"] == {
        "type": "api",
        "key": "stored-secret",
    }
    if os.name == "posix":
        assert stat.S_IMODE(auth_file.stat().st_mode) == 0o600
        assert stat.S_IMODE(auth_file.parent.stat().st_mode) == 0o700
    assert "stored-secret" not in output.getvalue()
    assert "litcode/auth.json" in output.getvalue().replace("\n", "")
    assert "0600" in output.getvalue()


def test_skill_create_and_list_do_not_require_model_credentials(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ui, output = make_ui()

    assert main(
        [
            "skill",
            "--workspace",
            str(tmp_path),
            "create",
            "plan-work",
            "--description",
            "Create an implementation plan.",
        ],
        ui,
    ) == 0
    assert main(
        ["skill", "--workspace", str(tmp_path), "list"], ui
    ) == 0

    rendered = output.getvalue()
    assert "Skill" in rendered
    assert (tmp_path / ".litcode" / "skills" / "plan-work" / "SKILL.md").is_file()
    assert "plan-work · project · Create an implementation plan." in rendered


def test_auth_login_uses_selected_project_credential_name(
    tmp_path: Path, monkeypatch
) -> None:
    config_dir = tmp_path / ".litcode"
    config_dir.mkdir()
    (config_dir / "settings.json").write_text(
        json.dumps(
            {
                "defaultModel": "deepseek",
                "models": {
                    "deepseek": {
                        "model": "deepseek-v4-flash",
                        "apiKeyEnv": "DEEPSEEK_API_KEY",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    user_home = tmp_path / "user-home"
    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt: "stored-secret")
    ui, _ = make_ui()

    assert main(["auth", "login", "--workspace", str(tmp_path)], ui) == 0

    auth_file = user_home / ".local" / "share" / "litcode" / "auth.json"
    stored = json.loads(auth_file.read_text(encoding="utf-8"))
    assert set(stored["credentials"]) == {"DEEPSEEK_API_KEY"}


def test_auth_login_rejects_secret_as_name_before_prompt(
    tmp_path: Path, monkeypatch
) -> None:
    def unexpected_prompt(prompt: str) -> str:
        raise AssertionError("invalid credential name must fail before prompting")

    monkeypatch.setattr(cli.getpass, "getpass", unexpected_prompt)
    ui, output = make_ui()

    assert main(["auth", "login", "sk-must-not-be-a-name"], ui) == 1

    assert "environment variable name" in output.getvalue()


def test_run_wires_configuration_model_tools_and_agent(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeModel:
        def complete(self, messages, tools):
            assert len(tools) == 7
            return AssistantTurn("Task complete")

    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("LITCODE_MODEL", "example-model")
    monkeypatch.setattr(cli, "OpenAIChatModel", lambda settings: FakeModel())
    ui, output = make_ui()

    assert (
        main(
            ["run", "inspect the project", "--workspace", str(tmp_path)],
            ui,
        )
        == 0
    )

    rendered = output.getvalue()
    assert "inspect the project" in rendered
    assert "正在请求模型" in rendered
    assert "Task complete" in rendered


def test_models_queries_api_and_marks_current_model(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeModel:
        model = "model-b"

        def list_models(self):
            return ("model-a", "model-b")

    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("LITCODE_MODEL", "model-b")
    monkeypatch.setattr(cli, "OpenAIChatModel", lambda settings: FakeModel())
    ui, output = make_ui()

    assert main(["models", "--workspace", str(tmp_path)], ui) == 0

    rendered = output.getvalue()
    assert "model-a" in rendered
    assert "model-b" in rendered
    assert "✓" in rendered


def test_chat_keeps_context_and_switches_model(tmp_path: Path, monkeypatch) -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.model = "model-a"
            self.requests = []

        def complete(self, messages, tools):
            self.requests.append(list(messages))
            return AssistantTurn(f"回答 {len(self.requests)}")

        def list_models(self):
            return ("model-a", "model-b")

        def select_model(self, model):
            self.model = model

    model = FakeModel()
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("LITCODE_MODEL", "model-a")
    monkeypatch.setattr(cli, "OpenAIChatModel", lambda settings: model)
    ui, output = make_ui("第一个问题", "/model", "2", "继续", "/exit")

    assert main(["chat", "--workspace", str(tmp_path)], ui) == 0

    assert model.model == "model-b"
    assert model.requests[1][-1] == {"role": "user", "content": "继续"}
    assert {"role": "assistant", "content": "回答 1"} in model.requests[1]
    rendered = output.getvalue()
    assert "LitCode Agent" in rendered
    assert "已切换到模型 model-b" in rendered
    assert "回答 2" in rendered


def test_no_arguments_opens_current_directory_tui(monkeypatch, tmp_path: Path) -> None:
    opened = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("LITCODE_MODEL", "model-a")
    monkeypatch.setattr(cli, "run_tui", lambda settings, model: opened.append(settings.workspace) or 0)

    assert main([]) == 0

    assert opened == [tmp_path.resolve()]


def test_path_argument_opens_that_directory_tui(monkeypatch, tmp_path: Path) -> None:
    opened = []
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("LITCODE_MODEL", "model-a")
    monkeypatch.setattr(cli, "run_tui", lambda settings, model: opened.append(settings.workspace) or 0)

    assert main([str(tmp_path)]) == 0

    assert opened == [tmp_path.resolve()]


def test_chat_opens_tui_without_credentials_for_new_users(
    monkeypatch, tmp_path: Path
) -> None:
    opened = []
    monkeypatch.setenv("HOME", str(tmp_path / "fresh-home"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        cli,
        "run_tui",
        lambda settings, model: opened.append(
            (settings.configured, settings.api_key)
        )
        or 0,
    )

    assert main(["chat", "--workspace", str(tmp_path)]) == 0

    assert opened == [(False, "")]


def test_schedule_list_reads_durable_tasks_without_api_credentials(
    monkeypatch, tmp_path: Path
) -> None:
    database = tmp_path / ".litcode" / "sessions.db"
    store = SessionStore(database)
    creator = store.create(tmp_path, "model-a", [])
    target = store.create_child(creator, title="scheduled")
    task = store.create_scheduled_task(
        creator,
        target,
        "检查测试",
        {"kind": "once", "run_at": "2099-01-01T00:00:00+00:00"},
        "UTC",
        4070908800.0,
    )
    store.close()
    monkeypatch.setenv("HOME", str(tmp_path / "fresh-home"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ui, output = make_ui()

    assert main(["schedule", "list", "--workspace", str(tmp_path)], ui) == 0

    assert task.id[:8] in output.getvalue()
    assert "检查测试" in output.getvalue()
