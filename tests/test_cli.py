from pathlib import Path
from io import StringIO

import litcode_agent.cli as cli
from litcode_agent.cli import main
from litcode_agent.model import AssistantTurn
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


def test_run_wires_configuration_model_tools_and_agent(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeModel:
        def complete(self, messages, tools):
            assert len(tools) == 5
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
