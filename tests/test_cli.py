from pathlib import Path

import litcode_agent.cli as cli
from litcode_agent.model import AssistantTurn
from litcode_agent.cli import main


def test_doctor_prints_safe_configuration(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-print")
    monkeypatch.setenv("LITCODE_MODEL", "example-model")

    assert main(["doctor", "--workspace", str(tmp_path)]) == 0

    output = capsys.readouterr().out
    assert "example-model" in output
    assert "do-not-print" not in output


def test_run_wires_configuration_model_tools_and_agent(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    class FakeModel:
        def complete(self, messages, tools):
            assert len(tools) == 5
            return AssistantTurn("Task complete")

    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("LITCODE_MODEL", "example-model")
    monkeypatch.setattr(cli, "OpenAIChatModel", lambda settings: FakeModel())

    assert main(["run", "inspect the project", "--workspace", str(tmp_path)]) == 0

    captured = capsys.readouterr()
    assert captured.out == "Task complete\n"
    assert "asking model" in captured.err
