from pathlib import Path

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
