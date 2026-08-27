from types import SimpleNamespace
from pathlib import Path

import pytest

from litcode_agent.config import Settings
from litcode_agent.model import ModelError, OpenAIChatModel


class FakeCompletions:
    def __init__(self, response) -> None:
        self.response = response
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.response


def fake_client(response):
    completions = FakeCompletions(response)
    return SimpleNamespace(
        chat=SimpleNamespace(completions=completions), completions=completions
    )


def settings(tmp_path: Path) -> Settings:
    return Settings.from_env(
        tmp_path,
        {"OPENAI_API_KEY": "secret", "LITCODE_MODEL": "example-model"},
    )


def test_normalizes_chat_completion_tool_calls(tmp_path: Path) -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call-1",
                            type="function",
                            function=SimpleNamespace(
                                name="read_file", arguments='{"path":"a.py"}'
                            ),
                        )
                    ],
                )
            )
        ]
    )
    client = fake_client(response)

    turn = OpenAIChatModel(settings(tmp_path), client).complete(
        [{"role": "user", "content": "read"}], []
    )

    assert turn.content is None
    assert turn.tool_calls[0].id == "call-1"
    assert turn.tool_calls[0].name == "read_file"
    assert turn.finish_reason == "tool_calls"
    assert client.completions.requests[0]["model"] == "example-model"


def test_rejects_response_without_choices(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="no choices"):
        OpenAIChatModel(
            settings(tmp_path), fake_client(SimpleNamespace(choices=[]))
        ).complete([], [])


def test_rejects_unsupported_tool_call_type(tmp_path: Path) -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[SimpleNamespace(type="custom")],
                )
            )
        ]
    )

    with pytest.raises(ModelError, match="unsupported"):
        OpenAIChatModel(settings(tmp_path), fake_client(response)).complete([], [])
