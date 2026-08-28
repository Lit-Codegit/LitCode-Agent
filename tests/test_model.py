from types import SimpleNamespace
from pathlib import Path

import pytest

from litcode_agent.config import Settings
from litcode_agent.model import ModelDelta, ModelError, OpenAIChatModel


class FakeCompletions:
    def __init__(self, response) -> None:
        self.response = response
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.response


class FakeModels:
    def __init__(self, model_ids: tuple[str, ...] = ()) -> None:
        self.model_ids = model_ids

    def list(self):
        return SimpleNamespace(
            data=[SimpleNamespace(id=model_id) for model_id in self.model_ids]
        )


def fake_client(response, model_ids: tuple[str, ...] = ()):
    completions = FakeCompletions(response)
    return SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
        completions=completions,
        models=FakeModels(model_ids),
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


def test_lists_sorted_unique_models_and_selects_one(tmp_path: Path) -> None:
    client = fake_client(
        SimpleNamespace(choices=[]),
        ("model-z", "model-a", "model-z"),
    )
    model = OpenAIChatModel(settings(tmp_path), client)

    assert model.list_models() == ("model-a", "model-z")

    model.select_model(" model-z ")

    assert model.model == "model-z"


def test_rejects_empty_model_selection(tmp_path: Path) -> None:
    model = OpenAIChatModel(
        settings(tmp_path), fake_client(SimpleNamespace(choices=[]))
    )

    with pytest.raises(ValueError, match="must not be empty"):
        model.select_model("  ")


def test_stream_normalizes_text_and_split_tool_arguments(tmp_path: Path) -> None:
    chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason=None,
                    delta=SimpleNamespace(content="你", tool_calls=None),
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason=None,
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="call-1",
                                function=SimpleNamespace(
                                    name="read_file", arguments='{"path":'
                                ),
                            )
                        ],
                    ),
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="tool_calls",
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id=None,
                                function=SimpleNamespace(
                                    name=None, arguments='"a.py"}'
                                ),
                            )
                        ],
                    ),
                )
            ]
        ),
    ]
    client = fake_client(chunks)

    deltas = list(
        OpenAIChatModel(settings(tmp_path), client).stream(
            [{"role": "user", "content": "读取"}], []
        )
    )

    assert deltas == [
        ModelDelta(content="你"),
        ModelDelta(
            tool_index=0,
            tool_call_id="call-1",
            tool_name="read_file",
            tool_arguments='{"path":',
        ),
        ModelDelta(
            tool_index=0,
            tool_arguments='"a.py"}',
            finish_reason="tool_calls",
        ),
    ]
    assert client.completions.requests[0]["stream"] is True
