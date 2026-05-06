from __future__ import annotations

from types import SimpleNamespace

from app.core.config import Settings
from app.services.infra.model_clients import ModelClients


def test_extract_tool_call_actions_from_langchain_shape() -> None:
    response = SimpleNamespace(
        content="",
        tool_calls=[
            {"name": "search_corpus", "args": {"reason": "local evidence"}},
            {"name": "compose", "args": {}},
        ],
    )

    actions, arguments = ModelClients._extract_tool_call_actions(response)

    assert actions == ["search_corpus", "compose"]
    assert arguments[0]["args"]["reason"] == "local evidence"


def test_extract_tool_call_actions_from_openai_shape() -> None:
    response = SimpleNamespace(
        additional_kwargs={
            "tool_calls": [
                {
                    "function": {
                        "name": "web_search",
                        "arguments": '{"reason": "dynamic citation count"}',
                    }
                }
            ]
        }
    )

    actions, arguments = ModelClients._extract_tool_call_actions(response)

    assert actions == ["web_search"]
    assert arguments == [{"name": "web_search", "args": {"reason": "dynamic citation count"}}]


def test_openai_tool_definitions_preserve_input_schema() -> None:
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    definitions = ModelClients._openai_tool_definitions(
        [
            {
                "name": "search_corpus",
                "description": "Search local PDFs.",
                "input_schema": schema,
            }
        ]
    )

    function = definitions[0]["function"]
    assert function["name"] == "search_corpus"
    assert function["parameters"] == schema
    assert "reason" not in function["parameters"]["properties"]


def test_extract_chunk_logprobs_reads_openai_content_entries_without_top_alternatives() -> None:
    chunk = SimpleNamespace(
        response_metadata={
            "logprobs": {
                "content": [
                    {"token": "A", "logprob": -0.1, "top_logprobs": [{"token": "B", "logprob": -5.0}]},
                    {"token": "B", "logprob": "-0.2"},
                ]
            }
        }
    )

    assert ModelClients._extract_chunk_logprobs(chunk) == [-0.1, -0.2]


def test_extract_chunk_logprobs_handles_generation_info_and_bad_values() -> None:
    chunk = SimpleNamespace(
        generation_info={"logprobs": {"content": [{"logprob": "bad"}, {"logprob": -0.3}]}},
        additional_kwargs={"ignored": True},
    )

    assert ModelClients._extract_chunk_logprobs(chunk) == [-0.3]


def test_stream_text_emits_each_logprob_batch_once() -> None:
    class StreamingModel:
        def __init__(self) -> None:
            self.bind_kwargs: list[dict[str, object]] = []

        def bind(self, **kwargs: object) -> "StreamingModel":
            self.bind_kwargs.append(kwargs)
            return self

        def stream(self, messages: object) -> list[SimpleNamespace]:
            return [
                SimpleNamespace(
                    content="A",
                    response_metadata={"logprobs": {"content": [{"token": "A", "logprob": -0.1}]}},
                ),
                SimpleNamespace(
                    content=[{"text": "B"}],
                    response_metadata={"logprobs": {"content": [{"token": "B", "logprob": -0.2}]}},
                ),
            ]

    model = StreamingModel()
    clients = ModelClients(Settings(_env_file=None, openai_api_key="test-key"))
    clients._chat = model  # type: ignore[assignment]
    deltas: list[str] = []
    logprob_batches: list[list[float]] = []

    text = clients.stream_text(
        system_prompt="system",
        human_prompt="human",
        on_delta=deltas.append,
        on_logprobs=logprob_batches.append,
        request_logprobs=True,
    )

    assert text == "AB"
    assert deltas == ["A", "B"]
    assert logprob_batches == [[-0.1], [-0.2]]
    assert model.bind_kwargs == [{"logprobs": True}]
