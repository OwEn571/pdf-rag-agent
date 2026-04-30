from __future__ import annotations

from types import SimpleNamespace

from app.services.model_clients import ModelClients


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
