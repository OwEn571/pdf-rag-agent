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
