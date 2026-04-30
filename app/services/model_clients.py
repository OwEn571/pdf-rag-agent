from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any, Callable

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.core.config import Settings

logger = logging.getLogger(__name__)


class ModelClients:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._chat: ChatOpenAI | None = None
        self._vlm: ChatOpenAI | None = None
        self._http_client: httpx.Client | None = None
        self._async_http_client: httpx.AsyncClient | None = None

    @property
    def http_client(self) -> httpx.Client:
        if self._http_client is None:
            self._http_client = httpx.Client(trust_env=False)
        return self._http_client

    @property
    def async_http_client(self) -> httpx.AsyncClient:
        if self._async_http_client is None:
            self._async_http_client = httpx.AsyncClient(trust_env=False)
        return self._async_http_client

    async def aclose(self) -> None:
        if self._async_http_client is not None:
            await self._async_http_client.aclose()
        if self._http_client is not None:
            self._http_client.close()
        self._async_http_client = None
        self._http_client = None
        self._chat = None
        self._vlm = None

    def close(self) -> None:
        if self._http_client is not None:
            self._http_client.close()
        self._http_client = None
        self._chat = None
        self._vlm = None

    @property
    def chat(self) -> ChatOpenAI | None:
        if not self.settings.openai_api_key:
            return None
        if self._chat is None:
            self._chat = ChatOpenAI(
                model=self.settings.chat_model,
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
                temperature=0.1,
                max_tokens=self.settings.chat_max_tokens,
                http_client=self.http_client,
                http_async_client=self.async_http_client,
            )
        return self._chat

    @property
    def vlm(self) -> ChatOpenAI | None:
        if not self.settings.openai_api_key or not self.settings.enable_figure_vlm:
            return None
        if self._vlm is None:
            self._vlm = ChatOpenAI(
                model=self.settings.vlm_model,
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
                temperature=0.0,
                max_tokens=self.settings.chat_max_tokens,
                http_client=self.http_client,
                http_async_client=self.async_http_client,
            )
        return self._vlm

    def invoke_text(self, *, system_prompt: str, human_prompt: str, fallback: str = "") -> str:
        model = self.chat
        if model is None:
            return fallback
        try:
            response = model.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
            return str(response.content or "").strip() or fallback
        except Exception as exc:  # noqa: BLE001
            logger.warning("chat invoke_text failed: %s", exc)
            return fallback

    def stream_text(
        self,
        *,
        system_prompt: str,
        human_prompt: str,
        on_delta: Callable[[str], None],
        on_logprobs: Callable[[list[float]], None] | None = None,
        request_logprobs: bool = False,
        fallback: str = "",
    ) -> str:
        model = self.chat
        if model is None:
            if fallback:
                on_delta(fallback)
            return fallback
        chunks: list[str] = []
        try:
            stream_model = model.bind(logprobs=True) if request_logprobs else model
            for chunk in stream_model.stream([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]):
                content = chunk.content
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text = "".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content)
                else:
                    text = str(content or "")
                if not text:
                    continue
                chunks.append(text)
                on_delta(text)
                if on_logprobs is not None:
                    logprobs = self._extract_chunk_logprobs(chunk)
                    if logprobs:
                        on_logprobs(logprobs)
            text = "".join(chunks).strip()
            return text or fallback
        except Exception as exc:  # noqa: BLE001
            logger.warning("chat stream_text failed: %s", exc)
            text = self.invoke_text(system_prompt=system_prompt, human_prompt=human_prompt, fallback=fallback)
            if text:
                on_delta(text)
            return text

    def invoke_json(self, *, system_prompt: str, human_prompt: str, fallback: Any) -> Any:
        text = self.invoke_text(system_prompt=system_prompt, human_prompt=human_prompt, fallback="")
        if not text:
            return fallback
        payload = self._safe_parse_json_object(text)
        return payload if payload is not None else fallback

    def invoke_text_messages(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        fallback: str = "",
    ) -> str:
        model = self.chat
        if model is None:
            return fallback
        try:
            response = model.invoke(self._chat_messages(system_prompt=system_prompt, messages=messages))
            return str(response.content or "").strip() or fallback
        except Exception as exc:  # noqa: BLE001
            logger.warning("chat invoke_text_messages failed: %s", exc)
            return fallback

    def invoke_json_messages(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        fallback: Any,
    ) -> Any:
        text = self.invoke_text_messages(system_prompt=system_prompt, messages=messages, fallback="")
        if not text:
            return fallback
        payload = self._safe_parse_json_object(text)
        return payload if payload is not None else fallback

    def invoke_tool_plan(
        self,
        *,
        system_prompt: str,
        human_prompt: str,
        tools: list[dict[str, Any]],
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        model = self.chat
        if model is None or not tools:
            return fallback
        try:
            bound_model = model.bind_tools(self._openai_tool_definitions(tools))
            response = bound_model.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
        except Exception as exc:  # noqa: BLE001
            logger.warning("chat invoke_tool_plan failed: %s", exc)
            return fallback
        actions, arguments = self._extract_tool_call_actions(response)
        if not actions:
            return fallback
        return {
            "thought": str(getattr(response, "content", "") or fallback.get("thought", "")),
            "actions": actions,
            "tool_call_args": arguments,
            "stop_conditions": ["tool_calls_selected"],
        }

    def invoke_tool_plan_messages(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        model = self.chat
        if model is None or not tools:
            return fallback
        try:
            bound_model = model.bind_tools(self._openai_tool_definitions(tools))
            response = bound_model.invoke(self._chat_messages(system_prompt=system_prompt, messages=messages))
        except Exception as exc:  # noqa: BLE001
            logger.warning("chat invoke_tool_plan_messages failed: %s", exc)
            return fallback
        actions, arguments = self._extract_tool_call_actions(response)
        if not actions:
            return fallback
        return {
            "thought": str(getattr(response, "content", "") or fallback.get("thought", "")),
            "actions": actions,
            "tool_call_args": arguments,
            "stop_conditions": ["tool_calls_selected"],
        }

    def invoke_multimodal_json(self, *, system_prompt: str, human_content: list[dict[str, Any]], fallback: Any) -> Any:
        model = self.vlm
        if model is None:
            return fallback
        try:
            response = model.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_content)])
        except Exception as exc:  # noqa: BLE001
            logger.warning("vlm invoke failed: %s", exc)
            return fallback
        payload = self._safe_parse_json_object(str(response.content or ""))
        return payload if payload is not None else fallback

    @staticmethod
    def _chat_messages(*, system_prompt: str, messages: list[dict[str, str]]) -> list[BaseMessage]:
        rendered: list[BaseMessage] = [SystemMessage(content=system_prompt)]
        for item in messages:
            role = str(item.get("role", "") or "").strip().lower()
            content = str(item.get("content", "") or "")
            if not content:
                continue
            if role in {"assistant", "ai"}:
                rendered.append(AIMessage(content=content))
            else:
                rendered.append(HumanMessage(content=content))
        return rendered

    @staticmethod
    def _openai_tool_definitions(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        definitions: list[dict[str, Any]] = []
        for tool in tools:
            name = str(tool.get("name", "")).strip()
            if not name:
                continue
            description = str(tool.get("description", "") or "").strip()
            if not description:
                description = " ".join(
                    part
                    for part in [
                        str(tool.get("when", "")).strip(),
                        str(tool.get("returns", "")).strip(),
                    ]
                    if part
                )
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description or name,
                        "parameters": ModelClients._tool_input_schema(tool),
                    },
                }
            )
        return definitions

    @staticmethod
    def _tool_input_schema(tool: dict[str, Any]) -> dict[str, Any]:
        schema = tool.get("input_schema")
        if schema is None:
            schema = tool.get("parameters")
        if not isinstance(schema, dict):
            return {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            }
        normalized = deepcopy(schema)
        if normalized.get("type") != "object":
            normalized["type"] = "object"
        if not isinstance(normalized.get("properties"), dict):
            normalized["properties"] = {}
        if "required" in normalized and not isinstance(normalized["required"], list):
            normalized["required"] = []
        return normalized

    @staticmethod
    def _extract_tool_call_actions(response: Any) -> tuple[list[str], list[dict[str, Any]]]:
        raw_tool_calls = getattr(response, "tool_calls", None) or []
        if not raw_tool_calls:
            additional_kwargs = getattr(response, "additional_kwargs", {}) or {}
            raw_tool_calls = additional_kwargs.get("tool_calls", []) or []
        actions: list[str] = []
        arguments: list[dict[str, Any]] = []
        for call in raw_tool_calls:
            name = ""
            args: Any = {}
            if isinstance(call, dict):
                name = str(call.get("name") or "")
                args = call.get("args", {})
                function_payload = call.get("function")
                if not name and isinstance(function_payload, dict):
                    name = str(function_payload.get("name") or "")
                    args = function_payload.get("arguments", args)
            else:
                name = str(getattr(call, "name", "") or "")
                args = getattr(call, "args", {})
            if not name:
                continue
            if isinstance(args, str):
                try:
                    parsed_args = json.loads(args)
                except json.JSONDecodeError:
                    parsed_args = {"raw": args}
            elif isinstance(args, dict):
                parsed_args = dict(args)
            else:
                parsed_args = {}
            actions.append(name)
            arguments.append({"name": name, "args": parsed_args})
        return actions, arguments

    @staticmethod
    def _extract_chunk_logprobs(chunk: Any) -> list[float]:
        payloads: list[Any] = []
        for attr in ("response_metadata", "generation_info", "additional_kwargs"):
            value = getattr(chunk, attr, None)
            if isinstance(value, dict) and value:
                payloads.append(value.get("logprobs", value))
        values: list[float] = []
        for payload in payloads:
            values.extend(ModelClients._extract_logprob_values(payload))
        return values

    @staticmethod
    def _extract_logprob_values(payload: Any) -> list[float]:
        values: list[float] = []
        if isinstance(payload, dict):
            content = payload.get("content")
            if isinstance(content, list):
                for item in content:
                    values.extend(ModelClients._extract_logprob_values(item))
                return values
            if "logprob" in payload:
                try:
                    values.append(float(payload["logprob"]))
                except (TypeError, ValueError):
                    pass
                return values
            for key, value in payload.items():
                if key == "top_logprobs":
                    continue
                values.extend(ModelClients._extract_logprob_values(value))
        elif isinstance(payload, list):
            for item in payload:
                values.extend(ModelClients._extract_logprob_values(item))
        return values

    @staticmethod
    def _safe_parse_json_object(raw: str) -> dict[str, Any] | None:
        raw = raw.strip()
        if not raw:
            return None
        if raw.startswith("{") and raw.endswith("}"):
            try:
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError:
                return None
        left = raw.find("{")
        right = raw.rfind("}")
        if left < 0 or right <= left:
            return None
        try:
            payload = json.loads(raw[left : right + 1])
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            return None
        return None
