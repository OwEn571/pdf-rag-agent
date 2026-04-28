from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


def _empty_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }


@dataclass(frozen=True, slots=True)
class AgentToolSpec:
    name: str
    when: str
    returns: str
    input_schema: dict[str, Any] = field(default_factory=_empty_input_schema)
    research_executable: bool = False
    conversation_executable: bool = False

    def manifest_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "when": self.when,
            "returns": self.returns,
            "description": " ".join(part for part in [self.when, self.returns] if part),
            "input_schema": self.input_schema,
        }


@dataclass(frozen=True, slots=True)
class RegisteredAgentTool:
    name: str
    handler: Callable[[], None]
    requires: tuple[str, ...] = ()
    terminal: bool = False


class AgentToolExecutor:
    def __init__(self, tools: dict[str, RegisteredAgentTool]) -> None:
        self.tools = tools
        self.executed: set[str] = set()

    def run(self, action: str) -> bool:
        return self._run(action=action, stack=())

    def _run(self, *, action: str, stack: tuple[str, ...]) -> bool:
        tool = self.tools.get(action)
        if tool is None:
            return False
        if action in stack:
            cycle = " -> ".join([*stack, action])
            raise RuntimeError(f"agent tool dependency cycle detected: {cycle}")
        for requirement in tool.requires:
            if requirement not in self.executed:
                self._run(action=requirement, stack=(*stack, action))
        if tool.name not in self.executed:
            tool.handler()
            self.executed.add(tool.name)
        return tool.terminal


AGENT_TOOL_SPECS: tuple[AgentToolSpec, ...] = (
    AgentToolSpec(
        name="read_memory",
        when="Use when the current turn refers to prior answers, selected targets, previous recommendations, or active research context.",
        returns="Retained conversation context, active targets, prior tool results, and target bindings.",
        input_schema={
            "type": "object",
            "properties": {
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional memory keys or topics to inspect.",
                },
                "max_turns": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 24,
                    "default": 6,
                    "description": "How many recent turns to include.",
                },
                "focus": {
                    "type": "string",
                    "description": "What prior context the next step needs.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        research_executable=True,
        conversation_executable=True,
    ),
    AgentToolSpec(
        name="search_corpus",
        when="Use when the user needs evidence from the local Zotero/PDF corpus: papers, text, tables, captions, figures, formulas, results, or definitions.",
        returns="Candidate papers plus grounding evidence blocks from the local corpus.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to run against the local paper and block indexes.",
                },
                "scope": {
                    "type": "string",
                    "enum": ["auto", "papers", "blocks"],
                    "default": "auto",
                    "description": "Whether to search paper metadata, evidence blocks, or both.",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 12,
                    "description": "Maximum number of candidates to retrieve.",
                },
                "filters": {
                    "type": "object",
                    "properties": {
                        "paper_ids": {"type": "array", "items": {"type": "string"}},
                        "year_from": {"type": "integer"},
                        "year_to": {"type": "integer"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "additionalProperties": False,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        research_executable=True,
    ),
    AgentToolSpec(
        name="bm25_search",
        when="Use for lexical keyword search over local paper cards and PDF blocks, especially exact terms, formulas, identifiers, and titles.",
        returns="Ranked local evidence blocks with paper_id, title, page, snippet, score, and search source.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "scope": {"type": "string", "enum": ["auto", "papers", "blocks"], "default": "auto"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 12},
                "paper_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        research_executable=True,
    ),
    AgentToolSpec(
        name="vector_search",
        when="Use for dense semantic search over local paper cards and PDF blocks when wording may differ from the user's query.",
        returns="Ranked local semantic evidence blocks with paper_id, title, page, snippet, score, and search source.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "scope": {"type": "string", "enum": ["auto", "papers", "blocks"], "default": "auto"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 12},
                "paper_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        research_executable=True,
    ),
    AgentToolSpec(
        name="hybrid_search",
        when="Use for combined lexical and dense retrieval over the local corpus when robust recall is more important than a single retrieval method.",
        returns="Fused ranked local evidence blocks with paper_id, title, page, snippet, score, and search source.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "scope": {"type": "string", "enum": ["auto", "papers", "blocks"], "default": "auto"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 12},
                "paper_ids": {"type": "array", "items": {"type": "string"}},
                "alpha": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.5},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        research_executable=True,
    ),
    AgentToolSpec(
        name="rerank",
        when="Use to rerank already collected local evidence against the current query or focus terms before composing an answer.",
        returns="The current evidence list reordered and trimmed by relevance.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 12},
                "focus": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        research_executable=True,
    ),
    AgentToolSpec(
        name="read_pdf_page",
        when="Use to read indexed text/table/caption blocks from a known local PDF paper_id and page range.",
        returns="Evidence blocks from the requested paper pages, preserving page numbers and block types.",
        input_schema={
            "type": "object",
            "properties": {
                "paper_id": {"type": "string"},
                "page_from": {"type": "integer", "minimum": 1},
                "page_to": {"type": "integer", "minimum": 1},
                "max_chars": {"type": "integer", "minimum": 200, "maximum": 20000, "default": 4000},
            },
            "required": ["paper_id", "page_from"],
            "additionalProperties": False,
        },
        research_executable=True,
    ),
    AgentToolSpec(
        name="grep_corpus",
        when="Use for exact string or regex lookup over local paper cards and PDF blocks, especially formulas, identifiers, section labels, and quoted terms.",
        returns="Matching local evidence snippets with paper_id, page, block type, and grep pattern metadata.",
        input_schema={
            "type": "object",
            "properties": {
                "regex": {"type": "string"},
                "scope": {"type": "string", "enum": ["auto", "papers", "blocks"], "default": "auto"},
                "paper_ids": {"type": "array", "items": {"type": "string"}},
                "max_hits": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
            },
            "required": ["regex"],
            "additionalProperties": False,
        },
        research_executable=True,
    ),
    AgentToolSpec(
        name="summarize",
        when="Use to compress text or collected evidence into a focused summary before further reasoning or final composition.",
        returns="A concise focused summary and the number of source characters summarized.",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "target_words": {"type": "integer", "minimum": 20, "maximum": 1000, "default": 120},
                "focus": {"type": "array", "items": {"type": "string"}},
            },
            "required": [],
            "additionalProperties": False,
        },
        research_executable=True,
        conversation_executable=True,
    ),
    AgentToolSpec(
        name="verify_claim",
        when="Use to check whether a specific claim is supported by provided or currently collected evidence before composing.",
        returns="A pass/retry/clarify status, confidence, supporting evidence ids, and missing terms.",
        input_schema={
            "type": "object",
            "properties": {
                "claim": {"type": "string"},
                "evidence": {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            {"type": "string"},
                            {
                                "type": "object",
                                "properties": {
                                    "doc_id": {"type": "string"},
                                    "paper_id": {"type": "string"},
                                    "title": {"type": "string"},
                                    "page": {"type": "integer"},
                                    "block_type": {"type": "string"},
                                    "snippet": {"type": "string"},
                                    "text": {"type": "string"},
                                },
                                "additionalProperties": True,
                            },
                        ],
                    },
                },
                "min_overlap": {"type": "integer", "minimum": 1, "maximum": 20, "default": 2},
            },
            "required": ["claim"],
            "additionalProperties": False,
        },
        research_executable=True,
        conversation_executable=True,
    ),
    AgentToolSpec(
        name="web_search",
        when="Use for latest/current facts, external sources, citation counts, or when local corpus evidence is insufficient.",
        returns="External web evidence or citation-count lookup results. Treat them as dynamic and cite source URLs.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Web search query.",
                },
                "depth": {
                    "type": "string",
                    "enum": ["basic", "advanced"],
                    "default": "basic",
                },
                "include_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional domain allow-list.",
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        research_executable=True,
        conversation_executable=True,
    ),
    AgentToolSpec(
        name="fetch_url",
        when="Use to fetch the readable text from a specific HTTPS URL after web_search or when the user provides a source URL.",
        returns="Readable page text, title, URL, HTTP status, or a safety/error reason.",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "max_chars": {
                    "type": "integer",
                    "minimum": 200,
                    "maximum": 20000,
                    "default": 4000,
                },
            },
            "required": ["url"],
            "additionalProperties": False,
        },
        research_executable=True,
        conversation_executable=True,
    ),
    AgentToolSpec(
        name="query_library_metadata",
        when=(
            "Use for questions about the indexed local paper library metadata, including counts, years, authors, "
            "titles, tags, categories, PDF availability, and whether matching papers exist. This is a read-only "
            "SQL-style metadata query over the current paper index, not PDF evidence retrieval."
        ),
        returns="A read-only SQL query result over paper metadata rows, plus a concise answer grounded in those rows.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's metadata question in natural language.",
                },
                "sql_like_filter": {
                    "type": "string",
                    "description": "Optional read-only filter over title, author, year, tags, and categories.",
                },
                "group_by": {
                    "type": "string",
                    "description": "Optional metadata field for grouping, such as year, author, tag, or category.",
                },
                "agg": {
                    "type": "string",
                    "enum": ["none", "count", "list", "min", "max"],
                    "default": "none",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 20,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        conversation_executable=True,
    ),
    AgentToolSpec(
        name="compose",
        when="Use when available memory/evidence is sufficient to answer, or to run the final internal solve/verify/compose phase for a research answer.",
        returns="A final answer with citations when possible, or a verification state that asks for clarification.",
        input_schema={
            "type": "object",
            "properties": {
                "style": {
                    "type": "string",
                    "enum": ["auto", "concise", "bullets", "table", "narrative"],
                    "default": "auto",
                },
                "sections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional answer sections to produce.",
                },
                "cite": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether citations should be included when evidence is available.",
                },
                "answer_language": {
                    "type": "string",
                    "enum": ["auto", "zh", "en"],
                    "default": "auto",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        research_executable=True,
        conversation_executable=True,
    ),
    AgentToolSpec(
        name="todo_write",
        when="Use to create or update the visible task list for multi-step work before continuing with evidence gathering or composition.",
        returns="The current task list with pending, doing, done, or cancelled status.",
        input_schema={
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "text": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "doing", "done", "cancelled"],
                            },
                        },
                        "required": ["id", "text", "status"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        },
        research_executable=True,
        conversation_executable=True,
    ),
    AgentToolSpec(
        name="remember",
        when="Use to persist a reusable learning, routing correction, or durable user preference for future turns.",
        returns="The learning key and file path where the memory was stored.",
        input_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["key", "content"],
            "additionalProperties": False,
        },
        research_executable=True,
        conversation_executable=True,
    ),
    AgentToolSpec(
        name="Task",
        when=(
            "Use for an independent subtask that should run through the agent's normal research/conversation tools, "
            "especially one part of a multi-part user request."
        ),
        returns="A subtask answer, citations, verification status, and a compact execution summary.",
        input_schema={
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "prompt": {"type": "string"},
                "tools_allowed": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tool allow-list for the subtask.",
                },
                "max_steps": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 16,
                    "default": 8,
                },
            },
            "required": ["description", "prompt"],
            "additionalProperties": False,
        },
        conversation_executable=True,
    ),
    AgentToolSpec(
        name="ask_human",
        when="Use when intent confidence is low or a required slot cannot be resolved without the user's choice.",
        returns="Clarification question and clickable options.",
        input_schema={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The clarification question to ask the user.",
                },
                "options": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "value": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["label", "value"],
                        "additionalProperties": False,
                    },
                },
                "reason": {
                    "type": "string",
                    "description": "Why the agent cannot proceed confidently without user input.",
                },
                "blocking": {
                    "type": "boolean",
                    "default": True,
                },
            },
            "required": ["question", "reason"],
            "additionalProperties": False,
        },
        research_executable=True,
        conversation_executable=True,
    ),
)


def agent_tool_manifest() -> list[dict[str, Any]]:
    return [spec.manifest_payload() for spec in AGENT_TOOL_SPECS]


def all_agent_tool_names() -> set[str]:
    return {spec.name for spec in AGENT_TOOL_SPECS}


def research_execution_tool_names() -> set[str]:
    return {spec.name for spec in AGENT_TOOL_SPECS if spec.research_executable}


def conversation_execution_tool_names() -> set[str]:
    return {spec.name for spec in AGENT_TOOL_SPECS if spec.conversation_executable}


def conversation_tool_sequence(*, planned_actions: list[str] | None = None, relation: str = "") -> list[str]:
    _ = relation
    allowed = conversation_execution_tool_names()
    planned = [str(item) for item in (planned_actions or []) if str(item) in allowed]
    return list(dict.fromkeys(planned))


def research_tool_sequence(
    *,
    planned_actions: list[str] | None,
    use_web_search: bool,
    needs_reflection: bool,
) -> list[str]:
    _ = (use_web_search, needs_reflection)
    allowed = research_execution_tool_names()
    actions = [str(item) for item in (planned_actions or []) if str(item) in allowed]
    return list(dict.fromkeys(actions))


def normalize_plan_actions(*, actions: object, allowed: set[str]) -> list[str]:
    if not isinstance(actions, list):
        return []
    return [str(item) for item in actions if str(item) in allowed]
