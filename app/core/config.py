from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ENV_FILE = PROJECT_ROOT / ".env"


def _default_zotero_root() -> Path:
    candidates = (
        PROJECT_ROOT.parent / "Zotero",
        Path.home() / "Zotero",
        Path("/Users/owen/Zotero"),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


class Settings(BaseSettings):
    app_name: str = "zotero-paper-rag-v4"
    env: str = "dev"
    log_level: str = "INFO"

    zotero_root: Path = Field(default_factory=_default_zotero_root)
    zotero_sqlite_path: Path | None = None
    zotero_storage_dir: Path | None = None

    openai_api_key: str = Field(default="", validation_alias=AliasChoices("OPENAI_API_KEY", "QIHANG_API"))
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias=AliasChoices("OPENAI_BASE_URL", "QIHANG_BASE_URL"),
    )
    chat_model: str = "gpt-4o-mini"
    chat_max_tokens: int = 1800
    embedding_model: str = "text-embedding-3-large"
    embedding_fallback_model: str = "text-embedding-3-small"
    vlm_model: str = "gpt-4.1-mini"
    enable_figure_vlm: bool = True
    enable_table_vlm: bool = True
    pdf_render_dpi: int = 180
    pdf_hi_res_max_pages_per_document: int = 6
    figure_vlm_timeout_seconds: float = 30.0
    figure_vlm_max_side: int = 1400

    milvus_uri: str = "http://localhost:19530"
    milvus_paper_collection: str = "zprag_v4_papers"
    milvus_block_collection: str = "zprag_v4_blocks"

    tavily_api_key: str = ""
    tavily_search_depth: str = "basic"
    tavily_timeout_seconds: float = 20.0

    paper_bm25_top_k: int = 12
    paper_dense_top_k: int = 12
    block_bm25_top_k: int = 16
    block_dense_top_k: int = 12
    paper_limit_default: int = 6
    evidence_limit_default: int = 14
    retrieval_paper_match_boosts: tuple[dict[str, Any], ...] = ()
    retrieval_formula_token_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "log σ": 2.5,
            "log sigma": 2.5,
            "πθ": 2.0,
            "pi_theta": 2.0,
            "πref": 2.0,
            "pi_ref": 2.0,
            "β": 1.8,
            "beta": 1.5,
            "yw": 1.4,
            "y_w": 1.4,
            "yl": 1.4,
            "y_l": 1.4,
            "preferred": 0.8,
            "dispreferred": 0.8,
            "reference policy": 0.8,
            "surrogate objective": 2.4,
            "clipped surrogate": 4.0,
            "l^clip": 4.0,
            "lclip": 4.0,
            "clip(": 3.0,
            "clip (": 3.0,
            "probability ratio": 2.0,
            "ratio r": 1.6,
            "advantage": 1.8,
            "epsilon": 1.2,
            "ϵ": 1.2,
            "θold": 1.4,
            "theta old": 1.4,
            "proximal policy optimization": 1.8,
        }
    )
    retrieval_target_formula_token_weights: dict[str, dict[str, float]] = Field(default_factory=dict)
    solver_metric_token_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "pba": 4.0,
            "win rate": 4.0,
            "accuracy": 3.0,
            "acc": 3.0,
            "review": 3.0,
            "roleplay": 3.0,
            "aime": 2.0,
            "mmlu": 2.0,
            "benchmark": 2.0,
        }
    )
    upsert_batch_size: int = 128
    embedding_request_timeout_seconds: float = 120.0
    embedding_batch_retry_attempts: int = 3
    llm_retry_budget: int = 1
    agent_history_max_turns: int = 24
    agent_max_steps: int = 8
    agent_max_parallel_tools: int = 4
    agent_confidence_floor: float = 0.6
    agent_max_clarification_attempts: int = 2
    agent_disambiguation_auto_resolve_threshold: float = 0.85
    agent_disambiguation_recommend_threshold: float = 0.65
    ingestion_excluded_tags: tuple[str, ...] = ("书籍", "book", "books")
    ingestion_allowed_item_types: tuple[str, ...] = (
        "journalArticle",
        "conferencePaper",
        "preprint",
        "thesis",
        "report",
        "manuscript",
    )
    ingestion_academic_web_hosts: tuple[str, ...] = (
        "arxiv.org",
        "openreview.net",
        "aclanthology.org",
        "proceedings.mlr.press",
        "papers.nips.cc",
        "iclr.cc",
        "thecvf.com",
        "cvf.com",
        "biorxiv.org",
        "medrxiv.org",
    )

    data_dir: Path = Path("./data")
    paper_store_path: Path = Path("./data/v4_papers.jsonl")
    block_store_path: Path = Path("./data/v4_blocks.jsonl")
    ingestion_state_path: Path = Path("./data/v4_ingestion_state.json")
    session_store_path: Path = Path("./data/v4_sessions.sqlite3")
    eval_cases_path: Path = Path("./evals/cases_test_md.yaml")

    admin_api_key: str = ""
    library_api_key: str = ""
    allow_local_pdf_without_api_key: bool = True
    allow_same_origin_pdf_without_api_key: bool = True
    api_rate_limit_window_seconds: int = 60
    admin_rate_limit_per_window: int = 10
    pdf_rate_limit_per_window: int = 120
    cors_allow_origins: tuple[str, ...] = ()

    model_config = SettingsConfigDict(
        env_file=PROJECT_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    @computed_field
    @property
    def resolved_zotero_sqlite_path(self) -> Path:
        return self.zotero_sqlite_path or (self.zotero_root / "zotero.sqlite")

    @computed_field
    @property
    def resolved_zotero_storage_dir(self) -> Path:
        return self.zotero_storage_dir or (self.zotero_root / "storage")

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def parse_cors_allow_origins(cls, value: Any) -> tuple[str, ...] | Any:
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return ()
            if cleaned.startswith("["):
                return value
            return tuple(part.strip() for part in cleaned.split(",") if part.strip())
        return value

    def ensure_runtime_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.paper_store_path.parent.mkdir(parents=True, exist_ok=True)
        self.block_store_path.parent.mkdir(parents=True, exist_ok=True)
        self.ingestion_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_store_path.parent.mkdir(parents=True, exist_ok=True)
        self.eval_cases_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_runtime_dirs()
    return settings
