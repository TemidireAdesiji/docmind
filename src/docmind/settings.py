"""Centralised, validated application settings.

All tunables are loaded from environment variables prefixed with
``DOCMIND_`` (see ``.env.example``).  Sensible defaults are provided
so the application works out of the box with a local Ollama instance.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

_PKG_ROOT = Path(__file__).resolve().parent


class _Paths(BaseSettings):
    """Derived directory paths (not user-configurable)."""

    base_dir: Path = _PKG_ROOT
    converted_docs: Path = _PKG_ROOT / "converted_docs"
    stored_chunks: Path = _PKG_ROOT / "stored_chunks"
    vector_data: Path = _PKG_ROOT / "vector_data"

    def ensure_dirs(self) -> None:
        for d in (
            self.converted_docs,
            self.stored_chunks,
            self.vector_data,
        ):
            d.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Application-wide configuration."""

    model_config = {"env_prefix": "DOCMIND_"}

    # ---- LLM provider ----
    llm_model: str = "qwen3:4b-instruct-2507-q4_K_M"
    llm_temperature: float = Field(default=0, ge=0, le=2)

    # ---- Embedding models ----
    dense_embedding: str = "sentence-transformers/all-mpnet-base-v2"
    sparse_embedding: str = "Qdrant/bm25"

    # ---- Vector index ----
    collection_label: str = "doc_child_segments"
    sparse_field: str = "sparse"

    # ---- Splitter tunables ----
    child_segment_size: int = Field(default=500, gt=0)
    child_segment_overlap: int = Field(default=100, ge=0)
    min_parent_length: int = Field(default=2000, gt=0)
    max_parent_length: int = Field(default=4000, gt=0)
    heading_levels: list[tuple[str, str]] = [
        ("#", "H1"),
        ("##", "H2"),
        ("###", "H3"),
    ]

    # ---- Agent limits ----
    max_tool_invocations: int = Field(default=8, gt=0)
    max_reasoning_cycles: int = Field(default=10, gt=0)
    graph_depth_limit: int = Field(default=50, gt=0)
    base_token_budget: int = Field(default=2000, gt=0)
    token_growth_rate: float = Field(default=0.9, gt=0, le=1)

    # ---- Tracing (Langfuse) ----
    tracing_enabled: bool = False
    tracing_public_key: str = ""
    tracing_secret_key: str = ""
    tracing_host: str = "http://localhost:3000"

    @field_validator("llm_temperature")
    @classmethod
    def _clamp_temperature(cls, v: float) -> float:
        return max(0.0, min(v, 2.0))

    @property
    def paths(self) -> _Paths:
        return _Paths()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton application settings."""
    # Allow .env to be loaded before pydantic reads env vars.
    from dotenv import load_dotenv

    load_dotenv(override=False)

    # Suppress HuggingFace tokenizer parallelism warnings.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    return Settings()
