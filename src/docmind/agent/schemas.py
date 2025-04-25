"""Pydantic models for structured LLM outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QuestionAssessment(BaseModel):
    """Result of analysing a user's question for clarity."""

    is_clear: bool = Field(
        description=(
            "Whether the question is unambiguous and "
            "can be answered with a document search."
        ),
    )
    refined_queries: list[str] = Field(
        description=(
            "One or more self-contained, search-optimised "
            "reformulations of the original question."
        ),
    )
    clarification_prompt: str = Field(
        description=(
            "If the question is unclear, an explanation "
            "of what additional information is needed. "
            "Empty string when the question is clear."
        ),
    )
