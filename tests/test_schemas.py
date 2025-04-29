"""Tests for Pydantic schema validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from docmind.agent.schemas import QuestionAssessment


class TestQuestionAssessment:
    """Verify structured output model."""

    def test_valid_clear_query(self) -> None:
        qa = QuestionAssessment(
            is_clear=True,
            refined_queries=["What is RAG?"],
            clarification_prompt="",
        )
        assert qa.is_clear is True
        assert len(qa.refined_queries) == 1

    def test_valid_unclear_query(self) -> None:
        qa = QuestionAssessment(
            is_clear=False,
            refined_queries=[],
            clarification_prompt="Please specify the topic.",
        )
        assert qa.is_clear is False
        assert qa.clarification_prompt != ""

    def test_multiple_refined_queries(self) -> None:
        qa = QuestionAssessment(
            is_clear=True,
            refined_queries=["Q1", "Q2", "Q3"],
            clarification_prompt="",
        )
        assert len(qa.refined_queries) == 3

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            QuestionAssessment(
                is_clear=True,
                # refined_queries missing
                clarification_prompt="",
            )
