"""Tests for prompt functions."""

from __future__ import annotations

from docmind.agent.prompts import (
    answer_aggregation_prompt,
    context_distillation_prompt,
    dialogue_condensation_prompt,
    emergency_synthesis_prompt,
    question_refinement_prompt,
    research_coordinator_prompt,
)

_ALL_PROMPTS = (
    dialogue_condensation_prompt,
    question_refinement_prompt,
    research_coordinator_prompt,
    emergency_synthesis_prompt,
    context_distillation_prompt,
    answer_aggregation_prompt,
)


class TestPromptIntegrity:
    """Verify all prompts are non-empty and well-formed."""

    def test_all_return_strings(self) -> None:
        for fn in _ALL_PROMPTS:
            result = fn()
            assert isinstance(result, str)
            assert len(result) > 50

    def test_no_placeholder_tokens(self) -> None:
        for fn in _ALL_PROMPTS:
            text = fn()
            assert "{{" not in text
            assert "}}" not in text
            assert "TODO" not in text

    def test_coordinator_mentions_tools(self) -> None:
        text = research_coordinator_prompt()
        assert "find_relevant_segments" in text
        assert "load_full_segment" in text

    def test_synthesis_mentions_sources(self) -> None:
        text = emergency_synthesis_prompt()
        assert "Sources" in text

    def test_aggregation_mentions_sources(self) -> None:
        text = answer_aggregation_prompt()
        assert "Sources" in text
