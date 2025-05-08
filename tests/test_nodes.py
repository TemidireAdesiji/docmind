"""Tests for agent graph nodes."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from docmind.agent.nodes import (
    condense_dialogue,
    gather_response,
    synthesize_responses,
)


class TestCondenseDialogue:
    """Verify conversation condensation node."""

    def test_short_history_returns_empty(self, mock_llm: MagicMock) -> None:
        state = {"messages": [HumanMessage(content="Hi")]}
        result = condense_dialogue(state, mock_llm)
        assert result["dialogue_summary"] == ""

    def test_long_history_invokes_llm(self, mock_llm: MagicMock) -> None:
        msgs = [
            HumanMessage(content="Q1"),
            AIMessage(content="A1"),
            HumanMessage(content="Q2"),
            AIMessage(content="A2"),
            HumanMessage(content="Q3"),
        ]
        result = condense_dialogue({"messages": msgs}, mock_llm)
        assert result["dialogue_summary"] != ""
        mock_llm.with_config.assert_called()


class TestGatherResponse:
    """Verify answer collection from final message."""

    def test_valid_ai_message(self) -> None:
        msg = AIMessage(content="The answer is 42.")
        state = {
            "messages": [msg],
            "question_idx": 0,
            "question_text": "What is it?",
        }
        result = gather_response(state)
        assert result["concluded_answer"] == "The answer is 42."
        assert result["collected_answers"][0]["index"] == 0

    def test_tool_call_message_falls_back(self) -> None:
        msg = MagicMock()
        msg.content = "partial"
        msg.tool_calls = [{"name": "search"}]
        # isinstance check will fail for MagicMock
        state = {
            "messages": [msg],
            "question_idx": 1,
            "question_text": "Test?",
        }
        result = gather_response(state)
        assert "Unable" in result["concluded_answer"]


class TestSynthesizeResponses:
    """Verify multi-answer aggregation."""

    def test_no_answers_returns_message(self, mock_llm: MagicMock) -> None:
        state = {
            "collected_answers": [],
            "raw_query": "test",
        }
        result = synthesize_responses(state, mock_llm)
        assert "No answers" in result["messages"][0].content

    def test_aggregates_multiple_answers(self, mock_llm: MagicMock) -> None:
        state = {
            "collected_answers": [
                {
                    "index": 0,
                    "question": "Q1",
                    "answer": "A1",
                },
                {
                    "index": 1,
                    "question": "Q2",
                    "answer": "A2",
                },
            ],
            "raw_query": "multi question",
        }
        result = synthesize_responses(state, mock_llm)
        mock_llm.invoke.assert_called_once()
        assert len(result["messages"]) == 1
