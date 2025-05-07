"""Tests for graph routing / edge functions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from docmind.agent.routing import (
    decide_next_action,
    dispatch_after_refinement,
)


class TestDispatchAfterRefinement:
    """Verify question-clarity routing."""

    def test_unclear_query_routes_to_clarification(
        self,
    ) -> None:
        state = {
            "query_is_clear": False,
            "refined_questions": [],
            "messages": [],
        }
        result = dispatch_after_refinement(state)
        assert result == "await_clarification"

    def test_clear_query_fans_out(self) -> None:
        state = {
            "query_is_clear": True,
            "refined_questions": ["Q1", "Q2"],
            "messages": [],
        }
        result = dispatch_after_refinement(state)
        assert len(result) == 2
        assert result[0].node == "reasoner"
        assert result[1].node == "reasoner"

    def test_single_question_sends_one(self) -> None:
        state = {
            "query_is_clear": True,
            "refined_questions": ["Only one"],
            "messages": [],
        }
        result = dispatch_after_refinement(state)
        assert len(result) == 1


class TestDecideNextAction:
    """Verify post-coordinator routing."""

    @patch("docmind.agent.routing.get_settings")
    def test_no_tool_calls_gathers(self, mock_cfg: MagicMock) -> None:
        mock_cfg.return_value.max_reasoning_cycles = 10
        mock_cfg.return_value.max_tool_invocations = 8
        msg = AIMessage(content="Here is the answer.")
        state = {
            "messages": [msg],
            "cycle_count": 1,
            "tool_call_tally": 0,
        }
        assert decide_next_action(state) == "gather_response"

    @patch("docmind.agent.routing.get_settings")
    def test_tool_calls_continue(self, mock_cfg: MagicMock) -> None:
        mock_cfg.return_value.max_reasoning_cycles = 10
        mock_cfg.return_value.max_tool_invocations = 8
        msg = MagicMock()
        msg.tool_calls = [{"id": "1", "name": "search"}]
        state = {
            "messages": [msg],
            "cycle_count": 1,
            "tool_call_tally": 1,
        }
        assert decide_next_action(state) == "tools"

    @patch("docmind.agent.routing.get_settings")
    def test_cycle_limit_triggers_fallback(self, mock_cfg: MagicMock) -> None:
        mock_cfg.return_value.max_reasoning_cycles = 5
        mock_cfg.return_value.max_tool_invocations = 8
        msg = MagicMock()
        msg.tool_calls = [{"id": "1", "name": "search"}]
        state = {
            "messages": [msg],
            "cycle_count": 5,
            "tool_call_tally": 2,
        }
        assert decide_next_action(state) == "best_effort_reply"

    @patch("docmind.agent.routing.get_settings")
    def test_tool_limit_triggers_fallback(self, mock_cfg: MagicMock) -> None:
        mock_cfg.return_value.max_reasoning_cycles = 10
        mock_cfg.return_value.max_tool_invocations = 3
        msg = MagicMock()
        msg.tool_calls = [{"id": "1", "name": "search"}]
        state = {
            "messages": [msg],
            "cycle_count": 2,
            "tool_call_tally": 4,
        }
        assert decide_next_action(state) == "best_effort_reply"
