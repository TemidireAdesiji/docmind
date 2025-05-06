"""Conditional edge functions for the reasoning graph."""

from __future__ import annotations

from typing import Literal

from langgraph.types import Send

from docmind.agent.state import ReasonerState, WorkflowState
from docmind.settings import get_settings


def dispatch_after_refinement(
    state: WorkflowState,
) -> Literal["await_clarification"] | list[Send]:
    """Route refined questions to parallel reasoners, or ask
    the user for more detail if the query is unclear.
    """
    if not state.get("query_is_clear", False):
        return "await_clarification"

    return [
        Send(
            "reasoner",
            {
                "question_text": q,
                "question_idx": idx,
                "messages": [],
            },
        )
        for idx, q in enumerate(state["refined_questions"])
    ]


def decide_next_action(
    state: ReasonerState,
) -> Literal["tools", "best_effort_reply", "gather_response"]:
    """After the coordinator replies, decide what happens next.

    * If it made tool calls → execute them.
    * If iteration/tool limits are exceeded → fall back.
    * Otherwise → collect the final answer.
    """
    cfg = get_settings()
    cycles = state.get("cycle_count", 0)
    calls = state.get("tool_call_tally", 0)

    if cycles >= cfg.max_reasoning_cycles or calls > cfg.max_tool_invocations:
        return "best_effort_reply"

    last = state["messages"][-1]
    pending = getattr(last, "tool_calls", None) or []

    if not pending:
        return "gather_response"

    return "tools"
