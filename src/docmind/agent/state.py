"""Graph state definitions for the reasoning pipeline.

Two state schemas are used:

* ``WorkflowState`` — top-level graph that handles history
  condensation, question refinement, fan-out, and answer
  synthesis.
* ``ReasonerState`` — per-question sub-graph that drives
  tool calls, context distillation, and answer collection.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from langgraph.graph import MessagesState


def _accumulate_or_clear(
    current: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reducer: append items, or reset when a sentinel is present."""
    if incoming and any(item.get("__clear__") for item in incoming):
        return []
    return current + incoming


def _union(a: set[str], b: set[str]) -> set[str]:
    return a | b


class WorkflowState(MessagesState):
    """Top-level orchestration state."""

    query_is_clear: bool = False
    dialogue_summary: str = ""
    raw_query: str = ""
    refined_questions: list[str] = []
    collected_answers: Annotated[
        list[dict[str, Any]], _accumulate_or_clear
    ] = []


class ReasonerState(MessagesState):
    """Per-question reasoning sub-graph state."""

    question_text: str = ""
    question_idx: int = 0
    distilled_context: str = ""
    seen_retrievals: Annotated[set[str], _union] = set()
    concluded_answer: str = ""
    collected_answers: list[dict[str, Any]] = []
    tool_call_tally: Annotated[int, operator.add] = 0
    cycle_count: Annotated[int, operator.add] = 0
