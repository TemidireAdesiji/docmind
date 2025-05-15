"""Compile the two-tier LangGraph reasoning pipeline.

The outer graph handles dialogue condensation, question
refinement, fan-out, and answer synthesis.  The inner
sub-graph handles per-question research with tool calling,
context distillation, and answer collection.
"""

from __future__ import annotations

from functools import partial

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from docmind.agent.nodes import (
    await_clarification,
    best_effort_reply,
    condense_dialogue,
    distill_context,
    evaluate_context_budget,
    gather_response,
    refine_question,
    research_coordinator,
    synthesize_responses,
)
from docmind.agent.routing import (
    decide_next_action,
    dispatch_after_refinement,
)
from docmind.agent.state import ReasonerState, WorkflowState
from docmind.log import logger


def build_reasoning_graph(llm: object, tool_list: list) -> StateGraph:
    """Construct, compile, and return the full agent graph.

    Parameters
    ----------
    llm:
        A LangChain chat model (e.g. ``ChatOllama``).
    tool_list:
        Retrieval tools created by ``RetrievalToolkit``.

    Returns
    -------
    CompiledStateGraph
        Ready to ``.stream()`` or ``.invoke()``.
    """
    armed_llm = llm.bind_tools(tool_list)
    tool_executor = ToolNode(tool_list)
    saver = InMemorySaver()

    # -- inner: per-question reasoner --------------------------------
    inner = StateGraph(ReasonerState)
    inner.add_node(
        "research_coordinator",
        partial(research_coordinator, llm_with_tools=armed_llm),
    )
    inner.add_node("tools", tool_executor)
    inner.add_node(evaluate_context_budget)
    inner.add_node(
        "distill_context",
        partial(distill_context, llm=llm),
    )
    inner.add_node(
        "best_effort_reply",
        partial(best_effort_reply, llm=llm),
    )
    inner.add_node(gather_response)

    inner.add_edge(START, "research_coordinator")
    inner.add_conditional_edges(
        "research_coordinator",
        decide_next_action,
        {
            "tools": "tools",
            "best_effort_reply": "best_effort_reply",
            "gather_response": "gather_response",
        },
    )
    inner.add_edge("tools", "evaluate_context_budget")
    inner.add_edge("distill_context", "research_coordinator")
    inner.add_edge("best_effort_reply", "gather_response")
    inner.add_edge("gather_response", END)

    reasoner_subgraph = inner.compile()

    # -- outer: orchestration ----------------------------------------
    outer = StateGraph(WorkflowState)
    outer.add_node(
        "condense_dialogue",
        partial(condense_dialogue, llm=llm),
    )
    outer.add_node(
        "refine_question",
        partial(refine_question, llm=llm),
    )
    outer.add_node(await_clarification)
    outer.add_node("reasoner", reasoner_subgraph)
    outer.add_node(
        "synthesize_responses",
        partial(synthesize_responses, llm=llm),
    )

    outer.add_edge(START, "condense_dialogue")
    outer.add_edge("condense_dialogue", "refine_question")
    outer.add_conditional_edges("refine_question", dispatch_after_refinement)
    outer.add_edge("await_clarification", "refine_question")
    outer.add_edge(["reasoner"], "synthesize_responses")
    outer.add_edge("synthesize_responses", END)

    compiled = outer.compile(
        checkpointer=saver,
        interrupt_before=["await_clarification"],
    )

    logger.info("Reasoning graph compiled successfully")
    return compiled
