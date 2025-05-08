"""Graph node implementations for the reasoning pipeline.

Each function receives the current state and any injected
dependencies (LLM, tools) and returns a state update dict.
"""

from __future__ import annotations

from typing import Literal

import tiktoken
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.types import Command

from docmind.agent.prompts import (
    answer_aggregation_prompt,
    context_distillation_prompt,
    dialogue_condensation_prompt,
    emergency_synthesis_prompt,
    question_refinement_prompt,
    research_coordinator_prompt,
)
from docmind.agent.schemas import QuestionAssessment
from docmind.agent.state import ReasonerState, WorkflowState
from docmind.settings import get_settings

# -------------------------------------------------------------------
# Token estimation
# -------------------------------------------------------------------

_ENCODING: tiktoken.Encoding | None = None


def _count_tokens(messages: list) -> int:
    """Rough token count across a list of LangChain messages."""
    global _ENCODING
    if _ENCODING is None:
        try:
            _ENCODING = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            _ENCODING = tiktoken.get_encoding("cl100k_base")

    total = 0
    for msg in messages:
        content = getattr(msg, "content", None)
        if content:
            total += len(_ENCODING.encode(str(content)))
    return total


# -------------------------------------------------------------------
# Top-level workflow nodes
# -------------------------------------------------------------------


def condense_dialogue(state: WorkflowState, llm: object) -> dict:
    """Summarise recent conversation to keep context lean."""
    if len(state["messages"]) < 4:
        return {"dialogue_summary": ""}

    turns = [
        m
        for m in state["messages"][:-1]
        if isinstance(m, (HumanMessage, AIMessage))
        and not getattr(m, "tool_calls", None)
    ]
    if not turns:
        return {"dialogue_summary": ""}

    history = "Dialogue:\n"
    for msg in turns[-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history += f"{role}: {msg.content}\n"

    result = llm.with_config(temperature=0.2).invoke(
        [
            SystemMessage(content=dialogue_condensation_prompt()),
            HumanMessage(content=history),
        ]
    )
    return {
        "dialogue_summary": result.content,
        "collected_answers": [{"__clear__": True}],
    }


def refine_question(state: WorkflowState, llm: object) -> dict:
    """Assess clarity and rewrite the user's question."""
    latest = state["messages"][-1]
    summary = state.get("dialogue_summary", "")

    context_block = ""
    if summary.strip():
        context_block = f"Conversation context:\n{summary}\n"
    context_block += f"User query:\n{latest.content}\n"

    structured_llm = llm.with_config(temperature=0.1).with_structured_output(
        QuestionAssessment
    )

    assessment = structured_llm.invoke(
        [
            SystemMessage(content=question_refinement_prompt()),
            HumanMessage(content=context_block),
        ]
    )

    if assessment.refined_queries and assessment.is_clear:
        purge = [
            RemoveMessage(id=m.id)
            for m in state["messages"]
            if not isinstance(m, SystemMessage)
        ]
        return {
            "query_is_clear": True,
            "messages": purge,
            "raw_query": latest.content,
            "refined_questions": assessment.refined_queries,
        }

    fallback = assessment.clarification_prompt
    if not fallback or len(fallback.strip()) <= 10:
        fallback = "Could you provide more detail so I can search effectively?"
    return {
        "query_is_clear": False,
        "messages": [AIMessage(content=fallback)],
    }


def await_clarification(state: WorkflowState) -> dict:
    """No-op node; graph interrupts before it runs."""
    return {}


# -------------------------------------------------------------------
# Per-question reasoner nodes
# -------------------------------------------------------------------


def research_coordinator(state: ReasonerState, llm_with_tools: object) -> dict:
    """Drive the tool-calling loop for a single question."""
    prior = state.get("distilled_context", "").strip()
    sys_msg = SystemMessage(content=research_coordinator_prompt())
    prior_injection = (
        [
            HumanMessage(content=("[PRIOR RESEARCH SUMMARY]\n\n" + prior)),
        ]
        if prior
        else []
    )

    if not state.get("messages"):
        user_msg = HumanMessage(content=state["question_text"])
        nudge = HumanMessage(
            content=(
                "Begin by calling 'find_relevant_segments' "
                "to locate evidence for this question."
            ),
        )
        reply = llm_with_tools.invoke(
            [sys_msg, *prior_injection, user_msg, nudge]
        )
        return {
            "messages": [user_msg, reply],
            "tool_call_tally": len(reply.tool_calls or []),
            "cycle_count": 1,
        }

    reply = llm_with_tools.invoke(
        [sys_msg] + prior_injection + state["messages"]
    )
    calls = reply.tool_calls if hasattr(reply, "tool_calls") else []
    return {
        "messages": [reply],
        "tool_call_tally": len(calls) if calls else 0,
        "cycle_count": 1,
    }


def best_effort_reply(state: ReasonerState, llm: object) -> dict:
    """Generate an answer from whatever data is available."""
    seen: set[str] = set()
    unique_tool_output: list[str] = []
    for m in state["messages"]:
        if isinstance(m, ToolMessage) and m.content not in seen:
            unique_tool_output.append(m.content)
            seen.add(m.content)

    prior = state.get("distilled_context", "").strip()

    sections: list[str] = []
    if prior:
        sections.append("## Prior Research Summary\n\n" + prior)
    if unique_tool_output:
        body = "\n\n".join(
            f"--- SOURCE {i} ---\n{c}"
            for i, c in enumerate(unique_tool_output, 1)
        )
        sections.append("## Current Retrieval Data\n\n" + body)

    evidence = (
        "\n\n".join(sections)
        if sections
        else "No data was retrieved from the documents."
    )

    prompt_body = (
        f"USER QUERY: {state.get('question_text')}\n\n"
        f"{evidence}\n\n"
        "Provide the best answer you can from the data above."
    )
    result = llm.invoke(
        [
            SystemMessage(content=emergency_synthesis_prompt()),
            HumanMessage(content=prompt_body),
        ]
    )
    return {"messages": [result]}


def evaluate_context_budget(
    state: ReasonerState,
) -> Command[Literal["distill_context", "research_coordinator"]]:
    """Decide whether to compress context or continue."""
    cfg = get_settings()
    msgs = state["messages"]

    new_keys: set[str] = set()
    for msg in reversed(msgs):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                name = tc["name"]
                args = tc.get("args", {})
                if name == "load_full_segment":
                    sid = args.get("segment_id") or args.get("id") or ""
                    if sid:
                        new_keys.add(f"seg::{sid}")
                elif name == "find_relevant_segments":
                    q = args.get("query", "")
                    if q:
                        new_keys.add(f"q::{q}")
            break

    merged = state.get("seen_retrievals", set()) | new_keys

    msg_tokens = _count_tokens(msgs)
    summary_tokens = _count_tokens(
        [HumanMessage(content=state.get("distilled_context", ""))]
    )
    total = msg_tokens + summary_tokens
    ceiling = cfg.base_token_budget + int(
        summary_tokens * cfg.token_growth_rate
    )

    target = "distill_context" if total > ceiling else "research_coordinator"
    return Command(update={"seen_retrievals": merged}, goto=target)


def distill_context(state: ReasonerState, llm: object) -> dict:
    """Compress the conversation into a dense summary."""
    msgs = state["messages"]
    existing = state.get("distilled_context", "").strip()

    if not msgs:
        return {}

    text = (
        f"USER QUESTION:\n{state.get('question_text')}\n\n"
        "Content to compress:\n\n"
    )
    if existing:
        text += f"[PRIOR COMPRESSED CONTEXT]\n{existing}\n\n"

    for msg in msgs[1:]:
        if isinstance(msg, AIMessage):
            tool_info = ""
            if getattr(msg, "tool_calls", None):
                calls = ", ".join(
                    f"{tc['name']}({tc['args']})" for tc in msg.tool_calls
                )
                tool_info = f" | Tools: {calls}"
            text += (
                f"[AGENT{tool_info}]\n{msg.content or '(tool invocation)'}\n\n"
            )
        elif isinstance(msg, ToolMessage):
            label = getattr(msg, "name", "tool")
            text += f"[TOOL RESULT — {label}]\n{msg.content}\n\n"

    summary = llm.invoke(
        [
            SystemMessage(content=context_distillation_prompt()),
            HumanMessage(content=text),
        ]
    ).content

    retrieved: set[str] = state.get("seen_retrievals", set())
    if retrieved:
        seg_ids = sorted(r for r in retrieved if r.startswith("seg::"))
        queries = sorted(
            r.removeprefix("q::") for r in retrieved if r.startswith("q::")
        )
        appendix = "\n\n---\n**Already executed (do NOT repeat):**\n"
        if seg_ids:
            appendix += (
                "Segments fetched:\n"
                + "\n".join(f"- {s.removeprefix('seg::')}" for s in seg_ids)
                + "\n"
            )
        if queries:
            appendix += (
                "Searches run:\n" + "\n".join(f"- {q}" for q in queries) + "\n"
            )
        summary += appendix

    removals = [RemoveMessage(id=m.id) for m in msgs[1:]]
    return {
        "distilled_context": summary,
        "messages": removals,
    }


def gather_response(state: ReasonerState) -> dict:
    """Collect the final answer from the last message."""
    last = state["messages"][-1]
    valid = (
        isinstance(last, (AIMessage, AIMessageChunk))
        and last.content
        and not getattr(last, "tool_calls", None)
    )
    answer = last.content if valid else "Unable to produce an answer."
    return {
        "concluded_answer": answer,
        "collected_answers": [
            {
                "index": state["question_idx"],
                "question": state["question_text"],
                "answer": answer,
            }
        ],
    }


# -------------------------------------------------------------------
# Aggregation
# -------------------------------------------------------------------


def synthesize_responses(state: WorkflowState, llm: object) -> dict:
    """Merge answers from parallel reasoners."""
    answers = state.get("collected_answers", [])
    if not answers:
        return {"messages": [AIMessage(content="No answers were produced.")]}

    ordered = sorted(answers, key=lambda a: a["index"])
    formatted = ""
    for i, entry in enumerate(ordered, 1):
        formatted += f"\nAnswer {i}:\n{entry['answer']}\n"

    user_msg = HumanMessage(
        content=(
            f"Original question: {state['raw_query']}\n"
            f"Retrieved answers:{formatted}"
        )
    )
    merged = llm.invoke(
        [
            SystemMessage(content=answer_aggregation_prompt()),
            user_msg,
        ]
    )
    return {"messages": [AIMessage(content=merged.content)]}
