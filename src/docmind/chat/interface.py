"""Streaming conversation handler for the Gradio UI.

Translates raw LangGraph stream events into a list of
Gradio-compatible message dicts, handling collapsible
system nodes, tool-call status, and incremental LLM tokens.
"""

from __future__ import annotations

import json
import re
from collections.abc import Generator
from typing import Any

from langchain_core.messages import (
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)

from docmind.log import logger

_HIDDEN_NODES = {"refine_question"}
_SYSTEM_NODES = {"condense_dialogue", "refine_question"}

_NODE_TITLES: dict[str, str] = {
    "refine_question": "Query Analysis & Rewriting",
    "condense_dialogue": "Dialogue Condensation",
}

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _make_msg(
    content: str,
    *,
    title: str | None = None,
    node: str | None = None,
) -> dict[str, Any]:
    msg: dict[str, Any] = {
        "role": "assistant",
        "content": content,
    }
    meta = {k: v for k, v in {"title": title, "node": node}.items() if v}
    if meta:
        msg["metadata"] = meta
    return msg


def _idx_by_node(messages: list[dict[str, Any]], node: str) -> int | None:
    for i, m in enumerate(messages):
        if m.get("metadata", {}).get("node") == node:
            return i
    return None


def _try_parse_assessment(buf: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*}", buf, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except (json.JSONDecodeError, ValueError):
        return None


def _format_assessment(buf: str) -> str:
    data = _try_parse_assessment(buf)
    if data is None:
        return "Analysing query …"
    if data.get("is_clear"):
        lines = ["**Query is clear**"]
        if data.get("refined_queries"):
            lines.append("\n**Refined queries:**")
            lines.extend(f"- {q}" for q in data["refined_queries"])
    else:
        lines = ["**Query needs clarification**"]
        note = data.get("clarification_prompt", "")
        if note and note.strip().lower() != "no":
            lines.append(f"\n{note}")
    return "\n".join(lines)


# -------------------------------------------------------------------
# Main handler
# -------------------------------------------------------------------


class ConversationHandler:
    """Stream agent responses into Gradio message format."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def stream_reply(
        self,
        user_text: str,
        history: list,
    ) -> Generator[list[dict[str, Any]], None, None]:
        """Yield incremental message lists for Gradio."""
        graph = self._engine.compiled_graph
        if graph is None:
            yield [_make_msg("System not initialised.")]
            return

        run_cfg = self._engine.run_config()
        current = graph.get_state(run_cfg)

        try:
            if current.next:
                graph.update_state(
                    run_cfg,
                    {"messages": [HumanMessage(content=user_text.strip())]},
                )
                stream_input = None
            else:
                stream_input = {
                    "messages": [HumanMessage(content=user_text.strip())]
                }

            out: list[dict[str, Any]] = []
            tool_map: dict[str, int] = {}
            sys_buf: dict[str, str] = {}

            for chunk, meta in graph.stream(
                stream_input,
                config=run_cfg,
                stream_mode="messages",
            ):
                node = meta.get("langgraph_node", "")

                if (
                    node in _SYSTEM_NODES
                    and isinstance(chunk, AIMessageChunk)
                    and chunk.content
                ):
                    self._handle_system(chunk, node, out, sys_buf)

                elif hasattr(chunk, "tool_calls") and chunk.tool_calls:
                    self._handle_tool_start(chunk, out, tool_map)

                elif isinstance(chunk, ToolMessage):
                    self._handle_tool_end(chunk, out, tool_map)

                elif (
                    isinstance(chunk, AIMessageChunk)
                    and chunk.content
                    and node not in _HIDDEN_NODES
                ):
                    self._handle_token(chunk, out)

                yield out

        except Exception:
            logger.exception("Streaming error")
            yield [_make_msg("An error occurred.")]

    def reset_session(self) -> None:
        """Clear conversation thread state."""
        self._engine.new_thread()
        self._engine.tracing.flush()

    # -- internal stream handlers ------------------------------------

    @staticmethod
    def _handle_system(
        chunk: AIMessageChunk,
        node: str,
        out: list[dict[str, Any]],
        buf: dict[str, str],
    ) -> None:
        buf[node] = buf.get(node, "") + chunk.content
        title = _NODE_TITLES.get(node, node)
        content = (
            _format_assessment(buf[node])
            if node == "refine_question"
            else buf[node]
        )
        idx = _idx_by_node(out, node)
        if idx is None:
            out.append(_make_msg(content, title=title, node=node))
        else:
            out[idx]["content"] = content

        if node == "refine_question":
            ConversationHandler._surface_clarification(buf[node], out)

    @staticmethod
    def _surface_clarification(buf: str, out: list[dict[str, Any]]) -> None:
        data = _try_parse_assessment(buf) or {}
        note = data.get("clarification_prompt", "")
        if not data.get("is_clear") and note.strip().lower() not in ("", "no"):
            idx = _idx_by_node(out, "clarification")
            if idx is None:
                out.append(_make_msg(note, node="clarification"))
            else:
                out[idx]["content"] = note

    @staticmethod
    def _handle_tool_start(
        chunk: AIMessageChunk,
        out: list[dict[str, Any]],
        tool_map: dict[str, int],
    ) -> None:
        for tc in chunk.tool_calls:
            tid = tc.get("id")
            if tid and tid not in tool_map:
                out.append(
                    _make_msg(
                        f"Running `{tc['name']}` …",
                        title=tc["name"],
                    )
                )
                tool_map[tid] = len(out) - 1

    @staticmethod
    def _handle_tool_end(
        chunk: ToolMessage,
        out: list[dict[str, Any]],
        tool_map: dict[str, int],
    ) -> None:
        idx = tool_map.get(chunk.tool_call_id)
        if idx is not None:
            preview = str(chunk.content)[:300]
            ellipsis = "\n…" if len(str(chunk.content)) > 300 else ""
            out[idx]["content"] = f"```\n{preview}{ellipsis}\n```"

    @staticmethod
    def _handle_token(
        chunk: AIMessageChunk,
        out: list[dict[str, Any]],
    ) -> None:
        last = out[-1] if out else None
        if not (
            last and last.get("role") == "assistant" and "metadata" not in last
        ):
            out.append(_make_msg(""))
        out[-1]["content"] += chunk.content
