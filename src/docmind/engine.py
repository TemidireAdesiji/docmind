"""Top-level knowledge engine that wires all components.

``KnowledgeEngine`` is the single entry point for bootstrapping
the full pipeline: vector index, chunk repository, splitter,
ingestion, LLM, tools, and the compiled reasoning graph.
"""

from __future__ import annotations

import uuid
from typing import Any

from langchain_ollama import ChatOllama

from docmind.agent.graph import build_reasoning_graph
from docmind.agent.tools import RetrievalToolkit
from docmind.chunking.splitter import HierarchicalSplitter
from docmind.log import logger
from docmind.observability.tracing import TracingBridge
from docmind.settings import Settings, get_settings
from docmind.storage.document import ChunkRepository
from docmind.storage.vector import VectorIndex


class KnowledgeEngine:
    """Assemble and manage the full RAG pipeline."""

    def __init__(self, cfg: Settings | None = None) -> None:
        self._cfg = cfg or get_settings()
        self._cfg.paths.ensure_dirs()

        self.vector_index = VectorIndex(self._cfg)
        self.chunk_repo = ChunkRepository(self._cfg)
        self.splitter = HierarchicalSplitter(self._cfg)
        self.tracing = TracingBridge(self._cfg)

        self.compiled_graph: Any = None
        self._thread_id = str(uuid.uuid4())

    # -- lifecycle ---------------------------------------------------

    def start(self) -> None:
        """Create the collection, build the graph, and go."""
        self.vector_index.ensure_collection()

        store = self.vector_index.get_store()
        llm = ChatOllama(
            model=self._cfg.llm_model,
            temperature=self._cfg.llm_temperature,
        )
        toolkit = RetrievalToolkit(store)
        tools = toolkit.build()

        self.compiled_graph = build_reasoning_graph(llm, tools)
        logger.info("KnowledgeEngine ready")

    def run_config(self) -> dict[str, Any]:
        """Return the LangGraph invocation config."""
        cfg: dict[str, Any] = {
            "configurable": {"thread_id": self._thread_id},
            "recursion_limit": self._cfg.graph_depth_limit,
        }
        handler = self.tracing.callback
        if handler:
            cfg["callbacks"] = [handler]
        return cfg

    def new_thread(self) -> None:
        """Discard current conversation and start fresh."""
        if self.compiled_graph is not None:
            try:
                self.compiled_graph.checkpointer.delete_thread(self._thread_id)
            except Exception:
                logger.debug(
                    "Could not delete thread {}",
                    self._thread_id,
                )
        self._thread_id = str(uuid.uuid4())
        logger.debug("New thread: {}", self._thread_id)
