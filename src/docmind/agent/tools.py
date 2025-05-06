"""Retrieval tools exposed to the reasoning agent.

Two tools are provided:

* ``find_relevant_segments`` — hybrid search over child
  segments in the vector index.
* ``load_full_segment`` — fetch the broader parent segment
  from the file-backed store.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import tool

from docmind.storage.document import ChunkRepository

if TYPE_CHECKING:
    from langchain_qdrant import QdrantVectorStore


class RetrievalToolkit:
    """Build the set of retrieval tools for the agent."""

    def __init__(self, store: QdrantVectorStore) -> None:
        self._store = store
        self._chunk_repo = ChunkRepository()

    def build(self) -> list:
        """Return a list of LangChain tool objects."""
        search_fn = self._search_segments
        fetch_fn = self._fetch_full_segment

        search_tool = tool("find_relevant_segments")(search_fn)
        fetch_tool = tool("load_full_segment")(fetch_fn)
        return [search_tool, fetch_tool]

    # -- tool implementations ----------------------------------------

    def _search_segments(self, query: str, limit: int) -> str:
        """Search for the top-K most relevant segments.

        Args:
            query: Natural-language search query.
            limit: Maximum number of results to return.
        """
        try:
            hits = self._store.similarity_search(
                query, k=limit, score_threshold=0.7
            )
        except Exception as exc:
            return f"SEARCH_ERROR: {exc}"

        if not hits:
            return "NO_MATCHING_SEGMENTS"

        parts: list[str] = []
        for doc in hits:
            meta = doc.metadata
            parts.append(
                f"Segment ID: {meta.get('parent_id', '')}\n"
                f"File: {meta.get('source', '')}\n"
                f"Content: {doc.page_content.strip()}"
            )
        return "\n\n".join(parts)

    def _fetch_full_segment(self, segment_id: str) -> str:
        """Retrieve the full parent segment by its ID.

        Args:
            segment_id: The identifier of the parent segment.
        """
        try:
            data = self._chunk_repo.fetch(segment_id)
        except FileNotFoundError:
            return "SEGMENT_NOT_FOUND"
        except Exception as exc:
            return f"FETCH_ERROR: {exc}"

        return (
            f"Segment ID: {data['parent_id']}\n"
            f"File: "
            f"{data.get('metadata', {}).get('source', '?')}\n"
            f"Content: {data['content'].strip()}"
        )
