"""Hierarchical parent / child document splitter.

Documents are first split along Markdown headings into
*parent* segments (coarse, context-rich), then each parent is
further divided into *child* segments (small, precise) for
vector indexing.  Small parents are merged with neighbours;
oversized parents are subdivided — guaranteeing every parent
falls within a configurable size window.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Sequence
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from docmind.log import logger
from docmind.settings import Settings, get_settings

ParentPair = tuple[str, Document]


class HierarchicalSplitter:
    """Split Markdown into parent / child segment pairs."""

    def __init__(self, cfg: Settings | None = None) -> None:
        self._cfg = cfg or get_settings()

        self._heading_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self._cfg.heading_levels,
            strip_headers=False,
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._cfg.child_segment_size,
            chunk_overlap=self._cfg.child_segment_overlap,
        )

    # -- public API --------------------------------------------------

    def split_file(
        self, md_path: str | Path
    ) -> tuple[list[ParentPair], list[Document]]:
        """Split a single Markdown file.

        Returns
        -------
        tuple
            ``(parent_pairs, child_docs)`` where each parent
            pair is ``(parent_id, Document)``.
        """
        md_path = Path(md_path)
        if not md_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {md_path}")

        text = md_path.read_text(encoding="utf-8")
        raw_parents = self._heading_splitter.split_text(text)
        if not raw_parents:
            logger.warning(
                "No headings found in {}; treating entire file as one parent",
                md_path.name,
            )
            raw_parents = [
                Document(
                    page_content=text,
                    metadata={"source": md_path.stem + ".pdf"},
                )
            ]

        refined = self._consolidate(raw_parents)

        parents: list[ParentPair] = []
        children: list[Document] = []
        self._emit_children(parents, children, refined, md_path)
        return parents, children

    def split_directory(
        self, directory: str | Path | None = None
    ) -> tuple[list[ParentPair], list[Document]]:
        """Split every ``*.md`` file in *directory*."""
        directory = Path(directory or self._cfg.paths.converted_docs)
        all_parents: list[ParentPair] = []
        all_children: list[Document] = []

        for fp in sorted(glob.glob(os.path.join(directory, "*.md"))):
            p, c = self.split_file(fp)
            all_parents.extend(p)
            all_children.extend(c)

        return all_parents, all_children

    # -- internal helpers --------------------------------------------

    def _consolidate(self, chunks: Sequence[Document]) -> list[Document]:
        """Merge small chunks and split large ones."""
        merged = self._merge_undersized(list(chunks))
        expanded = self._subdivide_oversized(merged)
        return self._absorb_runts(expanded)

    def _merge_undersized(self, chunks: list[Document]) -> list[Document]:
        if not chunks:
            return []

        result: list[Document] = []
        accumulator: Document | None = None

        for chunk in chunks:
            if accumulator is None:
                accumulator = chunk
            else:
                accumulator.page_content += "\n\n" + chunk.page_content
                self._merge_meta(accumulator, chunk)

            if len(accumulator.page_content) >= self._cfg.min_parent_length:
                result.append(accumulator)
                accumulator = None

        if accumulator is not None:
            if result:
                result[-1].page_content += "\n\n" + accumulator.page_content
                self._merge_meta(result[-1], accumulator)
            else:
                result.append(accumulator)

        return result

    def _subdivide_oversized(self, chunks: list[Document]) -> list[Document]:
        output: list[Document] = []
        for chunk in chunks:
            if len(chunk.page_content) <= self._cfg.max_parent_length:
                output.append(chunk)
            else:
                sub = RecursiveCharacterTextSplitter(
                    chunk_size=self._cfg.max_parent_length,
                    chunk_overlap=self._cfg.child_segment_overlap,
                ).split_documents([chunk])
                output.extend(sub)
        return output

    def _absorb_runts(self, chunks: list[Document]) -> list[Document]:
        """Absorb any remaining below-minimum chunks."""
        cleaned: list[Document] = []
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            if len(chunk.page_content) < self._cfg.min_parent_length:
                if cleaned:
                    cleaned[-1].page_content += "\n\n" + chunk.page_content
                    self._merge_meta(cleaned[-1], chunk)
                elif i < len(chunks) - 1:
                    nxt = chunks[i + 1]
                    nxt.page_content = (
                        chunk.page_content + "\n\n" + nxt.page_content
                    )
                    self._merge_meta(nxt, chunk, prepend=True)
                else:
                    cleaned.append(chunk)
            else:
                cleaned.append(chunk)
            i += 1
        return cleaned

    def _emit_children(
        self,
        parent_pairs: list[ParentPair],
        child_docs: list[Document],
        parents: list[Document],
        md_path: Path,
    ) -> None:
        for idx, parent in enumerate(parents):
            pid = f"{md_path.stem}_segment_{idx}"
            parent.metadata.update(
                {
                    "source": md_path.stem + ".pdf",
                    "parent_id": pid,
                }
            )
            parent_pairs.append((pid, parent))
            child_docs.extend(self._child_splitter.split_documents([parent]))

    @staticmethod
    def _merge_meta(
        target: Document,
        source: Document,
        *,
        prepend: bool = False,
    ) -> None:
        for key, val in source.metadata.items():
            if key in target.metadata:
                if prepend:
                    target.metadata[key] = f"{val} -> {target.metadata[key]}"
                else:
                    target.metadata[key] = f"{target.metadata[key]} -> {val}"
            else:
                target.metadata[key] = val
