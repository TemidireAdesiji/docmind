"""End-to-end document ingestion: convert, split, index.

Orchestrates the conversion of raw documents (PDF or Markdown)
through the splitting and indexing pipeline.  Duplicate files
are detected and skipped automatically.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from docmind.chunking.splitter import HierarchicalSplitter
from docmind.conversion.pdf import pdf_to_markdown
from docmind.log import logger
from docmind.settings import Settings, get_settings
from docmind.storage.document import ChunkRepository
from docmind.storage.vector import VectorIndex

ProgressFn = Callable[[float, str], Any]

_ACCEPTED_SUFFIXES = {".pdf", ".md"}


class IngestionOrchestrator:
    """Coordinate document ingestion into the knowledge base."""

    def __init__(
        self,
        vector_index: VectorIndex,
        chunk_repo: ChunkRepository,
        splitter: HierarchicalSplitter,
        cfg: Settings | None = None,
    ) -> None:
        self._vector = vector_index
        self._chunks = chunk_repo
        self._splitter = splitter
        self._cfg = cfg or get_settings()
        self._md_dir = self._cfg.paths.converted_docs
        self._md_dir.mkdir(parents=True, exist_ok=True)

    # -- public API --------------------------------------------------

    def ingest(
        self,
        file_paths: str | list[str],
        *,
        on_progress: ProgressFn | None = None,
    ) -> tuple[int, int]:
        """Ingest one or more documents.

        Parameters
        ----------
        file_paths:
            Single path or list of paths (PDF or Markdown).
        on_progress:
            Optional ``(fraction, description)`` callback.

        Returns
        -------
        tuple[int, int]
            ``(added, skipped)`` counts.
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        valid = [
            p
            for p in file_paths
            if p and Path(p).suffix.lower() in _ACCEPTED_SUFFIXES
        ]
        if not valid:
            return 0, 0

        added = 0
        skipped = 0

        for idx, raw_path in enumerate(valid):
            if on_progress:
                name = Path(raw_path).name
                on_progress(
                    (idx + 1) / len(valid),
                    f"Processing {name}",
                )

            try:
                ok = self._ingest_one(raw_path)
                if ok:
                    added += 1
                else:
                    skipped += 1
            except Exception:
                logger.exception("Failed to ingest {}", raw_path)
                skipped += 1

        logger.info(
            "Ingestion complete — added={}, skipped={}",
            added,
            skipped,
        )
        return added, skipped

    def list_documents(self) -> list[str]:
        """Return display names of all ingested documents."""
        if not self._md_dir.exists():
            return []
        return sorted(p.stem + ".pdf" for p in self._md_dir.glob("*.md"))

    def wipe(self, collection_name: str | None = None) -> None:
        """Remove all ingested data and recreate storage."""
        if self._md_dir.exists():
            shutil.rmtree(self._md_dir)
            self._md_dir.mkdir(parents=True, exist_ok=True)

        self._chunks.purge()

        label = collection_name or self._cfg.collection_label
        self._vector.drop_collection(label)
        self._vector.ensure_collection(label)
        logger.info("Knowledge base wiped")

    # -- internal ----------------------------------------------------

    def _ingest_one(self, raw_path: str) -> bool:
        """Process a single document.  Return True if added."""
        src = Path(raw_path)
        stem = src.stem
        md_dest = self._md_dir / f"{stem}.md"

        if md_dest.exists():
            logger.debug("{} already ingested", src.name)
            return False

        if src.suffix.lower() == ".md":
            shutil.copy(src, md_dest)
        else:
            pdf_to_markdown(src, self._md_dir)

        parents, children = self._splitter.split_file(md_dest)
        if not children:
            logger.warning("No segments produced from {}", src.name)
            return False

        store = self._vector.get_store()
        store.add_documents(children)
        self._chunks.store_batch(parents)
        return True
