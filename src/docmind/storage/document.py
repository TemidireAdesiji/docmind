"""File-backed JSON store for parent document segments.

Each parent segment is persisted as a JSON file on disk,
keyed by its identifier.  This keeps the vector index lean
(only child segments are embedded) while providing fast
look-up of the broader context surrounding any match.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from docmind.log import logger
from docmind.settings import Settings, get_settings


class ChunkRepository:
    """Persist and retrieve parent document segments."""

    def __init__(self, cfg: Settings | None = None) -> None:
        self._cfg = cfg or get_settings()
        self._root = self._cfg.paths.stored_chunks
        self._root.mkdir(parents=True, exist_ok=True)

    # -- write -------------------------------------------------------

    def store(
        self,
        segment_id: str,
        content: str,
        metadata: dict[str, Any],
    ) -> None:
        """Persist a single parent segment."""
        if not segment_id:
            raise ValueError("segment_id must not be empty")
        dest = self._id_to_path(segment_id)
        payload = {
            "page_content": content,
            "metadata": metadata,
        }
        dest.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def store_batch(
        self,
        pairs: list[tuple[str, Any]],
    ) -> int:
        """Persist multiple ``(segment_id, Document)`` pairs.

        Returns the number of segments written.
        """
        count = 0
        for seg_id, doc in pairs:
            self.store(seg_id, doc.page_content, doc.metadata)
            count += 1
        return count

    # -- read --------------------------------------------------------

    def fetch(self, segment_id: str) -> dict[str, Any]:
        """Load a segment by ID.

        Returns
        -------
        dict
            Keys: ``content``, ``parent_id``, ``metadata``.

        Raises
        ------
        FileNotFoundError
            If no segment with *segment_id* exists.
        """
        path = self._id_to_path(segment_id)
        if not path.exists():
            raise FileNotFoundError(
                f"No stored segment with id '{segment_id}'"
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {
            "content": raw["page_content"],
            "parent_id": segment_id,
            "metadata": raw.get("metadata", {}),
        }

    def fetch_many(self, segment_ids: list[str]) -> list[dict[str, Any]]:
        """Load multiple segments, sorted by numeric suffix."""
        unique = sorted(set(segment_ids), key=self._sort_key)
        results: list[dict[str, Any]] = []
        for sid in unique:
            try:
                results.append(self.fetch(sid))
            except FileNotFoundError:
                logger.warning("Segment '{}' not found — skipping", sid)
        return results

    # -- maintenance -------------------------------------------------

    def purge(self) -> None:
        """Delete all stored segments."""
        if self._root.exists():
            shutil.rmtree(self._root)
        self._root.mkdir(parents=True, exist_ok=True)
        logger.info("Purged all stored segments")

    # -- helpers -----------------------------------------------------

    def _id_to_path(self, segment_id: str) -> Path:
        name = segment_id
        if not name.lower().endswith(".json"):
            name += ".json"
        return self._root / name

    @staticmethod
    def _sort_key(sid: str) -> int:
        match = re.search(r"_segment_(\d+)$", sid)
        return int(match.group(1)) if match else 0
