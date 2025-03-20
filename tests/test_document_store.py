"""Tests for the file-backed chunk repository."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from docmind.settings import Settings
from docmind.storage.document import ChunkRepository


@pytest.fixture()
def repo(tmp_path: Path) -> ChunkRepository:
    cfg = Settings(_env_file=None)
    # Override internal path via object attribute
    r = ChunkRepository(cfg)
    r._root = tmp_path / "chunks"
    r._root.mkdir()
    return r


class TestStoreAndFetch:
    """Round-trip persistence tests."""

    def test_single_store_and_fetch(self, repo: ChunkRepository) -> None:
        repo.store("seg_0", "Hello world", {"source": "a.pdf"})
        result = repo.fetch("seg_0")
        assert result["content"] == "Hello world"
        assert result["parent_id"] == "seg_0"
        assert result["metadata"]["source"] == "a.pdf"

    def test_fetch_missing_raises(self, repo: ChunkRepository) -> None:
        with pytest.raises(FileNotFoundError):
            repo.fetch("nonexistent_id")

    def test_store_batch(self, repo: ChunkRepository) -> None:
        pairs = [
            (
                f"doc_segment_{i}",
                Document(
                    page_content=f"Content {i}",
                    metadata={"source": "b.pdf"},
                ),
            )
            for i in range(3)
        ]
        count = repo.store_batch(pairs)
        assert count == 3
        for i in range(3):
            data = repo.fetch(f"doc_segment_{i}")
            assert data["content"] == f"Content {i}"

    def test_empty_id_raises(self, repo: ChunkRepository) -> None:
        with pytest.raises(ValueError):
            repo.store("", "content", {})


class TestFetchMany:
    """Batch retrieval with deduplication and ordering."""

    def test_deduplicates_ids(self, repo: ChunkRepository) -> None:
        repo.store("x_segment_0", "A", {"source": "c.pdf"})
        results = repo.fetch_many(["x_segment_0", "x_segment_0"])
        assert len(results) == 1

    def test_sorted_by_numeric_suffix(self, repo: ChunkRepository) -> None:
        for i in [2, 0, 1]:
            repo.store(
                f"doc_segment_{i}",
                f"C{i}",
                {"source": "d.pdf"},
            )
        results = repo.fetch_many(
            [
                "doc_segment_2",
                "doc_segment_0",
                "doc_segment_1",
            ]
        )
        assert [r["parent_id"] for r in results] == [
            "doc_segment_0",
            "doc_segment_1",
            "doc_segment_2",
        ]

    def test_missing_ids_skipped(self, repo: ChunkRepository) -> None:
        repo.store("seg_segment_0", "OK", {"source": "e.pdf"})
        results = repo.fetch_many(["seg_segment_0", "missing_segment_99"])
        assert len(results) == 1


class TestPurge:
    """Verify complete data removal."""

    def test_purge_removes_all(self, repo: ChunkRepository) -> None:
        repo.store("a_segment_0", "x", {})
        repo.store("b_segment_1", "y", {})
        repo.purge()
        with pytest.raises(FileNotFoundError):
            repo.fetch("a_segment_0")
