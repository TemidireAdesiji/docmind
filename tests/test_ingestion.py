"""Tests for the ingestion pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from docmind.ingestion.pipeline import IngestionOrchestrator
from docmind.settings import Settings


@pytest.fixture()
def mock_pipeline(
    tmp_path: Path,
) -> tuple[IngestionOrchestrator, MagicMock, MagicMock]:
    cfg = Settings(_env_file=None)

    vi = MagicMock()
    cr = MagicMock()
    sp = MagicMock()

    pipe = IngestionOrchestrator(vi, cr, sp, cfg)
    pipe._md_dir = tmp_path / "converted"
    pipe._md_dir.mkdir()

    return pipe, vi, sp


class TestIngest:
    """Verify ingestion orchestration logic."""

    def test_empty_input_returns_zero(
        self,
        mock_pipeline: tuple,
    ) -> None:
        pipe, _, _ = mock_pipeline
        added, skipped = pipe.ingest([])
        assert added == 0
        assert skipped == 0

    def test_unsupported_extension_skipped(
        self,
        mock_pipeline: tuple,
    ) -> None:
        pipe, _, _ = mock_pipeline
        added, skipped = pipe.ingest(["file.txt"])
        assert added == 0
        assert skipped == 0

    def test_duplicate_file_skipped(
        self,
        mock_pipeline: tuple,
        tmp_path: Path,
    ) -> None:
        pipe, _vi, _sp = mock_pipeline

        # Simulate already-converted file
        (pipe._md_dir / "existing.md").write_text("data")
        md_src = tmp_path / "existing.md"
        md_src.write_text("data")

        added, skipped = pipe.ingest([str(md_src)])
        assert added == 0
        assert skipped == 1

    def test_markdown_ingestion_calls_splitter(
        self,
        mock_pipeline: tuple,
        tmp_path: Path,
    ) -> None:
        pipe, vi, sp = mock_pipeline

        md = tmp_path / "new_doc.md"
        md.write_text("# Title\nSome content here.")

        from langchain_core.documents import Document

        sp.split_file.return_value = (
            [("seg_0", Document(page_content="p", metadata={}))],
            [Document(page_content="c", metadata={})],
        )
        mock_store = MagicMock()
        vi.get_store.return_value = mock_store

        added, _skipped = pipe.ingest([str(md)])
        assert added == 1
        sp.split_file.assert_called_once()
        mock_store.add_documents.assert_called_once()

    def test_string_input_accepted(
        self,
        mock_pipeline: tuple,
        tmp_path: Path,
    ) -> None:
        pipe, vi, sp = mock_pipeline
        md = tmp_path / "single.md"
        md.write_text("# Hello")

        from langchain_core.documents import Document

        sp.split_file.return_value = (
            [("s0", Document(page_content="p", metadata={}))],
            [Document(page_content="c", metadata={})],
        )
        vi.get_store.return_value = MagicMock()

        added, _ = pipe.ingest(str(md))
        assert added == 1


class TestListDocuments:
    """Verify document listing."""

    def test_empty_dir(self, mock_pipeline: tuple) -> None:
        pipe, _, _ = mock_pipeline
        assert pipe.list_documents() == []

    def test_lists_converted_files(self, mock_pipeline: tuple) -> None:
        pipe, _, _ = mock_pipeline
        (pipe._md_dir / "report.md").write_text("text")
        (pipe._md_dir / "manual.md").write_text("text")
        docs = pipe.list_documents()
        assert "manual.pdf" in docs
        assert "report.pdf" in docs


class TestWipe:
    """Verify complete data removal."""

    def test_wipe_recreates_directories(self, mock_pipeline: tuple) -> None:
        pipe, vi, _ = mock_pipeline
        (pipe._md_dir / "file.md").write_text("data")
        pipe.wipe()
        assert pipe._md_dir.exists()
        assert list(pipe._md_dir.iterdir()) == []
        vi.drop_collection.assert_called_once()
        vi.ensure_collection.assert_called_once()
