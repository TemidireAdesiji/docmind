"""Tests for the hierarchical document splitter."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from docmind.chunking.splitter import HierarchicalSplitter
from docmind.settings import Settings


@pytest.fixture()
def splitter() -> HierarchicalSplitter:
    cfg = Settings(
        _env_file=None,
        child_segment_size=200,
        child_segment_overlap=30,
        min_parent_length=100,
        max_parent_length=3000,
    )
    return HierarchicalSplitter(cfg)


class TestSplitFile:
    """Verify single-file splitting behaviour."""

    def test_produces_parents_and_children(
        self,
        splitter: HierarchicalSplitter,
        sample_markdown: Path,
    ) -> None:
        parents, children = splitter.split_file(sample_markdown)
        assert len(parents) > 0
        assert len(children) > 0

    def test_parent_ids_are_unique(
        self,
        splitter: HierarchicalSplitter,
        sample_markdown: Path,
    ) -> None:
        parents, _ = splitter.split_file(sample_markdown)
        ids = [pid for pid, _ in parents]
        assert len(ids) == len(set(ids))

    def test_children_have_parent_id_metadata(
        self,
        splitter: HierarchicalSplitter,
        sample_markdown: Path,
    ) -> None:
        _, children = splitter.split_file(sample_markdown)
        for child in children:
            assert "parent_id" in child.metadata
            assert "source" in child.metadata

    def test_missing_file_raises(self, splitter: HierarchicalSplitter) -> None:
        with pytest.raises(FileNotFoundError):
            splitter.split_file("/nonexistent/path.md")

    def test_parent_source_ends_with_pdf(
        self,
        splitter: HierarchicalSplitter,
        sample_markdown: Path,
    ) -> None:
        parents, _ = splitter.split_file(sample_markdown)
        for _, doc in parents:
            assert doc.metadata["source"].endswith(".pdf")


class TestMergeLogic:
    """Verify undersized-chunk merging."""

    def test_small_chunks_get_merged(
        self, splitter: HierarchicalSplitter
    ) -> None:
        tiny = [
            Document(page_content="a" * 30, metadata={}),
            Document(page_content="b" * 30, metadata={}),
            Document(page_content="c" * 30, metadata={}),
        ]
        merged = splitter._merge_undersized(tiny)
        assert len(merged) <= len(tiny)
        total_chars = sum(len(d.page_content) for d in merged)
        assert total_chars >= 90

    def test_large_chunk_untouched(
        self, splitter: HierarchicalSplitter
    ) -> None:
        big = [Document(page_content="x" * 500, metadata={})]
        merged = splitter._merge_undersized(big)
        assert len(merged) == 1
        assert len(merged[0].page_content) == 500


class TestSubdivideOversized:
    """Verify that oversized parents are split."""

    def test_oversized_gets_split(self) -> None:
        cfg = Settings(
            _env_file=None,
            max_parent_length=200,
            child_segment_overlap=10,
            min_parent_length=50,
        )
        s = HierarchicalSplitter(cfg)
        big = [Document(page_content="word " * 200, metadata={})]
        result = s._subdivide_oversized(big)
        assert len(result) > 1

    def test_within_limit_unchanged(self) -> None:
        cfg = Settings(
            _env_file=None,
            max_parent_length=5000,
            min_parent_length=50,
        )
        s = HierarchicalSplitter(cfg)
        normal = [Document(page_content="ok " * 10, metadata={})]
        result = s._subdivide_oversized(normal)
        assert len(result) == 1


class TestSplitDirectory:
    """Verify batch splitting across a directory."""

    def test_empty_dir_returns_nothing(
        self, splitter: HierarchicalSplitter, tmp_path: Path
    ) -> None:
        parents, children = splitter.split_directory(tmp_path)
        assert parents == []
        assert children == []

    def test_processes_all_md_files(
        self,
        splitter: HierarchicalSplitter,
        tmp_path: Path,
        sample_markdown: Path,
    ) -> None:
        import shutil

        dest = tmp_path / "docs"
        dest.mkdir()
        shutil.copy(sample_markdown, dest / "a.md")
        shutil.copy(sample_markdown, dest / "b.md")
        parents, _children = splitter.split_directory(dest)
        sources = {doc.metadata["source"] for _, doc in parents}
        assert "a.pdf" in sources
        assert "b.pdf" in sources
