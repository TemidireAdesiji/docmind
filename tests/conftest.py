"""Shared test fixtures and helpers."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from docmind.settings import Settings


@pytest.fixture()
def tmp_settings(tmp_path: Path) -> Settings:
    """Return a ``Settings`` instance with temp directories."""
    return Settings(
        _env_file=None,
        llm_model="test-model",
        llm_temperature=0,
        dense_embedding="test-dense",
        sparse_embedding="test-sparse",
    )


@pytest.fixture()
def sample_markdown(tmp_path: Path) -> Path:
    """Create a small Markdown file for splitting tests."""
    md = tmp_path / "sample.md"
    md.write_text(
        textwrap.dedent("""\
        # Introduction

        This is the introduction to our sample document.
        It contains enough text to serve as a meaningful
        parent segment for testing purposes. We need to
        ensure the text exceeds the minimum parent length
        so that the splitter treats it as a valid unit.
        Additional filler content is provided here to push
        the character count past the configured threshold.
        Lorem ipsum dolor sit amet, consectetur adipiscing
        elit. Sed do eiusmod tempor incididunt ut labore et
        dolore magna aliqua. Ut enim ad minim veniam, quis
        nostrud exercitation ullamco laboris nisi ut aliquip
        ex ea commodo consequat. Duis aute irure dolor in
        reprehenderit in voluptate velit esse cillum dolore
        eu fugiat nulla pariatur. Excepteur sint occaecat
        cupidatat non proident, sunt in culpa qui officia
        deserunt mollit anim id est laborum. More text is
        needed to cross the two-thousand character boundary
        so the merger logic can be exercised correctly.
        Padding padding padding padding padding padding.
        Padding padding padding padding padding padding.
        Padding padding padding padding padding padding.
        Padding padding padding padding padding padding.
        Padding padding padding padding padding padding.
        Padding padding padding padding padding padding.
        Padding padding padding padding padding padding.
        Padding padding padding padding padding padding.
        Padding padding padding padding padding padding.
        Padding padding padding padding padding padding.
        Padding padding padding padding padding padding.

        ## Details

        Here we discuss the details of the document in a
        separate section.  This section also needs to have
        substantial content to form a valid parent segment.
        We will add technical details and domain-specific
        terminology to make the test data realistic.
        Configuration parameters include chunk_size=500,
        overlap=100, and various tuning knobs for the
        hierarchical splitting pipeline.  Each parameter
        affects how the document is processed downstream.
        Additional sentences ensure we meet length needs.
        Lorem ipsum dolor sit amet, consectetur adipiscing
        elit. Sed do eiusmod tempor incididunt ut labore et
        dolore magna aliqua. Ut enim ad minim veniam, quis
        nostrud exercitation ullamco laboris nisi ut aliquip
        ex ea commodo consequat. Duis aute irure dolor in
        reprehenderit in voluptate velit esse cillum dolore
        eu fugiat nulla pariatur. Excepteur sint occaecat
        cupidatat non proident, sunt in culpa qui officia
        deserunt mollit anim id est laborum. More text is
        needed to cross the two-thousand character boundary
        so the merger logic can be exercised correctly.
        Padding padding padding padding padding padding.
        Padding padding padding padding padding padding.
        Padding padding padding padding padding padding.
        Padding padding padding padding padding padding.
        Padding padding padding padding padding padding.
        Padding padding padding padding padding padding.

        ### Sub-details

        A tertiary section with brief notes. Short.
        """),
        encoding="utf-8",
    )
    return md


@pytest.fixture()
def mock_llm() -> MagicMock:
    """Return a mock LLM that echoes a canned response."""
    llm = MagicMock()
    response = MagicMock()
    response.content = "Mocked LLM response."
    response.tool_calls = []
    llm.invoke.return_value = response

    configured = MagicMock()
    configured.invoke.return_value = response
    configured.with_structured_output.return_value = configured
    llm.with_config.return_value = configured
    return llm


@pytest.fixture()
def sample_documents() -> list[Document]:
    """A handful of LangChain Documents for testing."""
    return [
        Document(
            page_content=f"Content for segment {i}.",
            metadata={
                "source": "test.pdf",
                "parent_id": f"test_segment_{i}",
            },
        )
        for i in range(5)
    ]
