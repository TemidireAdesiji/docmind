"""Tests for application settings."""

from __future__ import annotations

from docmind.settings import Settings


class TestSettingsDefaults:
    """Verify default configuration values."""

    def test_default_llm_model(self) -> None:
        cfg = Settings(_env_file=None)
        assert "qwen3" in cfg.llm_model

    def test_default_temperature_is_zero(self) -> None:
        cfg = Settings(_env_file=None)
        assert cfg.llm_temperature == 0

    def test_temperature_clamped_to_max(self) -> None:
        cfg = Settings(_env_file=None, llm_temperature=1.5)
        assert cfg.llm_temperature == 1.5

    def test_child_segment_defaults(self) -> None:
        cfg = Settings(_env_file=None)
        assert cfg.child_segment_size == 500
        assert cfg.child_segment_overlap == 100

    def test_heading_levels_default(self) -> None:
        cfg = Settings(_env_file=None)
        assert len(cfg.heading_levels) == 3

    def test_tracing_disabled_by_default(self) -> None:
        cfg = Settings(_env_file=None)
        assert cfg.tracing_enabled is False

    def test_paths_are_accessible(self) -> None:
        cfg = Settings(_env_file=None)
        paths = cfg.paths
        assert paths.converted_docs.name == "converted_docs"
        assert paths.stored_chunks.name == "stored_chunks"
        assert paths.vector_data.name == "vector_data"
