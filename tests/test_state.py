"""Tests for graph state reducers."""

from __future__ import annotations

from docmind.agent.state import _accumulate_or_clear, _union


class TestAccumulateOrClear:
    """Verify the list reducer with sentinel-based reset."""

    def test_appends_normally(self) -> None:
        current = [{"a": 1}]
        incoming = [{"b": 2}]
        result = _accumulate_or_clear(current, incoming)
        assert result == [{"a": 1}, {"b": 2}]

    def test_clears_on_sentinel(self) -> None:
        current = [{"a": 1}, {"a": 2}]
        incoming = [{"__clear__": True}]
        result = _accumulate_or_clear(current, incoming)
        assert result == []

    def test_empty_incoming_no_change(self) -> None:
        current = [{"x": 1}]
        result = _accumulate_or_clear(current, [])
        assert result == [{"x": 1}]

    def test_both_empty(self) -> None:
        assert _accumulate_or_clear([], []) == []


class TestSetUnion:
    """Verify the set union reducer."""

    def test_basic_union(self) -> None:
        assert _union({"a", "b"}, {"b", "c"}) == {
            "a",
            "b",
            "c",
        }

    def test_empty_sets(self) -> None:
        assert _union(set(), set()) == set()

    def test_disjoint_sets(self) -> None:
        assert _union({"x"}, {"y"}) == {"x", "y"}
