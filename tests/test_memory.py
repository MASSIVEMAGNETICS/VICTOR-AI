"""Tests for victor.memory.manager (MemoryManager)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from victor.memory.manager import MemoryManager, _DEFAULT_CONFIG


class TestMemoryManagerNoOp:
    """MemoryManager behaves safely when mem0ai is not installed."""

    def _make_manager(self) -> MemoryManager:
        with patch("victor.memory.manager._MEM0_AVAILABLE", False):
            mgr = MemoryManager(user_id="test_user")
        mgr._memory = None
        return mgr

    def test_add_returns_empty_list(self) -> None:
        mgr = self._make_manager()
        result = mgr.add([{"role": "user", "content": "hello"}])
        assert result == []

    def test_search_returns_empty_list(self) -> None:
        mgr = self._make_manager()
        result = mgr.search("anything")
        assert result == []

    def test_get_all_returns_empty_list(self) -> None:
        mgr = self._make_manager()
        assert mgr.get_all() == []

    def test_delete_is_silent(self) -> None:
        mgr = self._make_manager()
        mgr.delete("some-id")  # must not raise

    def test_clear_is_silent(self) -> None:
        mgr = self._make_manager()
        mgr.clear()  # must not raise

    def test_build_context_snippet_empty(self) -> None:
        mgr = self._make_manager()
        assert mgr.build_context_snippet("query") == ""


class TestMemoryManagerWithMock:
    """MemoryManager delegates correctly to the Mem0 backend."""

    def _make_manager_with_mock_mem0(self) -> tuple[MemoryManager, MagicMock]:
        mock_mem0 = MagicMock()
        mgr = MemoryManager.__new__(MemoryManager)
        mgr.user_id = "mock_user"
        mgr._config = _DEFAULT_CONFIG
        mgr._memory = mock_mem0
        return mgr, mock_mem0

    def test_add_delegates_to_backend(self) -> None:
        mgr, mock_mem0 = self._make_manager_with_mock_mem0()
        mock_mem0.add.return_value = [{"id": "1", "memory": "test fact"}]
        msgs = [{"role": "user", "content": "I like cats"}]
        result = mgr.add(msgs)
        mock_mem0.add.assert_called_once_with(msgs, user_id="mock_user")
        assert result == [{"id": "1", "memory": "test fact"}]

    def test_search_delegates_to_backend(self) -> None:
        mgr, mock_mem0 = self._make_manager_with_mock_mem0()
        mock_mem0.search.return_value = [{"memory": "I like cats"}]
        results = mgr.search("cats", limit=5)
        mock_mem0.search.assert_called_once_with("cats", user_id="mock_user", limit=5)
        assert len(results) == 1

    def test_get_all_delegates(self) -> None:
        mgr, mock_mem0 = self._make_manager_with_mock_mem0()
        mock_mem0.get_all.return_value = []
        assert mgr.get_all() == []
        mock_mem0.get_all.assert_called_once_with(user_id="mock_user")

    def test_delete_delegates(self) -> None:
        mgr, mock_mem0 = self._make_manager_with_mock_mem0()
        mgr.delete("abc-123")
        mock_mem0.delete.assert_called_once_with("abc-123")

    def test_clear_calls_delete_all(self) -> None:
        mgr, mock_mem0 = self._make_manager_with_mock_mem0()
        mgr.clear()
        mock_mem0.delete_all.assert_called_once_with(user_id="mock_user")

    def test_build_context_snippet_formats_memories(self) -> None:
        mgr, mock_mem0 = self._make_manager_with_mock_mem0()
        mock_mem0.search.return_value = [
            {"memory": "User likes cats"},
            {"memory": "User is working on Victor AI"},
        ]
        snippet = mgr.build_context_snippet("project status")
        assert "[Victor's relevant memories]" in snippet
        assert "User likes cats" in snippet
        assert "User is working on Victor AI" in snippet

    def test_build_context_snippet_empty_when_no_memories(self) -> None:
        mgr, mock_mem0 = self._make_manager_with_mock_mem0()
        mock_mem0.search.return_value = []
        assert mgr.build_context_snippet("query") == ""
