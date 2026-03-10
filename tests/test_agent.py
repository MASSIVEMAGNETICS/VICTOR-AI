"""Tests for victor.agent (AgentExecutor and tools)."""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from victor.agent.tools import (
    fetch_url,
    list_directory,
    parse_json,
    read_local_file,
    send_email,
    ALL_TOOLS,
)
from victor.agent.executor import AgentExecutor


# ---------------------------------------------------------------------------
# Tool tests
# ---------------------------------------------------------------------------


class TestReadLocalFile:
    def test_reads_existing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "hello.txt"
        f.write_text("hello world")
        result = read_local_file.invoke({"file_path": str(f)})
        assert result == "hello world"

    def test_error_on_missing_file(self, tmp_path: Path) -> None:
        result = read_local_file.invoke({"file_path": str(tmp_path / "nonexistent.txt")})
        assert "Error" in result

    def test_error_on_directory(self, tmp_path: Path) -> None:
        result = read_local_file.invoke({"file_path": str(tmp_path)})
        assert "Error" in result


class TestListDirectory:
    def test_lists_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        result = list_directory.invoke({"directory_path": str(tmp_path)})
        assert "a.txt" in result
        assert "b.txt" in result

    def test_empty_directory(self, tmp_path: Path) -> None:
        result = list_directory.invoke({"directory_path": str(tmp_path)})
        assert "empty" in result

    def test_error_on_missing_directory(self, tmp_path: Path) -> None:
        result = list_directory.invoke({"directory_path": str(tmp_path / "missing")})
        assert "Error" in result


class TestParseJson:
    def test_valid_json_is_pretty_printed(self) -> None:
        result = parse_json.invoke({"json_string": '{"key": "value", "num": 42}'})
        data = json.loads(result)
        assert data["key"] == "value"
        assert data["num"] == 42

    def test_invalid_json_returns_error(self) -> None:
        result = parse_json.invoke({"json_string": "not json {"})
        assert "Error" in result


class TestFetchUrl:
    def test_network_error_returns_error_string(self) -> None:
        result = fetch_url.invoke({"url": "http://127.0.0.1:19999/nonexistent"})
        assert "Error" in result


class TestSendEmail:
    def test_smtp_connection_failure_returns_error(self) -> None:
        result = send_email.invoke({
            "to_address": "test@example.com",
            "subject": "Test",
            "body": "Hello",
        })
        assert "Error" in result


class TestAllTools:
    def test_all_tools_list_not_empty(self) -> None:
        assert len(ALL_TOOLS) > 0

    def test_all_tools_have_invoke(self) -> None:
        for tool in ALL_TOOLS:
            assert hasattr(tool, "invoke"), f"{tool} has no invoke method"


# ---------------------------------------------------------------------------
# AgentExecutor tests
# ---------------------------------------------------------------------------


class TestAgentExecutorNoOp:
    """AgentExecutor returns no-op strings when langgraph is unavailable."""

    def test_run_returns_noop_string(self) -> None:
        with patch("victor.agent.executor._LANGGRAPH_AVAILABLE", False):
            executor = AgentExecutor()
        result = executor.run("test task")
        assert "no-op" in result or "test task" in result


class TestAgentExecutorWithMock:
    """AgentExecutor invokes the compiled graph when langgraph is available."""

    def test_run_delegates_to_graph(self) -> None:
        mock_graph = MagicMock()
        mock_ai_message = MagicMock()
        mock_ai_message.content = "Task completed successfully."

        # Simulate AIMessage type check
        from langchain_core.messages import AIMessage  # noqa: PLC0415
        real_ai_msg = AIMessage(content="Task completed successfully.")
        mock_graph.invoke.return_value = {"messages": [real_ai_msg]}

        executor = AgentExecutor.__new__(AgentExecutor)
        executor._graph = mock_graph
        executor._tools = []
        executor._model_name = "llama3"
        executor._ollama_base_url = "http://localhost:11434"
        executor._max_iterations = 10

        result = executor.run("Do something useful")
        assert result == "Task completed successfully."
        mock_graph.invoke.assert_called_once()
