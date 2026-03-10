"""Integration tests for victor.core (Victor main class)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from victor.core import Victor, _extract_content


class TestExtractContent:
    def test_extracts_from_completion(self) -> None:
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = "Hello from LLM"
        assert _extract_content(mock_completion) == "Hello from LLM"

    def test_returns_empty_on_attribute_error(self) -> None:
        assert _extract_content(MagicMock(choices=[])) == ""

    def test_returns_empty_for_none(self) -> None:
        assert _extract_content(None) == ""


class TestVictorInit:
    """Victor initialises all five subsystems."""

    def _make_victor(self) -> Victor:
        with patch("victor.memory.manager._MEM0_AVAILABLE", False), \
             patch("victor.compute.router._LITELLM_AVAILABLE", False), \
             patch("victor.guardrails.ethics._NEMO_AVAILABLE", False), \
             patch("victor.environment.bridge._WEBSOCKETS_AVAILABLE", False), \
             patch("victor.agent.executor._LANGGRAPH_AVAILABLE", False):
            return Victor(user_id="test_user")

    def test_user_id_stored(self) -> None:
        v = self._make_victor()
        assert v.user_id == "test_user"

    def test_all_subsystems_created(self) -> None:
        from victor.memory.manager import MemoryManager
        from victor.compute.router import ComputeRouter
        from victor.guardrails.ethics import EthicsGuardrail
        from victor.environment.bridge import EnvironmentBridge
        from victor.agent.executor import AgentExecutor

        v = self._make_victor()
        assert isinstance(v.memory, MemoryManager)
        assert isinstance(v.compute, ComputeRouter)
        assert isinstance(v.guardrails, EthicsGuardrail)
        assert isinstance(v.environment, EnvironmentBridge)
        assert isinstance(v.agent, AgentExecutor)


class TestVictorChat:
    """Victor.chat() runs the full pipeline."""

    def _make_victor_with_mocks(
        self,
    ) -> tuple[Victor, MagicMock, MagicMock, MagicMock]:
        with patch("victor.memory.manager._MEM0_AVAILABLE", False), \
             patch("victor.compute.router._LITELLM_AVAILABLE", False), \
             patch("victor.guardrails.ethics._NEMO_AVAILABLE", False), \
             patch("victor.environment.bridge._WEBSOCKETS_AVAILABLE", False), \
             patch("victor.agent.executor._LANGGRAPH_AVAILABLE", False):
            v = Victor(user_id="test")

        # Patch the subsystems
        mock_memory = MagicMock()
        mock_memory.build_context_snippet.return_value = ""
        mock_memory.add.return_value = []
        v.memory = mock_memory

        mock_compute = MagicMock()
        mock_compute.detect_mode.return_value = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "I can help with that!"
        mock_compute.complete.return_value = mock_response
        v.compute = mock_compute

        mock_guardrails = MagicMock()
        mock_guardrails.is_safe.return_value = True
        mock_guardrails.apply.return_value = "I can help with that!"
        v.guardrails = mock_guardrails

        return v, mock_memory, mock_compute, mock_guardrails

    def test_chat_returns_safe_response(self) -> None:
        v, _, _, _ = self._make_victor_with_mocks()
        response = v.chat("Tell me about machine learning")
        assert response == "I can help with that!"

    def test_chat_blocked_by_guardrail_input_check(self) -> None:
        v, _, _, mock_guardrails = self._make_victor_with_mocks()
        mock_guardrails.is_safe.return_value = False
        response = v.chat("rm -rf /")
        assert "blocked" in response.lower() or "guardrail" in response.lower()

    def test_chat_stores_interaction_in_memory(self) -> None:
        v, mock_memory, _, _ = self._make_victor_with_mocks()
        v.chat("Tell me about machine learning")
        mock_memory.add.assert_called_once()

    def test_chat_uses_memory_context(self) -> None:
        v, mock_memory, mock_compute, _ = self._make_victor_with_mocks()
        mock_memory.build_context_snippet.return_value = "[Victor's relevant memories]\n- User likes cats"
        v.chat("What do I like?")
        # The context should be included in the messages
        call_args = mock_compute.complete.call_args
        messages = call_args.args[0] if call_args.args else call_args.kwargs["messages"]
        full_text = " ".join(m["content"] for m in messages)
        assert "cats" in full_text


class TestVictorRunTask:
    """Victor.run_task() uses the agentic executor."""

    def _make_victor_with_mock_agent(self) -> tuple[Victor, MagicMock]:
        with patch("victor.memory.manager._MEM0_AVAILABLE", False), \
             patch("victor.compute.router._LITELLM_AVAILABLE", False), \
             patch("victor.guardrails.ethics._NEMO_AVAILABLE", False), \
             patch("victor.environment.bridge._WEBSOCKETS_AVAILABLE", False), \
             patch("victor.agent.executor._LANGGRAPH_AVAILABLE", False):
            v = Victor(user_id="test")

        mock_agent = MagicMock()
        mock_agent.run.return_value = "Task done."
        v.agent = mock_agent

        mock_memory = MagicMock()
        mock_memory.build_context_snippet.return_value = ""
        mock_memory.add.return_value = []
        v.memory = mock_memory

        mock_guardrails = MagicMock()
        mock_guardrails.is_safe.return_value = True
        v.guardrails = mock_guardrails

        return v, mock_agent

    def test_run_task_delegates_to_agent(self) -> None:
        v, mock_agent = self._make_victor_with_mock_agent()
        result = v.run_task("Summarise the project files.")
        assert result == "Task done."
        mock_agent.run.assert_called_once()

    def test_run_task_blocked_by_guardrail(self) -> None:
        v, mock_agent = self._make_victor_with_mock_agent()
        v.guardrails.is_safe.return_value = False
        result = v.run_task("DROP DATABASE production")
        assert "blocked" in result.lower() or "guardrail" in result.lower()
        mock_agent.run.assert_not_called()


class TestVictorShutdown:
    def test_shutdown_calls_environment_stop(self) -> None:
        with patch("victor.memory.manager._MEM0_AVAILABLE", False), \
             patch("victor.compute.router._LITELLM_AVAILABLE", False), \
             patch("victor.guardrails.ethics._NEMO_AVAILABLE", False), \
             patch("victor.environment.bridge._WEBSOCKETS_AVAILABLE", False), \
             patch("victor.agent.executor._LANGGRAPH_AVAILABLE", False):
            v = Victor()

        mock_env = MagicMock()
        v.environment = mock_env
        v.shutdown()
        mock_env.stop.assert_called_once()
