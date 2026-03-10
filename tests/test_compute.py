"""Tests for victor.compute.router (ComputeRouter)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from victor.compute.router import ComputeMode, ComputeRouter, _MODEL_PROFILES


class TestComputeRouterModeDetection:
    """ComputeRouter correctly auto-detects the right compute tier."""

    def _make_router(self) -> ComputeRouter:
        return ComputeRouter()

    def test_heavy_keywords_route_to_heavy(self) -> None:
        router = self._make_router()
        mode = router.detect_mode("Please help me with advanced system design for a distributed architecture")
        assert mode == ComputeMode.HEAVY

    def test_code_keywords_route_to_code(self) -> None:
        router = self._make_router()
        mode = router.detect_mode("Write a Python function to parse CSV files")
        assert mode == ComputeMode.CODE

    def test_plain_request_routes_to_balanced(self) -> None:
        router = self._make_router()
        mode = router.detect_mode("What is the weather like today?")
        assert mode == ComputeMode.BALANCED

    def test_custom_default_mode_used_for_unmatched(self) -> None:
        router = ComputeRouter(default_mode=ComputeMode.FAST)
        mode = router.detect_mode("Sort this list: 3, 1, 2")
        assert mode == ComputeMode.FAST

    def test_heavy_takes_precedence_over_code(self) -> None:
        router = self._make_router()
        # "complex" is a heavy keyword, "python" is a code keyword
        mode = router.detect_mode("Write complex python code")
        assert mode == ComputeMode.HEAVY


class TestComputeRouterNoOp:
    """ComputeRouter returns no-op objects when litellm is unavailable."""

    def _make_noop_router(self) -> ComputeRouter:
        with patch("victor.compute.router._LITELLM_AVAILABLE", False):
            router = ComputeRouter()
        router  # _LITELLM_AVAILABLE checked at call time too — patch for calls
        return router

    def test_complete_returns_noop_response(self) -> None:
        with patch("victor.compute.router._LITELLM_AVAILABLE", False):
            router = ComputeRouter()
            result = router.complete([{"role": "user", "content": "hi"}])
        # No-op response has no meaningful choices
        assert result.choices == []

    def test_stream_complete_yields_nothing(self) -> None:
        with patch("victor.compute.router._LITELLM_AVAILABLE", False):
            router = ComputeRouter()
            chunks = list(router.stream_complete([{"role": "user", "content": "hi"}]))
        assert chunks == []


class TestComputeRouterWithMock:
    """ComputeRouter calls litellm with the correct model and parameters."""

    def test_complete_calls_litellm_with_correct_model(self) -> None:
        mock_litellm = MagicMock()
        mock_litellm.completion.return_value = MagicMock(choices=[])
        with patch("victor.compute.router._LITELLM_AVAILABLE", True), \
             patch("victor.compute.router.litellm", mock_litellm):
            router = ComputeRouter()
            msgs = [{"role": "user", "content": "Hello"}]
            router.complete(msgs, mode=ComputeMode.FAST)

        expected_model = f"ollama/{_MODEL_PROFILES['fast']}"
        mock_litellm.completion.assert_called_once()
        call_kwargs = mock_litellm.completion.call_args
        assert call_kwargs.kwargs["model"] == expected_model or \
               call_kwargs.args[0] == expected_model

    def test_complete_auto_detects_code_mode(self) -> None:
        mock_litellm = MagicMock()
        mock_litellm.completion.return_value = MagicMock(choices=[])
        with patch("victor.compute.router._LITELLM_AVAILABLE", True), \
             patch("victor.compute.router.litellm", mock_litellm):
            router = ComputeRouter()
            msgs = [{"role": "user", "content": "write a python script"}]
            router.complete(msgs)

        expected_model = f"ollama/{_MODEL_PROFILES['code']}"
        call_kwargs = mock_litellm.completion.call_args
        used_model = call_kwargs.kwargs.get("model") or call_kwargs.args[0]
        assert used_model == expected_model
