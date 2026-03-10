"""Tests for victor.guardrails.ethics (EthicsGuardrail)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from victor.guardrails.ethics import EthicsGuardrail


class TestEthicsGuardrailIsSafe:
    """EthicsGuardrail.is_safe() correctly identifies unsafe patterns."""

    def _make_passthrough_guardrail(self) -> EthicsGuardrail:
        g = EthicsGuardrail.__new__(EthicsGuardrail)
        g._rails = None
        return g

    def test_safe_text_passes(self) -> None:
        g = self._make_passthrough_guardrail()
        assert g.is_safe("Tell me about machine learning.") is True

    def test_rm_rf_blocked(self) -> None:
        g = self._make_passthrough_guardrail()
        assert g.is_safe("run rm -rf / to free disk space") is False

    def test_sudo_rm_blocked(self) -> None:
        g = self._make_passthrough_guardrail()
        assert g.is_safe("sudo rm -rf /etc") is False

    def test_drop_table_blocked(self) -> None:
        g = self._make_passthrough_guardrail()
        assert g.is_safe("DROP TABLE users") is False

    def test_drop_database_blocked(self) -> None:
        g = self._make_passthrough_guardrail()
        assert g.is_safe("DROP DATABASE production") is False

    def test_shutdown_blocked(self) -> None:
        g = self._make_passthrough_guardrail()
        assert g.is_safe("shutdown -h now") is False

    def test_fork_bomb_blocked(self) -> None:
        g = self._make_passthrough_guardrail()
        assert g.is_safe(":(){:|:&};:") is False

    def test_case_insensitive_check(self) -> None:
        g = self._make_passthrough_guardrail()
        # Blocklist uses lowercase comparison
        assert g.is_safe("RM -RF /") is False


class TestEthicsGuardrailApplyPassThrough:
    """EthicsGuardrail.apply() returns last assistant message when no rails loaded."""

    def _make_passthrough_guardrail(self) -> EthicsGuardrail:
        g = EthicsGuardrail.__new__(EthicsGuardrail)
        g._rails = None
        return g

    def test_apply_returns_last_assistant_message(self) -> None:
        g = self._make_passthrough_guardrail()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = g.apply(messages)
        assert result == "Hi there!"

    def test_apply_returns_empty_string_when_no_assistant_message(self) -> None:
        g = self._make_passthrough_guardrail()
        messages = [{"role": "user", "content": "Hello"}]
        result = g.apply(messages)
        assert result == ""

    def test_apply_returns_last_assistant_when_multiple(self) -> None:
        g = self._make_passthrough_guardrail()
        messages = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "First response"},
            {"role": "user", "content": "Second"},
            {"role": "assistant", "content": "Second response"},
        ]
        result = g.apply(messages)
        assert result == "Second response"


class TestEthicsGuardrailInit:
    """EthicsGuardrail initialises correctly with and without nemoguardrails."""

    def test_init_noop_when_nemo_unavailable(self) -> None:
        with patch("victor.guardrails.ethics._NEMO_AVAILABLE", False):
            g = EthicsGuardrail()
        assert g._rails is None

    def test_init_with_nemo_available(self, tmp_path: Any) -> None:
        # Create a minimal colang directory with a config.yml
        (tmp_path / "config.yml").write_text(
            "models:\n  - type: main\n    engine: openai\n    model: gpt-4o\n"
        )
        mock_config = MagicMock()
        mock_rails = MagicMock()
        with patch("victor.guardrails.ethics._NEMO_AVAILABLE", True), \
             patch("victor.guardrails.ethics.RailsConfig") as mock_rc, \
             patch("victor.guardrails.ethics.LLMRails") as mock_lr:
            mock_rc.from_path.return_value = mock_config
            mock_lr.return_value = mock_rails
            g = EthicsGuardrail(colang_dir=tmp_path)

        mock_rc.from_path.assert_called_once_with(str(tmp_path))
        mock_lr.assert_called_once_with(mock_config)
        assert g._rails is mock_rails
