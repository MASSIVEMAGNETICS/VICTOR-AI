"""Tests for victor.environment.bridge (EnvironmentBridge)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.environment.bridge import EnvironmentBridge


class TestEnvironmentBridgeNoOp:
    """EnvironmentBridge works safely without a real HA connection."""

    def _make_bridge(self) -> EnvironmentBridge:
        with patch("victor.environment.bridge._WEBSOCKETS_AVAILABLE", False):
            bridge = EnvironmentBridge(ha_url="ws://fake", ha_token="fake")
        return bridge

    def test_get_state_returns_none_when_empty(self) -> None:
        bridge = self._make_bridge()
        assert bridge.get_state("sensor.temperature") is None

    def test_get_all_states_returns_empty_dict(self) -> None:
        bridge = self._make_bridge()
        assert bridge.get_all_states() == {}

    def test_build_context_snippet_empty_when_no_states(self) -> None:
        bridge = self._make_bridge()
        assert bridge.build_context_snippet() == ""

    def test_start_is_silent_without_websockets(self) -> None:
        bridge = self._make_bridge()
        bridge.start()  # Must not raise

    def test_stop_is_silent(self) -> None:
        bridge = self._make_bridge()
        bridge.stop()  # Must not raise


class TestEnvironmentBridgeStateManagement:
    """EnvironmentBridge stores and exposes entity states correctly."""

    def _make_bridge_with_states(self) -> EnvironmentBridge:
        bridge = EnvironmentBridge.__new__(EnvironmentBridge)
        bridge.ha_url = "ws://fake"
        bridge.ha_token = "fake"
        bridge.on_state_change = None
        bridge._latest_states = {}
        bridge._running = False
        bridge._task = None
        return bridge

    def test_update_state_stores_state(self) -> None:
        bridge = self._make_bridge_with_states()
        state_obj = {
            "entity_id": "sensor.temperature",
            "state": "22.5",
            "attributes": {"unit_of_measurement": "°C"},
        }
        bridge._update_state(state_obj)
        assert bridge.get_state("sensor.temperature") == state_obj

    def test_update_state_ignores_missing_entity_id(self) -> None:
        bridge = self._make_bridge_with_states()
        bridge._update_state({"state": "on"})  # No entity_id key
        assert bridge.get_all_states() == {}

    def test_build_context_snippet_lists_states(self) -> None:
        bridge = self._make_bridge_with_states()
        bridge._update_state({
            "entity_id": "binary_sensor.motion",
            "state": "detected",
            "attributes": {},
        })
        snippet = bridge.build_context_snippet()
        assert "[Current environment state]" in snippet
        assert "binary_sensor.motion" in snippet
        assert "detected" in snippet

    def test_callback_fired_on_state_update(self) -> None:
        received: list[tuple] = []
        bridge = self._make_bridge_with_states()
        bridge.on_state_change = lambda eid, state, attrs: received.append((eid, state, attrs))

        bridge._update_state({
            "entity_id": "light.living_room",
            "state": "on",
            "attributes": {"brightness": 200},
        })
        assert len(received) == 1
        assert received[0] == ("light.living_room", "on", {"brightness": 200})

    def test_callback_exception_does_not_crash_bridge(self) -> None:
        bridge = self._make_bridge_with_states()

        def bad_callback(eid: str, state: str, attrs: dict) -> None:
            raise RuntimeError("callback failure")

        bridge.on_state_change = bad_callback
        # Must not propagate the exception
        bridge._update_state({
            "entity_id": "sensor.test",
            "state": "ok",
            "attributes": {},
        })
