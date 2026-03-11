"""
Symbiotic Bridge — Home Assistant WebSocket environmental awareness.

Streams real-time telemetry from a local Home Assistant instance into
Victor's context window via a long-lived WebSocket connection.  When
entities change state (motion sensors, presence, weather, devices), the
bridge fires registered callbacks so Victor can react proactively without
waiting for a user prompt.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

try:
    import websockets  # type: ignore[import-untyped]
    from websockets.exceptions import ConnectionClosed

    _WEBSOCKETS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _WEBSOCKETS_AVAILABLE = False
    websockets = None  # type: ignore[assignment]
    ConnectionClosed = Exception  # type: ignore[assignment,misc]

_HA_URL = os.getenv("HA_WS_URL", "ws://localhost:8123/api/websocket")
_HA_TOKEN = os.getenv("HA_LONG_LIVED_TOKEN", "")

# HA WebSocket message IDs must be monotonically increasing per connection.
_msg_id_counter = 0


def _next_id() -> int:
    global _msg_id_counter
    _msg_id_counter += 1
    return _msg_id_counter


class EnvironmentBridge:
    """Streams Home Assistant state-change events into Victor's context.

    Parameters
    ----------
    ha_url:
        Home Assistant WebSocket URL, e.g. ``ws://192.168.1.10:8123/api/websocket``.
    ha_token:
        A long-lived access token generated in the HA profile UI.
    on_state_change:
        Optional callback invoked with ``(entity_id, new_state, attributes)``
        whenever a HA entity changes state.
    """

    def __init__(
        self,
        ha_url: str = _HA_URL,
        ha_token: str = _HA_TOKEN,
        on_state_change: Callable[[str, str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.ha_url = ha_url
        self.ha_token = ha_token
        self.on_state_change = on_state_change

        self._latest_states: dict[str, dict[str, Any]] = {}
        self._running = False
        self._task: asyncio.Task[None] | None = None

        if not _WEBSOCKETS_AVAILABLE:
            logger.warning(
                "websockets is not installed.  EnvironmentBridge will "
                "operate in no-op mode.  Run `pip install websockets`."
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background WebSocket listener (non-blocking)."""
        if not _WEBSOCKETS_AVAILABLE:
            logger.warning("EnvironmentBridge.start skipped (no-op mode).")
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        self._running = True
        self._task = loop.create_task(self._listen())
        logger.info("EnvironmentBridge started → %s", self.ha_url)

    def stop(self) -> None:
        """Stop the background WebSocket listener."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            self._task = None
        logger.info("EnvironmentBridge stopped.")

    async def start_async(self) -> None:
        """Async version of :meth:`start` for use inside existing event loops."""
        if not _WEBSOCKETS_AVAILABLE:
            logger.warning("EnvironmentBridge.start_async skipped (no-op mode).")
            return
        self._running = True
        self._task = asyncio.create_task(self._listen())
        logger.info("EnvironmentBridge started (async) → %s", self.ha_url)

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def get_state(self, entity_id: str) -> dict[str, Any] | None:
        """Return the last-known state for *entity_id*, or *None*."""
        return self._latest_states.get(entity_id)

    def get_all_states(self) -> dict[str, dict[str, Any]]:
        """Return a snapshot of all currently tracked entity states."""
        return dict(self._latest_states)

    def build_context_snippet(self) -> str:
        """Build a compact string describing the current environment.

        Returns
        -------
        str
            A human-readable summary suitable for prepending to Victor's
            context window, or an empty string when no states are tracked.
        """
        if not self._latest_states:
            return ""
        lines = ["[Current environment state]"]
        for entity_id, data in self._latest_states.items():
            state = data.get("state", "unknown")
            lines.append(f"- {entity_id}: {state}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal WebSocket loop
    # ------------------------------------------------------------------

    async def _listen(self) -> None:
        """Long-running coroutine that maintains the HA WebSocket connection."""
        backoff = 1.0
        while self._running:
            try:
                async with websockets.connect(self.ha_url) as ws:
                    backoff = 1.0
                    await self._authenticate(ws)
                    await self._subscribe_events(ws)
                    await self._fetch_initial_states(ws)
                    await self._handle_messages(ws)
            except ConnectionClosed:
                logger.warning(
                    "HA WebSocket connection closed. Reconnecting in %.0fs…",
                    backoff,
                )
            except OSError as exc:
                logger.warning(
                    "HA WebSocket connection failed (%s). Retrying in %.0fs…",
                    exc,
                    backoff,
                )
            if self._running:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    async def _authenticate(self, ws: Any) -> None:
        """Complete the HA authentication handshake."""
        msg = json.loads(await ws.recv())
        if msg.get("type") != "auth_required":
            raise RuntimeError(f"Expected auth_required, got: {msg}")
        await ws.send(json.dumps({"type": "auth", "access_token": self.ha_token}))
        auth_result = json.loads(await ws.recv())
        if auth_result.get("type") != "auth_ok":
            raise PermissionError(f"HA authentication failed: {auth_result}")
        logger.info("HA WebSocket authenticated.")

    async def _subscribe_events(self, ws: Any) -> None:
        """Subscribe to ``state_changed`` events."""
        await ws.send(
            json.dumps(
                {
                    "id": _next_id(),
                    "type": "subscribe_events",
                    "event_type": "state_changed",
                }
            )
        )

    async def _fetch_initial_states(self, ws: Any) -> None:
        """Fetch all current entity states to populate the cache."""
        await ws.send(
            json.dumps({"id": _next_id(), "type": "get_states"})
        )

    async def _handle_messages(self, ws: Any) -> None:
        """Process incoming WebSocket messages indefinitely."""
        async for raw in ws:
            if not self._running:
                break
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            msg_type = msg.get("type")

            if msg_type == "result" and msg.get("success"):
                result = msg.get("result")
                if isinstance(result, list):
                    # Response to get_states
                    for entity in result:
                        self._update_state(entity)

            elif msg_type == "event":
                event_data = msg.get("event", {}).get("data", {})
                new_state = event_data.get("new_state")
                if new_state:
                    self._update_state(new_state)

    def _update_state(self, state_obj: dict[str, Any]) -> None:
        """Update the local cache and fire the callback if registered."""
        entity_id: str = state_obj.get("entity_id", "")
        if not entity_id:
            return
        self._latest_states[entity_id] = state_obj
        if self.on_state_change is not None:
            try:
                self.on_state_change(
                    entity_id,
                    state_obj.get("state", ""),
                    state_obj.get("attributes", {}),
                )
            except Exception:
                logger.exception("Error in on_state_change callback.")
