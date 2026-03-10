"""
Persistent Long-Term Memory Manager — Victor's "Ego Core".

Uses Mem0 backed by a PostgreSQL/pgvector store to extract, persist, and
retrieve facts, entities, and relationships across sessions.  Before Victor
answers any prompt the MemoryManager is queried to inject relevant memories
into the context, giving him a continuous identity that survives context-window
resets.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# mem0ai is the primary memory backend.  We import lazily so the rest of the
# codebase can still be imported even when the optional dependency is absent
# (useful for testing with mocks).
try:
    from mem0 import Memory  # type: ignore[import-untyped]

    _MEM0_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MEM0_AVAILABLE = False
    Memory = None  # type: ignore[assignment,misc]


_DEFAULT_CONFIG: dict[str, Any] = {
    "vector_store": {
        "provider": "pgvector",
        "config": {
            "host": os.getenv("PGVECTOR_HOST", "localhost"),
            "port": int(os.getenv("PGVECTOR_PORT", "5432")),
            "dbname": os.getenv("PGVECTOR_DB", "victor"),
            "user": os.getenv("PGVECTOR_USER", "victor"),
            "password": os.getenv("PGVECTOR_PASSWORD", "victor"),
        },
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": os.getenv("MEMORY_LLM_MODEL", "llama3"),
            "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": os.getenv("MEMORY_EMBED_MODEL", "nomic-embed-text"),
            "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        },
    },
}


class MemoryManager:
    """Manages Victor's long-term memory via Mem0 + PostgreSQL/pgvector.

    Parameters
    ----------
    user_id:
        A stable identifier for the user (or agent session).  All memories
        are scoped to this ID so different users/sessions stay isolated.
    config:
        Optional Mem0 configuration dict.  Defaults to ``_DEFAULT_CONFIG``
        which reads connection parameters from environment variables.
    """

    def __init__(
        self,
        user_id: str = "victor_default",
        config: dict[str, Any] | None = None,
    ) -> None:
        self.user_id = user_id
        self._config = config or _DEFAULT_CONFIG

        if not _MEM0_AVAILABLE:
            logger.warning(
                "mem0ai is not installed.  MemoryManager will operate in "
                "no-op mode.  Run `pip install mem0ai` to enable persistent memory."
            )
            self._memory: Any = None
        else:
            self._memory = Memory.from_config(self._config)
            logger.info("MemoryManager initialised for user_id=%r", self.user_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, messages: list[dict[str, str]]) -> list[dict[str, Any]]:
        """Extract and persist memories from a list of chat messages.

        Parameters
        ----------
        messages:
            A list of ``{"role": ..., "content": ...}`` dicts in the same
            format used by OpenAI-compatible chat APIs.

        Returns
        -------
        list[dict]
            The memories that were newly created or updated.
        """
        if self._memory is None:
            logger.debug("Memory add skipped (no-op mode).")
            return []
        result = self._memory.add(messages, user_id=self.user_id)
        logger.debug("Memory add result: %s", result)
        return result

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Retrieve the *limit* most relevant memories for *query*.

        Parameters
        ----------
        query:
            Natural-language query string.
        limit:
            Maximum number of memories to return.

        Returns
        -------
        list[dict]
            Each entry is a memory object with at least a ``"memory"`` key
            containing the stored text.
        """
        if self._memory is None:
            logger.debug("Memory search skipped (no-op mode).")
            return []
        results = self._memory.search(query, user_id=self.user_id, limit=limit)
        logger.debug("Memory search returned %d results.", len(results))
        return results

    def get_all(self) -> list[dict[str, Any]]:
        """Return all stored memories for the current user."""
        if self._memory is None:
            return []
        return self._memory.get_all(user_id=self.user_id)

    def delete(self, memory_id: str) -> None:
        """Delete a specific memory by its ID."""
        if self._memory is None:
            return
        self._memory.delete(memory_id)
        logger.info("Deleted memory id=%r", memory_id)

    def clear(self) -> None:
        """Wipe **all** memories for the current user (use with care)."""
        if self._memory is None:
            return
        self._memory.delete_all(user_id=self.user_id)
        logger.info("Cleared all memories for user_id=%r", self.user_id)

    def build_context_snippet(self, query: str) -> str:
        """Return a formatted string of relevant memories for injection into a prompt.

        Parameters
        ----------
        query:
            The user's current prompt / query.

        Returns
        -------
        str
            A newline-separated string ready to be prepended to Victor's
            context window, or an empty string when no memories exist.
        """
        memories = self.search(query)
        if not memories:
            return ""
        lines = ["[Victor's relevant memories]"]
        for item in memories:
            text = item.get("memory", "")
            if text:
                lines.append(f"- {text}")
        return "\n".join(lines)
