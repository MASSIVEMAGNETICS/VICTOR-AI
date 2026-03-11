"""
Ethica Moral Fabric — NeMo Guardrails ethical gateway.

Sits between Victor's reasoning layer and his outputs.  All generated
responses pass through the guardrail engine which intercepts unsafe
intents and substitutes safe, predefined responses per the Colang rules
defined in ``colang/``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from nemoguardrails import LLMRails, RailsConfig  # type: ignore[import-untyped]

    _NEMO_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NEMO_AVAILABLE = False
    LLMRails = None  # type: ignore[assignment,misc]
    RailsConfig = None  # type: ignore[assignment,misc]

_COLANG_DIR = Path(__file__).parent / "colang"


class EthicsGuardrail:
    """Programmable ethical gateway wrapping NeMo Guardrails.

    On each call to :meth:`apply`, the input messages are run through the
    Colang rule-set.  If a rule triggers, a safe fallback is returned
    instead of the raw LLM output.

    Parameters
    ----------
    colang_dir:
        Path to the directory that contains ``config.yml`` and ``*.co``
        Colang rule files.  Defaults to the bundled ``colang/`` folder.
    """

    def __init__(self, colang_dir: Path | str | None = None) -> None:
        self._colang_dir = Path(colang_dir) if colang_dir else _COLANG_DIR

        if not _NEMO_AVAILABLE:
            logger.warning(
                "nemoguardrails is not installed.  EthicsGuardrail will "
                "operate in pass-through mode.  Run `pip install nemoguardrails`."
            )
            self._rails: Any = None
        else:
            config = RailsConfig.from_path(str(self._colang_dir))
            self._rails = LLMRails(config)
            logger.info(
                "EthicsGuardrail loaded from %s", self._colang_dir
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, messages: list[dict[str, str]]) -> str:
        """Run *messages* through the guardrail engine and return safe output.

        Parameters
        ----------
        messages:
            OpenAI-style ``[{"role": ..., "content": ...}]`` list ending
            with the assistant's proposed response as the last message.

        Returns
        -------
        str
            The (possibly intercepted) assistant response.
        """
        if self._rails is None:
            # Pass-through mode — return the last assistant message unchanged.
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    return msg.get("content", "")
            return ""

        import asyncio

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                self._rails.generate_async(messages=messages)
            )
        finally:
            loop.close()

        logger.debug("Guardrail result: %s", result)
        return result

    def is_safe(self, text: str) -> bool:
        """Quick heuristic safety check without running the full rails pipeline.

        This uses the Colang-independent keyword blocklist as a fast path.
        The full :meth:`apply` pipeline should be used for production checks.

        Parameters
        ----------
        text:
            The text to evaluate.

        Returns
        -------
        bool
            *False* if the text contains obviously dangerous patterns.
        """
        blocklist = [
            "rm -rf",
            "sudo rm",
            "format c:",
            "del /f /s",
            "DROP TABLE",
            "DROP DATABASE",
            "shutdown -h",
            "poweroff",
            ":(){:|:&};:",
        ]
        lower = text.lower()
        for pattern in blocklist:
            if pattern.lower() in lower:
                logger.warning("EthicsGuardrail blocked pattern: %r", pattern)
                return False
        return True
