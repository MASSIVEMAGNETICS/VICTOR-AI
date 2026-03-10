"""
Dynamic Compute Router — Victor's "Endocrine System".

Uses LiteLLM as a unified gateway in front of locally-hosted Ollama models.
Routes prompts to fast, small models for lightweight tasks and to larger
models for complex reasoning, simulating shifting "hormonal" compute states.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Any, Iterator

logger = logging.getLogger(__name__)

try:
    import litellm  # type: ignore[import-untyped]

    _LITELLM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LITELLM_AVAILABLE = False
    litellm = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Model profiles
# ---------------------------------------------------------------------------

_OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Each profile maps to an Ollama model name.
# Override via environment variables to swap models without code changes.
_MODEL_PROFILES: dict[str, str] = {
    "fast": os.getenv("COMPUTE_MODEL_FAST", "phi3"),
    "balanced": os.getenv("COMPUTE_MODEL_BALANCED", "llama3"),
    "heavy": os.getenv("COMPUTE_MODEL_HEAVY", "llama3:70b"),
    "code": os.getenv("COMPUTE_MODEL_CODE", "codellama"),
}


class ComputeMode(str, Enum):
    """Named compute modes that map to specific Ollama model profiles."""

    FAST = "fast"
    BALANCED = "balanced"
    HEAVY = "heavy"
    CODE = "code"


# Keywords that bump a request into a heavier compute tier automatically.
_HEAVY_KEYWORDS = frozenset(
    [
        "rust",
        "architecture",
        "design pattern",
        "refactor",
        "optimize",
        "complex",
        "advanced",
        "system design",
        "distributed",
    ]
)
_CODE_KEYWORDS = frozenset(
    [
        "code",
        "script",
        "function",
        "class",
        "debug",
        "implement",
        "python",
        "javascript",
        "typescript",
        "golang",
        "c++",
        "java",
    ]
)


class ComputeRouter:
    """Routes LLM completions to the most appropriate Ollama model via LiteLLM.

    Parameters
    ----------
    default_mode:
        The ``ComputeMode`` used when no task-specific routing applies.
    api_base:
        Ollama server base URL.  Defaults to the ``OLLAMA_BASE_URL``
        environment variable (``http://localhost:11434``).
    """

    def __init__(
        self,
        default_mode: ComputeMode = ComputeMode.BALANCED,
        api_base: str = _OLLAMA_BASE,
    ) -> None:
        self.default_mode = default_mode
        self.api_base = api_base

        if not _LITELLM_AVAILABLE:
            logger.warning(
                "litellm is not installed.  ComputeRouter will operate in "
                "no-op mode.  Run `pip install litellm` to enable routing."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_mode(self, prompt: str) -> ComputeMode:
        """Heuristically choose a ``ComputeMode`` based on prompt content.

        Parameters
        ----------
        prompt:
            The user's natural-language request.

        Returns
        -------
        ComputeMode
            The inferred compute tier.
        """
        lower = prompt.lower()
        if any(kw in lower for kw in _HEAVY_KEYWORDS):
            return ComputeMode.HEAVY
        if any(kw in lower for kw in _CODE_KEYWORDS):
            return ComputeMode.CODE
        return self.default_mode

    def complete(
        self,
        messages: list[dict[str, str]],
        mode: ComputeMode | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Send a chat-completion request to the appropriate Ollama model.

        Parameters
        ----------
        messages:
            OpenAI-style ``[{"role": ..., "content": ...}]`` message list.
        mode:
            Override the compute mode.  If *None*, the mode is auto-detected
            from the last user message.
        stream:
            When *True*, returns a streaming generator.
        **kwargs:
            Additional keyword arguments forwarded to ``litellm.completion``.

        Returns
        -------
        Any
            A LiteLLM ``ModelResponse`` (or streaming generator when
            ``stream=True``).
        """
        if not _LITELLM_AVAILABLE:
            logger.warning("ComputeRouter.complete called in no-op mode.")
            return _NoOpResponse()

        if mode is None:
            last_user = next(
                (m["content"] for m in reversed(messages) if m["role"] == "user"),
                "",
            )
            mode = self.detect_mode(last_user)

        model_name = _MODEL_PROFILES[mode.value]
        litellm_model = f"ollama/{model_name}"

        logger.info(
            "ComputeRouter: mode=%s model=%s (api_base=%s)",
            mode.value,
            model_name,
            self.api_base,
        )

        return litellm.completion(
            model=litellm_model,
            messages=messages,
            api_base=self.api_base,
            stream=stream,
            **kwargs,
        )

    def stream_complete(
        self,
        messages: list[dict[str, str]],
        mode: ComputeMode | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Convenience streaming wrapper.

        Yields
        ------
        str
            Individual content delta chunks from the model.
        """
        response = self.complete(messages, mode=mode, stream=True, **kwargs)
        if isinstance(response, _NoOpResponse):
            return
        for chunk in response:
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content


class _NoOpResponse:
    """Minimal stand-in returned when litellm is unavailable."""

    @property
    def choices(self) -> list[Any]:
        return []

    def __iter__(self) -> Iterator[Any]:
        return iter([])
