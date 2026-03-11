"""
Victor — the central orchestration class.

Wires together all five subsystems:

1. **MemoryManager** — Ego Core (Mem0 / pgvector)
2. **ComputeRouter** — Endocrine System (Ollama / LiteLLM)
3. **EthicsGuardrail** — Ethica Moral Fabric (NeMo Guardrails)
4. **EnvironmentBridge** — Symbiotic Bridge (Home Assistant WebSockets)
5. **AgentExecutor** — Reality Translation Engine (LangGraph)

Typical usage::

    from victor import Victor

    v = Victor(user_id="alice")
    response = v.chat("Summarise the files in my ~/Documents folder.")
    print(response)
"""

from __future__ import annotations

import logging
from typing import Any

from victor.agent.executor import AgentExecutor
from victor.compute.router import ComputeMode, ComputeRouter
from victor.environment.bridge import EnvironmentBridge
from victor.guardrails.ethics import EthicsGuardrail
from victor.memory.manager import MemoryManager

logger = logging.getLogger(__name__)


class Victor:
    """The main Victor AI interface.

    Parameters
    ----------
    user_id:
        Stable identifier for the current user/session.  Used to scope
        long-term memories.
    memory_config:
        Optional Mem0 configuration dict.  See :class:`MemoryManager`.
    ha_url:
        Home Assistant WebSocket URL.
    ha_token:
        Home Assistant long-lived access token.
    enable_environment:
        Whether to start the Home Assistant WebSocket bridge on init.
    agent_model:
        Ollama model used by the LangGraph agent executor.
    colang_dir:
        Directory containing NeMo Guardrails Colang rule files.
    """

    def __init__(
        self,
        user_id: str = "victor_default",
        memory_config: dict[str, Any] | None = None,
        ha_url: str | None = None,
        ha_token: str | None = None,
        enable_environment: bool = False,
        agent_model: str | None = None,
        colang_dir: str | None = None,
    ) -> None:
        self.user_id = user_id

        # 1. Ego Core — persistent memory
        self.memory = MemoryManager(user_id=user_id, config=memory_config)

        # 2. Endocrine System — dynamic compute routing
        self.compute = ComputeRouter()

        # 3. Ethica Moral Fabric — ethical guardrails
        self.guardrails = EthicsGuardrail(colang_dir=colang_dir)

        # 4. Symbiotic Bridge — environmental awareness
        env_kwargs: dict[str, Any] = {}
        if ha_url:
            env_kwargs["ha_url"] = ha_url
        if ha_token:
            env_kwargs["ha_token"] = ha_token
        self.environment = EnvironmentBridge(**env_kwargs)
        if enable_environment:
            self.environment.start()

        # 5. Reality Translation Engine — agentic execution
        agent_kwargs: dict[str, Any] = {}
        if agent_model:
            agent_kwargs["model"] = agent_model
        self.agent = AgentExecutor(**agent_kwargs)

        logger.info("Victor initialised for user_id=%r", user_id)

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """Process a natural-language message and return Victor's response.

        The pipeline is:

        1. Query long-term memory for relevant context.
        2. Build environment snapshot (if bridge is connected).
        3. Route the completion to the appropriate compute tier.
        4. Pass the proposed response through the ethics guardrail.
        5. Store the interaction in long-term memory.
        6. Return the safe, context-aware response.

        Parameters
        ----------
        user_message:
            The user's message / instruction.

        Returns
        -------
        str
            Victor's response.
        """
        # --- Rapid safety pre-check on the input itself ------------------
        if not self.guardrails.is_safe(user_message):
            blocked = (
                "I'm unable to process that request — it has been intercepted "
                "by the Ethica guardrail."
            )
            logger.warning("Input blocked by guardrail: %r", user_message[:120])
            return blocked

        # --- Build context snippet from memory and environment -----------
        memory_snippet = self.memory.build_context_snippet(user_message)
        env_snippet = self.environment.build_context_snippet()

        extra_context_parts = []
        if memory_snippet:
            extra_context_parts.append(memory_snippet)
        if env_snippet:
            extra_context_parts.append(env_snippet)
        extra_context = "\n\n".join(extra_context_parts)

        # --- Build message list ------------------------------------------
        messages: list[dict[str, str]] = []
        if extra_context:
            messages.append({"role": "system", "content": extra_context})
        messages.append({"role": "user", "content": user_message})

        # --- Route to appropriate compute tier ---------------------------
        compute_mode = self.compute.detect_mode(user_message)
        logger.debug("Compute mode selected: %s", compute_mode)
        completion = self.compute.complete(messages, mode=compute_mode)

        # Extract text from the completion object
        raw_response = _extract_content(completion)

        # --- Ethical output filter ---------------------------------------
        messages.append({"role": "assistant", "content": raw_response})
        safe_response = self.guardrails.apply(messages)
        if not safe_response:
            safe_response = raw_response

        # --- Persist interaction to long-term memory ---------------------
        self.memory.add(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": safe_response},
            ]
        )

        return safe_response

    def run_task(self, task: str) -> str:
        """Execute an agentic, multi-step task using the LangGraph executor.

        Unlike :meth:`chat`, which performs a single-shot completion,
        :meth:`run_task` allows Victor to reason, call tools, observe
        results, and iterate until the task is complete.

        Parameters
        ----------
        task:
            A natural-language description of the task to complete.

        Returns
        -------
        str
            The final result of the agentic execution.
        """
        if not self.guardrails.is_safe(task):
            return (
                "Task blocked by the Ethica guardrail — this action is not "
                "permitted under the current safety policy."
            )

        memory_snippet = self.memory.build_context_snippet(task)
        env_snippet = self.environment.build_context_snippet()
        extra = "\n\n".join(filter(None, [memory_snippet, env_snippet]))

        result = self.agent.run(task, extra_context=extra)

        # Persist the task and result
        self.memory.add(
            [
                {"role": "user", "content": f"[task] {task}"},
                {"role": "assistant", "content": result},
            ]
        )
        return result

    def shutdown(self) -> None:
        """Gracefully shut down background services."""
        self.environment.stop()
        logger.info("Victor shut down.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_content(completion: Any) -> str:
    """Extract the text content from a LiteLLM ModelResponse or no-op object."""
    try:
        return completion.choices[0].message.content or ""
    except (AttributeError, IndexError):
        # If the completion itself is a plain string, return it directly.
        if isinstance(completion, str):
            return completion
        return ""
