"""
Reality Translation Engine — LangGraph agentic executor.

Builds a cyclic ReAct-style graph where Victor can:

1. Reason about a task.
2. Select and call a tool.
3. Observe the result.
4. Loop until the task is complete or a stop condition is met.

This gives Victor "hands" — the ability to manipulate files, send emails,
fetch web pages, and interact with external APIs rather than just generating
text.
"""

from __future__ import annotations

import logging
import os
from typing import Any, TypedDict

logger = logging.getLogger(__name__)

# LangGraph + LangChain imports are optional so the module can be imported
# without them installed (tests can mock the internals).
try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
    from langchain_core.runnables import RunnableConfig
    from langchain_community.chat_models import ChatOllama  # type: ignore[import-untyped]
    from langgraph.graph import END, StateGraph  # type: ignore[import-untyped]
    from langgraph.prebuilt import ToolNode  # type: ignore[import-untyped]

    _LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LANGGRAPH_AVAILABLE = False
    BaseMessage = object  # type: ignore[assignment,misc]

from victor.agent.tools import ALL_TOOLS

_OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_AGENT_MODEL = os.getenv("AGENT_MODEL", "llama3")


class AgentState(TypedDict):
    """Shared state passed between graph nodes."""

    messages: list[Any]


def _should_continue(state: AgentState) -> str:
    """Determine the next node: tool call or end."""
    messages = state["messages"]
    last = messages[-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def _call_model(state: AgentState, llm: Any) -> AgentState:
    """Invoke the LLM with the current message history."""
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}


class AgentExecutor:
    """LangGraph-powered agentic executor with pluggable tools.

    Parameters
    ----------
    tools:
        List of LangChain tool objects to bind to the agent.  Defaults to
        :data:`victor.agent.tools.ALL_TOOLS`.
    model:
        Ollama model name.
    ollama_base_url:
        Ollama server base URL.
    max_iterations:
        Maximum number of tool-call/observe cycles before forcing termination.
    """

    def __init__(
        self,
        tools: list[Any] | None = None,
        model: str = _AGENT_MODEL,
        ollama_base_url: str = _OLLAMA_BASE,
        max_iterations: int = 10,
    ) -> None:
        self._tools = tools if tools is not None else ALL_TOOLS
        self._model_name = model
        self._ollama_base_url = ollama_base_url
        self._max_iterations = max_iterations
        self._graph: Any = None

        if not _LANGGRAPH_AVAILABLE:
            logger.warning(
                "langgraph / langchain_community not installed.  "
                "AgentExecutor will operate in no-op mode.  "
                "Run `pip install langgraph langchain-community`."
            )
        else:
            self._graph = self._build_graph()
            logger.info(
                "AgentExecutor initialised with %d tools, model=%s",
                len(self._tools),
                model,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, task: str, extra_context: str = "") -> str:
        """Execute *task* and return the final textual result.

        Parameters
        ----------
        task:
            Natural-language description of what Victor should do.
        extra_context:
            Optional additional context (memories, env state) prepended to
            the system prompt.

        Returns
        -------
        str
            Victor's final response after all tool calls are resolved.
        """
        if self._graph is None:
            return f"[no-op] Task received: {task}"

        system_content = (
            "You are Victor, a highly capable autonomous AI agent operating "
            "under the Ethica AI ethical framework.\n"
            "Use the available tools to accomplish the user's task step by step.\n"
        )
        if extra_context:
            system_content += f"\n{extra_context}\n"

        from langchain_core.messages import SystemMessage  # type: ignore[import-untyped]

        initial_state: AgentState = {
            "messages": [
                SystemMessage(content=system_content),
                HumanMessage(content=task),
            ]
        }

        config: RunnableConfig = {"recursion_limit": self._max_iterations * 2 + 2}
        result = self._graph.invoke(initial_state, config=config)
        messages = result.get("messages", [])

        # Return the last AI message content.
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                return msg.content or ""
        return ""

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
        """Compile and return the LangGraph state machine."""
        llm = ChatOllama(
            model=self._model_name,
            base_url=self._ollama_base_url,
        ).bind_tools(self._tools)

        tool_node = ToolNode(self._tools)

        graph = StateGraph(AgentState)
        graph.add_node("agent", lambda state: _call_model(state, llm))
        graph.add_node("tools", tool_node)

        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
        graph.add_edge("tools", "agent")

        return graph.compile()
