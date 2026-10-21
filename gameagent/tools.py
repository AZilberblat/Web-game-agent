"""The only tools that exist, as a LangGraph ToolNode. The ToolNode itself is the allowlist: a tool call whose
name is not in this list cannot execute. Parameter validation happens earlier in Policy.check."""
from __future__ import annotations

import time
from typing import Callable

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from .browser import Browser
from .policy import TOOL_PARAMS


def make_tool_node(browser: Browser, remaining_seconds: Callable[[], float],
                   sleep: Callable[[float], None] = time.sleep):
    @tool
    def press_mouse(coord: list[int]) -> str:
        """Click at pixel coordinate [x, y] of the screenshot."""
        browser.click(int(coord[0]), int(coord[1]))
        return "ok"

    @tool
    def press_key(key: str) -> str:
        """Press one allowed key."""
        browser.press_key(key)
        return "ok"

    @tool
    def wait(seconds: float) -> str:
        """Wait (no game action). Clamped to the remaining time budget."""
        sleep(max(0.0, min(seconds, remaining_seconds())))
        return "ok"

    tools = [press_mouse, press_key, wait]
    # A tool with no policy entry gets no permissions: refuse to build the node.
    unknown = {t.name for t in tools} - set(TOOL_PARAMS)
    if unknown:
        raise ValueError(f"tools without a policy entry: {sorted(unknown)}")
    node = ToolNode(tools)  # the allowlist: only these names can execute
    g = StateGraph(MessagesState)  # wrapped so it also runs standalone (ToolNode wants a graph runtime)
    g.add_node("tools", node)
    g.add_edge(START, "tools")
    g.add_edge("tools", END)
    return g.compile()


def dispatch(node, name: str, args: dict) -> tuple[bool, str]:
    """Run one tool call through the ToolNode. Returns (ok, message)."""
    msg = AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call_0", "type": "tool_call"}])
    out = node.invoke({"messages": [msg]})["messages"][-1]
    return getattr(out, "status", "success") != "error", str(out.content)
