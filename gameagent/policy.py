"""Code-level policy: the single registry of allowed tools/parameters, and the pre-execution gate.

Nothing here depends on the model behaving. A tool that is not in TOOL_PARAMS cannot be built into the
ToolNode (tools.make_tool_node asserts this) and cannot pass `Policy.check`, so a newly connected tool
has no permissions until it is added here deliberately.
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

# tool name -> exact set of parameter names. No url / path / file parameters exist.
TOOL_PARAMS: dict[str, frozenset[str]] = {
    "press_mouse": frozenset({"coord"}),
    "press_key": frozenset({"key"}),
    "wait": frozenset({"seconds"}),  # a check/wait: not a game action
}
COUNTS_AS_GAME_ACTION = {"press_mouse": True, "press_key": True, "wait": False}

# Fields the model may return alongside the action; everything else is rejected.
META_FIELDS = frozenset({"action", "expected_outcome", "confidence", "learned", "safety_alert"})

ALLOWED_KEYS = frozenset(
    list("abcdefghijklmnopqrstuvwxyz0123456789")
    + ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Space", "Enter", "Escape"]
)


@dataclass
class PolicyResult:
    ok: bool
    reason: str = ""
    tool: str = ""
    args: dict | None = None
    counts_as_action: bool = False


class Policy:
    def __init__(self, allowed_hosts: tuple[str, ...], viewport: tuple[int, int], max_wait: float):
        self.allowed_hosts = tuple(h.lower() for h in allowed_hosts if h)
        self.viewport = viewport
        self.max_wait = max_wait

    def url_allowed(self, url: str) -> bool:
        if url in ("", "about:blank"):
            return True
        p = urlparse(url)
        return p.scheme in ("http", "https") and (p.hostname or "").lower() in self.allowed_hosts

    def check(self, proposal: object) -> PolicyResult:
        if not isinstance(proposal, dict):
            return PolicyResult(False, "action is not an object")
        name = proposal.get("action")
        if not isinstance(name, str) or name not in TOOL_PARAMS:
            return PolicyResult(False, f"tool not allowed: {name!r}")
        extra = set(proposal) - META_FIELDS - TOOL_PARAMS[name]
        if extra:
            return PolicyResult(False, f"parameters not allowed: {sorted(extra)}", tool=name)
        missing = TOOL_PARAMS[name] - set(proposal)
        if missing:
            return PolicyResult(False, f"missing parameters: {sorted(missing)}", tool=name)
        exp = proposal.get("expected_outcome")
        if not isinstance(exp, str) or not exp.strip() or len(exp) > 400:
            return PolicyResult(False, "expected_outcome must be a non-empty string (<=400 chars)", tool=name)

        args: dict
        if name == "press_mouse":
            c = proposal["coord"]
            w, h = self.viewport
            if (
                not isinstance(c, (list, tuple))
                or len(c) != 2
                or not all(isinstance(v, int) and not isinstance(v, bool) for v in c)
                or not (0 <= c[0] < w and 0 <= c[1] < h)
            ):
                return PolicyResult(False, f"coord must be two ints inside the {w}x{h} viewport", tool=name)
            args = {"coord": [c[0], c[1]]}
        elif name == "press_key":
            k = proposal["key"]
            if not isinstance(k, str) or k not in ALLOWED_KEYS:
                return PolicyResult(False, f"key not allowed: {k!r}", tool=name)
            args = {"key": k}
        else:
            s = proposal["seconds"]
            if isinstance(s, bool) or not isinstance(s, (int, float)) or not (0 < s <= self.max_wait):
                return PolicyResult(False, f"seconds must be in (0, {self.max_wait}]", tool=name)
            args = {"seconds": float(s)}
        return PolicyResult(True, "", name, args, COUNTS_AS_GAME_ACTION[name])
