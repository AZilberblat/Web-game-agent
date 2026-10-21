"""The LangGraph loop (section 4).

  setup (once) -> guard -> capture -> decide -> gate -> intent -> execute -> verify -> record -> guard ...

`guard` is the only node that ends the run; it runs before every dispatch, and `gate` re-checks the clock
after the model call because model latency counts against the 5-minute budget.
"""
from __future__ import annotations

import time
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from .brain import Brain, DecideContext, SetupInfo, Usage, VerifyContext
from .browser import Browser, digest, settle
from .config import Config, GameSpec
from .policy import TOOL_PARAMS, Policy
from .runlog import RunLog
from .tools import dispatch, make_tool_node

GOAL = "Complete level 1 of this game."


class State(TypedDict, total=False):
    run_id: str
    step: int
    actions_used: int
    subgoal: str  # "start" (one-time) then "play"
    setup_done: bool
    knowledge: dict  # instructions, start_hint, learned[]
    history: list  # compact per-step records; sliced before going to the model
    notes: list  # one-shot notes for the next model call (e.g. "previous outcome unknown after crash")
    obs_path: str | None
    proposal: Any
    model: str
    action: dict | None  # policy-approved {tool, args, expected, counts}
    exec_error: str | None
    no_progress: int
    fail_streak: int
    blocked_attempts: int
    runtime_errors: int
    success: bool
    status: str  # running | done
    stop_reason: str
    cls: str  # verify -> record hand-off
    obs_text: str
    step_error: bool


def short(tool: str, args: dict) -> str:
    return f"{tool}({','.join(str(v) for v in args.values())})"


class Agent:
    def __init__(self, spec: GameSpec, cfg: Config, browser: Browser, brain: Brain, log: RunLog, run_id: str,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 elapsed_base: float = 0.0):
        self.spec, self.cfg, self.browser, self.brain, self.log, self.run_id = spec, cfg, browser, brain, log, run_id
        self.clock, self.sleep = clock, sleep
        self.t0 = clock() - elapsed_base  # elapsed_base carries time spent before a resume
        self.policy = Policy(spec.hosts(), spec.viewport, cfg.max_wait_action)
        self.tool_node = make_tool_node(browser, self.remaining_seconds, sleep)
        self.dir = log.run_dir(run_id)
        self.png: bytes = b""  # current observation (kept out of graph state)

    # ---- helpers
    def elapsed(self) -> float:
        return self.clock() - self.t0

    def remaining_seconds(self) -> float:
        return max(0.0, self.cfg.max_seconds - self.elapsed())

    def ev(self, s: State, kind: str, **detail) -> None:
        self.log.event(self.run_id, s.get("step", 0), self.elapsed(), kind, detail)

    def usages(self, s: State, us: list[Usage]) -> None:
        for u in us:
            self.log.model_call(self.run_id, s.get("step", 0), u.role, u.model, u.tokens_in, u.tokens_out,
                                u.cost_usd, u.latency_s)

    def save(self, name: str, png: bytes) -> str:
        p = self.dir / name
        p.write_bytes(png)
        return str(p.relative_to(self.log.root))

    def runtime_error(self, s: State, where: str, e: Exception) -> dict:
        self.ev(s, "runtime_error", where=where, error=repr(e))
        return {"runtime_errors": s.get("runtime_errors", 0) + 1}

    # ---- nodes
    def setup(self, s: State) -> dict:
        step = s.get("step", 0)
        info = SetupInfo(self.spec.instructions or "", None)
        try:
            png = self.browser.screenshot()
            self.png = png
            self.save("setup.png", png)
            info, us = self.brain.setup(png, self.spec)
            self.usages(s, us)
        except Exception as e:  # setup is best effort: the loop can still read the screen each iteration
            self.ev(s, "runtime_error", where="setup", error=repr(e))
        if self.spec.instructions and not info.instructions:
            info.instructions = self.spec.instructions
        know = {"instructions": info.instructions[:800], "start_hint": info.start_action, "learned": []}
        self.log.set_setup(self.run_id, know)
        self.ev(s, "setup", instructions=info.instructions[:300], start_action=info.start_action)
        return {"setup_done": True, "knowledge": know, "step": step,
                "subgoal": "start" if info.start_action else "play"}

    def guard(self, s: State) -> dict:
        c = self.cfg
        reason = None
        if s.get("success"):
            reason = "success"
        elif s.get("actions_used", 0) >= c.max_actions:
            reason = "budget_actions"
        elif self.elapsed() >= c.max_seconds:
            reason = "budget_time"
        elif s.get("blocked_attempts", 0) >= c.max_blocked_attempts:
            reason = "safety_blocked_attempts"
        elif s.get("runtime_errors", 0) >= c.max_runtime_errors:
            reason = "unrecoverable_error"
        elif s.get("fail_streak", 0) >= c.max_consecutive_failures:
            reason = "failure_limit"
        elif s.get("step", 0) >= c.max_iterations:
            reason = "iteration_cap"
        if reason:
            self.ev(s, "stop", reason=reason, actions_used=s.get("actions_used", 0))
            return {"status": "done", "stop_reason": reason}
        return {"status": "running"}

    def capture(self, s: State) -> dict:
        step = s.get("step", 0) + 1
        last: Exception | None = None
        for _ in range(3):  # screenshot is a safe step: retry it
            try:
                self.png = self.browser.screenshot()
                last = None
                break
            except Exception as e:
                last = e
                self.sleep(0.5)
        if last is not None:
            return {"step": step, "obs_path": None, **self.runtime_error({**s, "step": step}, "screenshot", last)}
        return {"step": step, "obs_path": self.save(f"step_{step:03d}_before.png", self.png)}

    def _context(self, s: State) -> DecideContext:
        hist = s.get("history", [])
        succ = [h for h in hist if h.get("result") == "success"][-self.cfg.history_successes:]
        recent = hist[-self.cfg.history_recent:]
        sl = sorted({h["step"]: h for h in succ + recent}.values(), key=lambda h: h["step"])
        tried: dict[str, dict] = {}
        for h in hist:
            if h.get("act"):
                t = tried.setdefault(h["act"], {})
                t[h["result"]] = t.get(h["result"], 0) + 1
        know = s.get("knowledge", {})
        state = {"instructions": know.get("instructions", ""), "start_hint": know.get("start_hint"),
                 "learned": know.get("learned", [])[-10:], "tried_action_outcomes": dict(list(tried.items())[-15:]),
                 "notes": s.get("notes", [])}
        return DecideContext(
            goal=GOAL, subgoal=s.get("subgoal", "play"), screenshot=self.png, state=state, history=sl,
            tools={k: sorted(v) for k, v in TOOL_PARAMS.items()},
            budget={"actions_left": self.cfg.max_actions - s.get("actions_used", 0),
                    "seconds_left": round(self.remaining_seconds())},
            no_progress=s.get("no_progress", 0))

    def decide(self, s: State) -> dict:
        ctx = self._context(s)
        err: Exception | None = None
        for _ in range(2):
            try:
                p = self.brain.decide(ctx)
                self.usages(s, p.usages)
                if p.escalated:
                    self.ev(s, "escalation", model=p.model, no_progress=s.get("no_progress", 0))
                return {"proposal": p.action, "model": p.model, "notes": []}
            except Exception as e:
                err = e
                self.ev(s, "model_error", error=repr(e))
        return {"proposal": None, **self.runtime_error(s, "decide", err)}  # type: ignore[arg-type]

    def gate(self, s: State) -> dict:
        prop = s.get("proposal")
        if prop is None:
            return {"action": None}
        if isinstance(prop, dict) and prop.get("safety_alert"):
            self.ev(s, "safety_alert", alert=str(prop["safety_alert"])[:300])  # informational only
        r = self.policy.check(prop)
        if not r.ok:
            self.ev(s, "policy_block", reason=r.reason, proposal=prop if isinstance(prop, (dict, str)) else str(prop))
            h = s.get("history", []) + [{"step": s["step"], "act": None, "result": "rejected_by_policy",
                                          "observed": r.reason}]
            return {"action": None, "history": h, "blocked_attempts": s.get("blocked_attempts", 0) + 1}
        if self.elapsed() >= self.cfg.max_seconds:  # model latency used the time: do not dispatch
            self.ev(s, "stop", reason="budget_time", detail="expired during model call, nothing dispatched")
            return {"action": None, "status": "done", "stop_reason": "budget_time"}
        if r.counts_as_action and s.get("actions_used", 0) >= self.cfg.max_actions:
            return {"action": None}
        return {"action": {"tool": r.tool, "args": r.args, "counts": r.counts_as_action,
                           "expected": prop["expected_outcome"].strip(), "learned": prop.get("learned")}}

    def intent(self, s: State) -> dict:
        a = s["action"]
        self.log.write_intent(self.run_id, s["step"], self.elapsed(), s.get("subgoal"), s.get("obs_path"),
                              {"tool": a["tool"], **a["args"]}, a["expected"], a["counts"], s.get("model"),
                              a.get("learned") if isinstance(a.get("learned"), str) else None)
        return {}

    def execute(self, s: State) -> dict:
        a = s["action"]
        used = s.get("actions_used", 0) + (1 if a["counts"] else 0)  # counted at dispatch
        err = None
        try:
            ok, msg = dispatch(self.tool_node, a["tool"], a["args"])
            if not ok:
                err = f"tool error: {msg}"
        except Exception as e:
            err = f"dispatch exception: {e!r}"  # outcome unknown; never retried
        upd: dict = {"actions_used": used, "exec_error": err}
        if err:
            self.ev(s, "runtime_error", where="execute", error=err)
        try:  # network/URL layer check, runs regardless of what the model expected
            url = self.browser.current_url()
            if not self.policy.url_allowed(url):
                kind = "network_block" if url.startswith("chrome-error") else "leak"
                self.ev(s, kind, url=url, action=short(a["tool"], a["args"]))
                self.browser.restore()
                upd["blocked_attempts"] = s.get("blocked_attempts", 0) + 1
        except Exception as e:
            self.ev(s, "runtime_error", where="url_check", error=repr(e))
        return upd

    def verify(self, s: State) -> dict:
        step, a = s["step"], s["action"]
        before = self.png
        errors = [s["exec_error"]] if s.get("exec_error") else []
        after, cls, observed, evidence, vmodel, claimed = before, "unknown", "", "", None, False
        try:
            budget = min(self.cfg.settle_timeout, self.remaining_seconds())
            after, settled = settle(self.browser, budget, self.cfg.settle_poll, self.sleep, self.clock)
            changed = digest(after) != digest(before)
            if not changed and a["tool"] != "wait" and self.remaining_seconds() > 0:
                # unchanged is not failure: one more bounded wait before judging
                self.sleep(min(self.cfg.unchanged_extra_wait, self.remaining_seconds()))
                after = self.browser.screenshot()
                changed = digest(after) != digest(before)
            if self.remaining_seconds() <= 0:
                evidence = "time budget exhausted before verification"
            else:
                v = self.brain.verify(VerifyContext(self.spec, s.get("subgoal", "play"), {"tool": a["tool"], **a["args"]},
                                                    a["expected"], before, after, changed))
                self.usages(s, v.usages)
                cls, observed, evidence, vmodel, claimed = v.classification, v.observed, v.evidence, v.model, v.level_complete
                if not changed and cls == "failure":
                    cls, evidence = "unknown", f"screen unchanged after settle (not proof of failure). {evidence}"
        except Exception as e:
            errors.append(f"verify error: {e!r}")
            self.ev(s, "runtime_error", where="verify", error=repr(e))
        success = False
        if claimed:  # the model's claim is not independent of the screenshot: confirm, then DOM if configured
            try:
                ok, ev2, us = self.brain.confirm_success(after, self.spec)
                self.usages(s, us)
                dom = None
                if self.spec.dom_success_text:
                    dom = self.spec.dom_success_text.lower() in self.browser.page_text().lower()
                success = ok and dom is not False
                self.ev(s, "success_check", verifier_claim=True, confirm=ok, confirm_evidence=ev2, dom=dom,
                        verified=success)
                if success:
                    cls = "success"
                    evidence += f" | cue confirmed: {ev2}" + (" | DOM text present" if dom else "")
            except Exception as e:
                errors.append(f"success check error: {e!r}")
        res_ref = self.save(f"step_{step:03d}_after.png", after)
        self.log.write_result(self.run_id, step, res_ref, observed, cls, evidence, "; ".join(errors) or None, vmodel,
                              None)
        return {"cls": cls, "obs_text": observed, "step_error": bool(errors), "success": success or s.get("success", False)}

    def record(self, s: State) -> dict:  # fold the verified result into compact state
        a, cls = s["action"], s.get("cls", "unknown")
        h = s.get("history", []) + [{"step": s["step"], "subgoal": s.get("subgoal"), "act": short(a["tool"], a["args"]),
                                     "expected": a["expected"], "result": cls, "observed": s.get("obs_text", "")[:160]}]
        know = dict(s.get("knowledge", {}))
        if isinstance(a.get("learned"), str) and a["learned"].strip():
            know["learned"] = (know.get("learned", []) + [a["learned"].strip()[:200]])[-20:]
        upd: dict = {"history": h, "knowledge": know,
                     "no_progress": 0 if cls == "success" else s.get("no_progress", 0) + 1,
                     "runtime_errors": s.get("runtime_errors", 0) + 1 if s.get("step_error") else 0}
        if a["tool"] != "wait":  # waits neither reset nor extend the failure streak
            upd["fail_streak"] = 0 if cls == "success" else s.get("fail_streak", 0) + 1
        if s.get("subgoal") == "start" and cls == "success":
            upd["subgoal"] = "play"  # start was a one-time subgoal; global objective is unchanged
        return upd


def build_graph(agent: Agent):
    g = StateGraph(State)
    for name in ("setup", "guard", "capture", "decide", "gate", "intent", "execute", "verify", "record"):
        g.add_node(name, getattr(agent, name))
    g.add_conditional_edges(START, lambda s: "guard" if s.get("setup_done") else "setup")
    g.add_edge("setup", "guard")
    g.add_conditional_edges("guard", lambda s: END if s.get("status") == "done" else "capture")
    g.add_conditional_edges("capture", lambda s: "decide" if s.get("obs_path") else "guard")
    g.add_conditional_edges("decide", lambda s: "gate" if s.get("proposal") is not None else "guard")
    g.add_conditional_edges("gate", lambda s: END if s.get("status") == "done" else ("intent" if s.get("action") else "guard"))
    g.add_edge("intent", "execute")
    g.add_edge("execute", "verify")
    g.add_edge("verify", "record")
    g.add_edge("record", "guard")
    return g.compile()
