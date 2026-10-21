"""Run / resume entry points. One run at a time."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Callable

from .brain import Brain
from .browser import Browser
from .config import Config, GameSpec
from .graph import Agent, State, build_graph, short
from .runlog import RunLog


@dataclass
class RunResult:
    run_id: str
    reached: bool
    stop_reason: str
    actions: int
    seconds: float
    cost_usd: float
    blocked_attempts: int  # stopped by policy gate or network layer
    leaks: int  # off-site content actually loaded
    model_calls: int


def _result(log: RunLog, run_id: str, state: dict, seconds: float) -> RunResult:
    ev = log.events(run_id)
    kinds = [e["kind"] for e in ev]
    return RunResult(run_id, bool(state.get("success")), state.get("stop_reason", "crashed"),
                     log.actions_used(run_id), round(seconds, 2), log.totals(run_id)["cost_usd"],
                     kinds.count("policy_block") + kinds.count("network_block"), kinds.count("leak"),
                     log.totals(run_id)["model_calls"])


def _invoke(agent: Agent, log: RunLog, run_id: str, init: State, cfg: Config, close: Callable[[], None] | None):
    final: dict = dict(init)
    try:
        final = build_graph(agent).invoke(init, {"recursion_limit": cfg.max_iterations * 14 + 50})
        reason = final.get("stop_reason", "unknown")
        log.finish_run(run_id, "completed" if final.get("success") else "not_completed", reason)
    except Exception as e:  # logged; the run can be resumed from the log
        log.event(run_id, init.get("step", 0), agent.elapsed(), "crash", {"error": repr(e)})
        log.finish_run(run_id, "crashed", repr(e))
        final = {**init, "stop_reason": f"crashed: {e!r}"}
    finally:
        if close:
            close()
    return _result(log, run_id, final, agent.elapsed())


def run_game(spec: GameSpec, cfg: Config, browser_factory: Callable[[], Browser], brain: Brain,
             run_id: str | None = None, clock=time.monotonic, sleep=time.sleep) -> RunResult:
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    log = RunLog(cfg.runs_dir)
    log.start_run(run_id, spec.name, {"config": cfg.as_json(), "spec": asdict(spec)})
    start = clock()  # the 5 minutes include browser start-up
    browser = browser_factory()
    agent = Agent(spec, cfg, browser, brain, log, run_id, clock, sleep, elapsed_base=clock() - start)
    return _invoke(agent, log, run_id, {"run_id": run_id, "step": 0, "actions_used": 0, "history": [], "notes": [],
                                       "no_progress": 0, "fail_streak": 0, "blocked_attempts": 0,
                                       "runtime_errors": 0, "success": False}, cfg, browser.close)


def resume_run(run_id: str, spec: GameSpec, cfg: Config, browser: Browser, brain: Brain,
               clock=time.monotonic, sleep=time.sleep) -> RunResult:
    """Continue a run from its log. An intent with no result is an action whose outcome is UNKNOWN: it is
    recorded as such, counted as used, and never re-dispatched; the agent re-observes with a fresh screenshot."""
    log = RunLog(cfg.runs_dir)
    row = log.run_row(run_id)
    if row is None:
        raise KeyError(run_id)
    notes: list[str] = []
    base = log.max_elapsed(run_id)
    agent = Agent(spec, cfg, browser, brain, log, run_id, clock, sleep, elapsed_base=base)
    for d in log.dangling_intents(run_id):
        log.write_result(run_id, d["step"], None, "", "unknown", "interrupted before verification; NOT retried",
                         "no result record at resume", None, None)
        notes.append(f"step {d['step']} ({d['action_json']}) was dispatched but its outcome is unknown; "
                     "do not assume it happened or did not happen - judge from the current screen")
    log.event(run_id, log.max_step(run_id), base, "resume", {"unknown_outcomes": len(notes)})
    log.db.execute("UPDATE runs SET status='running', stop_reason=NULL WHERE run_id=?", (run_id,))
    log.db.commit()

    hist, learned = [], []
    for r in log.steps(run_id):
        a = json.loads(r["action_json"])
        hist.append({"step": r["step"], "subgoal": r["subgoal"], "act": short(a["tool"], {k: v for k, v in a.items() if k != "tool"}),
                     "expected": r["expected_outcome"], "result": r["classification"], "observed": (r["observed_result"] or "")[:160]})
        if r["learned"]:
            learned.append(r["learned"])
    ev = log.events(run_id)
    kinds = [e["kind"] for e in ev]
    know = json.loads(row["setup_json"] or "{}") or {"instructions": spec.instructions or "", "learned": []}
    know["learned"] = learned[-20:]
    tail_np = 0
    for h in reversed(hist):
        if h["result"] == "success":
            break
        tail_np += 1
    started = any(h["subgoal"] == "start" and h["result"] == "success" for h in hist)
    init: State = {"run_id": run_id, "step": log.max_step(run_id), "actions_used": log.actions_used(run_id),
                   "history": hist, "notes": notes, "knowledge": know, "setup_done": True,
                   "subgoal": "play" if started or not know.get("start_hint") else "start",
                   "no_progress": tail_np, "fail_streak": 0, "runtime_errors": 0, "success": False,
                   "blocked_attempts": kinds.count("policy_block") + kinds.count("network_block") + kinds.count("leak")}
    return _invoke(agent, log, run_id, init, cfg, None)
