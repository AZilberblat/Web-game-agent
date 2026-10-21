"""Human-readable per-run report built only from the durable log."""
from __future__ import annotations

import json

from .runlog import RunLog


def report(log: RunLog, run_id: str, tail: int = 8) -> str:
    run = log.run_row(run_id)
    if run is None:
        return f"unknown run {run_id}"
    steps, events, tot = log.steps(run_id), log.events(run_id), log.totals(run_id)
    out = [f"RUN {run_id}  game={run['game']}  status={run['status']}  stop_reason={run['stop_reason']}",
           f"game actions: {log.actions_used(run_id)}   steps: {len(steps)}   elapsed: {log.max_elapsed(run_id):.1f}s   "
           f"model calls: {tot['model_calls']}   tokens: {tot['tokens_in']}in/{tot['tokens_out']}out   cost: ${tot['cost_usd']:.4f}"]
    if run["status"] == "not_completed":
        out.append(f"NOT COMPLETED: {run['stop_reason']}")
    elif run["status"] in ("running", "crashed"):
        out.append("RUN DID NOT FINISH CLEANLY (resume from log)")
    out.append("\nSteps:")
    for r in steps:
        a = json.loads(r["action_json"])
        out.append(f" {r['step']:>3} [{r['subgoal']}] {a} -> {r['classification'] or 'NO RESULT (outcome unknown)'}"
                   f" | expected: {r['expected_outcome']} | observed: {r['observed_result'] or ''} | {r['evidence'] or ''}"
                   + (f" | ERR {r['errors']}" if r["errors"] else ""))
    out.append("\nPolicy / network / safety events (blocked attempts vs actual leaks):")
    sec = [e for e in events if e["kind"] in ("policy_block", "network_block", "leak", "safety_alert")]
    out += [f" step {e['step']} {e['kind']}: {e['detail_json']}" for e in sec] or [" none"]
    out.append("\nOther events:")
    out += [f" step {e['step']} {e['kind']}: {e['detail_json'][:200]}" for e in events
            if e["kind"] in ("runtime_error", "model_error", "escalation", "success_check", "resume", "stop", "crash")] or [" none"]
    # why-stuck hints
    done = [r for r in steps if r["classification"]]
    hints = []
    acts = [r["action_json"] for r in done]
    for a in set(acts):
        n = acts.count(a)
        if n >= 3 and all(r["classification"] != "success" for r in done if r["action_json"] == a):
            hints.append(f"action repeated {n}x without success: {a}")
    if sum(r["classification"] == "unknown" for r in done) >= max(3, len(done) // 2):
        hints.append("many 'unknown' results: check settle delay and whether the success/expected cues are visible")
    out.append("\nHints: " + ("; ".join(hints) if hints else "none"))
    return "\n".join(out)
