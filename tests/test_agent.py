import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from fakes import SPEC, WIN, Clock, CrashNow, FakeBrowser, ScriptedBrain, act  # noqa: E402

from gameagent.brain import AnthropicBrain, DecideContext  # noqa: E402
from gameagent.config import Config  # noqa: E402
from gameagent.evaluate import metrics, should_switch  # noqa: E402
from gameagent.policy import Policy  # noqa: E402
from gameagent.report import report  # noqa: E402
from gameagent.run import resume_run, run_game  # noqa: E402
from gameagent.runlog import RunLog  # noqa: E402
from gameagent.tools import dispatch, make_tool_node  # noqa: E402


def cfg(tmp_path, **kw):
    return Config(runs_dir=str(tmp_path), settle_timeout=0.5, settle_poll=0.1, unchanged_extra_wait=0.1, **kw)


def go(tmp_path, browser, brain, clock=None, **kw):
    clock = clock or Clock()
    return run_game(SPEC, cfg(tmp_path, **kw), lambda: browser, brain, run_id="r1", clock=clock, sleep=clock.sleep)


def test_success_path_and_log(tmp_path):
    b = FakeBrowser()
    res = go(tmp_path, b, ScriptedBrain(b, WIN))
    assert res.reached and res.stop_reason == "success" and res.actions == 3
    log = RunLog(tmp_path)
    rows = log.steps("r1")
    assert [r["classification"] for r in rows] == ["success"] * 3
    assert rows[0]["subgoal"] == "start" and rows[1]["subgoal"] == "play"  # start is a one-time subgoal
    assert all(r["ts_intent"] <= r["ts_result"] and r["observation_ref"] and r["result_ref"] for r in rows)
    assert log.run_row("r1")["status"] == "completed"
    assert "NOT COMPLETED" not in report(log, "r1")


def test_action_budget_checked_before_dispatch(tmp_path):
    b = FakeBrowser(hits_needed=999)
    res = go(tmp_path, b, ScriptedBrain(b, WIN), max_actions=5, max_consecutive_failures=99)
    assert not res.reached and res.stop_reason == "budget_actions"
    assert len(b.clicks) == 5 == res.actions
    assert "NOT COMPLETED: budget_actions" in report(RunLog(tmp_path), "r1")


def test_time_budget_includes_model_latency_and_blocks_dispatch(tmp_path):
    clock, b = Clock(), FakeBrowser()
    brain = ScriptedBrain(b, WIN, clock=clock, latency=200.0)  # each model call burns 200s
    res = go(tmp_path, b, brain, clock=clock)
    assert res.stop_reason == "budget_time" and not res.reached
    assert len(b.clicks) == 1  # 2nd model call finished at t=400 > 300: nothing dispatched after it


def test_policy_blocks_before_execution_and_safety_stop(tmp_path):
    b = FakeBrowser()
    bad = [{"action": "navigate", "url": "http://evil.test", "expected_outcome": "x"},
           act(coord=[1, 1], url="http://evil.test"), act("press_key", key="ctrl+w"),
           act(coord=[9999, 1]), "not json at all"]
    res = go(tmp_path, b, ScriptedBrain(b, bad))
    assert res.stop_reason == "safety_blocked_attempts" and res.blocked_attempts == 5
    assert b.clicks == [] and b.keys == [] and res.actions == 0
    assert res.leaks == 0 and len(RunLog(tmp_path).events("r1", "policy_block")) == 5


def test_network_block_logged_and_restored(tmp_path):
    from fakes import LINK
    b = FakeBrowser(hits_needed=1)
    script = [act(coord=[200, 150]), act(coord=list(LINK)), act(coord=[100, 100])]
    res = go(tmp_path, b, ScriptedBrain(b, script))
    assert res.reached and res.blocked_attempts == 1 and res.leaks == 0
    assert b.url == "http://game.test/"
    assert len(RunLog(tmp_path).events("r1", "network_block")) == 1


def test_crash_between_intent_and_result_then_resume_does_not_retry(tmp_path):
    b = FakeBrowser(hits_needed=1)
    b.crash_on_click = 2  # dies while dispatching the 2nd action
    with pytest.raises(CrashNow):
        go(tmp_path, b, ScriptedBrain(b, WIN))
    log = RunLog(tmp_path)
    assert [r["step"] for r in log.dangling_intents("r1")] == [2]  # intent durable, no result
    clicks_before = len(b.clicks)
    b.crash_on_click = None
    clock = Clock()
    brain = ScriptedBrain(b, [act(coord=[100, 100])])
    res = resume_run("r1", SPEC, cfg(tmp_path), b, brain, clock=clock, sleep=clock.sleep)
    rows = log.steps("r1")
    assert rows[1]["classification"] == "unknown" and "NOT retried" in rows[1]["evidence"]
    assert clicks_before == 2 and len(b.clicks) == 3  # exactly one new click: the model's fresh choice
    assert any("unknown" in n for n in brain.decide_contexts[0].state["notes"])
    assert res.actions == 3 and res.reached  # interrupted action still counts toward the budget


def test_unchanged_screen_is_unknown_not_failure(tmp_path):
    b = FakeBrowser(hits_needed=999)
    brain = ScriptedBrain(b, [act(coord=[5, 5])], verdict_override="failure")
    go(tmp_path, b, brain, max_actions=2)
    rows = RunLog(tmp_path).steps("r1")
    assert {r["classification"] for r in rows} == {"unknown"}


def test_lagging_screen_settles_instead_of_failing(tmp_path):
    b = FakeBrowser(hits_needed=1, lag_frames=2)
    res = go(tmp_path, b, ScriptedBrain(b, WIN))
    assert res.reached


def test_success_requires_dom_evidence_when_configured(tmp_path):
    b = FakeBrowser(hits_needed=1, dom_text_on_complete=False)
    res = go(tmp_path, b, ScriptedBrain(b, WIN), max_actions=6)
    assert not res.reached  # model + screenshot claim success, DOM disagrees
    assert any(json.loads(e["detail_json"])["verified"] is False for e in RunLog(tmp_path).events("r1", "success_check"))


def test_runtime_errors_are_recovered_then_limit_stops(tmp_path):
    b = FakeBrowser()
    b.fail_screenshots = 2  # recovered by retry
    assert go(tmp_path, b, ScriptedBrain(b, WIN)).reached
    b2 = FakeBrowser()
    b2.fail_screenshots = 10**6
    res = run_game(SPEC, cfg(tmp_path / "x"), lambda: b2, ScriptedBrain(b2, WIN), run_id="r2", clock=(c := Clock()), sleep=c.sleep)
    assert res.stop_reason == "unrecoverable_error"


def test_tool_allowlist_and_policy():
    node = make_tool_node(FakeBrowser(), lambda: 100.0, lambda s: None)
    ok, msg = dispatch(node, "upload_file", {"path": "/etc/passwd"})
    assert not ok  # not in the ToolNode
    p = Policy(("game.test",), (400, 300), 5)
    assert p.url_allowed("http://game.test/x") and not p.url_allowed("https://evil.test/") and not p.url_allowed("file:///etc/passwd")
    assert not p.check(act(coord=[1.5, 2])).ok and not p.check(act("press_key", key="Control+t")).ok
    assert p.check(act("wait", seconds=1)).ok and not p.check(act("wait", seconds=99)).ok
    assert not p.check(act("wait", seconds=1)).counts_as_action


class FakeClient:
    def __init__(self, replies):
        self.replies, self.models = list(replies), []
        self.messages = SimpleNamespace(create=self.create)

    def create(self, model, **kw):
        self.models.append(model)
        t = self.replies.pop(0)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=t)], usage=SimpleNamespace(input_tokens=100, output_tokens=10))


def ctx(np=0):
    return DecideContext("g", "play", b"png", {}, [], {}, {}, np)


def test_escalation_rules():
    good = json.dumps(act(coord=[1, 1], confidence="high"))
    low = json.dumps(act(coord=[1, 1], confidence="low"))
    c = Config()
    fc = FakeClient([good]); p = AnthropicBrain(c, fc).decide(ctx())
    assert fc.models == [c.cheap_model] and not p.escalated
    fc = FakeClient([low, good]); p = AnthropicBrain(c, fc).decide(ctx())
    assert fc.models == [c.cheap_model, c.strong_model] and p.escalated
    fc = FakeClient(["garbage", good]); AnthropicBrain(c, fc).decide(ctx())
    assert fc.models == [c.cheap_model, c.strong_model]
    fc = FakeClient([good]); AnthropicBrain(c, fc).decide(ctx(np=3))
    assert fc.models == [c.strong_model]  # no-progress streak goes straight to the strong model


def test_switch_rule():
    base = {"reached_rate": .5, "mean_actions": 50, "mean_cost_usd": .1, "mean_seconds": 100, "blocked_attempts": 1, "leaks": 0}
    assert should_switch(base, {**base, "reached_rate": .8})[0]
    assert not should_switch(base, {**base, "reached_rate": .8, "blocked_attempts": 2})[0]
    assert not should_switch(base, {**base, "reached_rate": .8, "mean_cost_usd": .2})[0]
    assert not should_switch(base, base)[0]
