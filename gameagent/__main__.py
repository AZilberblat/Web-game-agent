"""CLI:  python -m gameagent run|report ...   (resume and eval are library calls: see README)"""
from __future__ import annotations

import argparse

from .brain import AnthropicBrain
from .browser import SeleniumBrowser
from .config import Config, GameSpec
from .report import report
from .run import run_game
from .runlog import RunLog


def main() -> None:
    p = argparse.ArgumentParser(prog="gameagent")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--url", required=True)
    r.add_argument("--cue", required=True, help="explicit visual cue that level 1 is complete")
    r.add_argument("--instructions")
    r.add_argument("--dom-text", help="optional text that must be in the page DOM at success")
    r.add_argument("--runs-dir", default="runs")
    r.add_argument("--headed", action="store_true")
    q = sub.add_parser("report")
    q.add_argument("run_id")
    q.add_argument("--runs-dir", default="runs")
    a = p.parse_args()
    if a.cmd == "report":
        print(report(RunLog(a.runs_dir), a.run_id))
        return
    cfg = Config(runs_dir=a.runs_dir)
    spec = GameSpec(a.name, a.url, a.cue, a.instructions, dom_success_text=a.dom_text)
    res = run_game(spec, cfg, lambda: SeleniumBrowser(spec.start_url, spec.hosts(), spec.viewport, not a.headed),
                   AnthropicBrain(cfg))
    print(res)
    print(report(RunLog(a.runs_dir), res.run_id))


main()
