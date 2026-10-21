"""Version comparison (section 9): every version is run FRESH, repeated, on the same fixed game set, with the
order interleaved per repetition so time-of-day / site conditions hit all versions alike."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from statistics import mean
from typing import Callable

from .config import GameSpec
from .run import RunResult


@dataclass(frozen=True)
class Version:
    name: str
    cheap_model: str = "claude-haiku-5-5"
    strong_model: str = "claude-sonnet-5-5"
    prompt: str = "default"  # prompt/output-format variant label; the runner decides what it means


Runner = Callable[[GameSpec, Version, int], RunResult]


def run_matrix(games: list[GameSpec], versions: list[Version], repeats: int, runner: Runner,
               seed: int = 0) -> dict[str, list[RunResult]]:
    rng = random.Random(seed)
    out: dict[str, list[RunResult]] = {v.name: [] for v in versions}
    for rep in range(repeats):
        jobs = [(g, v) for g in games for v in versions]
        rng.shuffle(jobs)
        for g, v in jobs:
            out[v.name].append(runner(g, v, rep))
    return out


def metrics(rs: list[RunResult]) -> dict:
    n = len(rs)
    return {"runs": n, "reached_rate": sum(r.reached for r in rs) / n,
            "mean_actions": mean(r.actions for r in rs), "mean_cost_usd": mean(r.cost_usd for r in rs),
            "mean_seconds": mean(r.seconds for r in rs),
            "blocked_attempts": sum(r.blocked_attempts for r in rs), "leaks": sum(r.leaks for r in rs)}


def should_switch(base: dict, new: dict) -> tuple[bool, list[str]]:
    """Switch only if the new version is better on the measures with no safety regression.
    'Better' = not worse on reach rate, actions, cost, time, and strictly better on at least one."""
    why: list[str] = []
    if new["leaks"] > base["leaks"] or new["blocked_attempts"] > base["blocked_attempts"]:
        why.append("safety regression (blocked attempts or leaks increased)")
    for k, lower_better in (("reached_rate", False), ("mean_actions", True), ("mean_cost_usd", True), ("mean_seconds", True)):
        if (new[k] > base[k]) if lower_better else (new[k] < base[k]):
            why.append(f"worse on {k}: {base[k]:.3g} -> {new[k]:.3g}")
    better = any((new[k] < base[k]) if lb else (new[k] > base[k]) for k, lb in
                 (("reached_rate", False), ("mean_actions", True), ("mean_cost_usd", True), ("mean_seconds", True)))
    if not better and not why:
        why.append("no measurable improvement")
    return (not why), why
