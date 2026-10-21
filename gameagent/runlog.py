"""Durable per-step log (SQLite, one DB file per run directory root). Written during the run, committed per write.

Intent rows are written BEFORE the tool is dispatched; result columns are filled AFTER verification.
A row with ts_intent and no ts_result means "dispatched, outcome unknown" (section 5).
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(
  run_id TEXT PRIMARY KEY, game TEXT, started_at REAL, config_json TEXT, setup_json TEXT,
  status TEXT, stop_reason TEXT, ended_at REAL);
CREATE TABLE IF NOT EXISTS steps(
  run_id TEXT, step INTEGER,
  -- intent (written before execution)
  ts_intent REAL, elapsed_s REAL, subgoal TEXT, observation_ref TEXT, action_json TEXT,
  expected_outcome TEXT, counts_as_action INTEGER, model_used TEXT, learned TEXT,
  -- result (written after verification)
  ts_result REAL, result_ref TEXT, observed_result TEXT, classification TEXT, evidence TEXT, errors TEXT,
  verify_model TEXT, step_seconds REAL,
  PRIMARY KEY(run_id, step));
CREATE TABLE IF NOT EXISTS model_calls(
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, step INTEGER, role TEXT, model TEXT,
  tokens_in INTEGER, tokens_out INTEGER, cost_usd REAL, latency_s REAL, ts REAL);
-- kind: policy_block | network_block (stopped by the layer) | leak (off-site content actually loaded) |
--       safety_alert | runtime_error | model_error | escalation | resume | success_check | stop | setup
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, step INTEGER, ts REAL, elapsed_s REAL, kind TEXT, detail_json TEXT);
"""


class RunLog:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "runs.db")
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    def run_dir(self, run_id: str) -> Path:
        d = self.root / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _w(self, sql: str, args: tuple = ()) -> None:
        self.db.execute(sql, args)
        self.db.commit()

    def start_run(self, run_id: str, game: str, config: dict) -> None:
        self._w("INSERT INTO runs(run_id,game,started_at,config_json,status) VALUES(?,?,?,?,?)",
                (run_id, game, time.time(), json.dumps(config), "running"))

    def set_setup(self, run_id: str, setup: dict) -> None:
        self._w("UPDATE runs SET setup_json=? WHERE run_id=?", (json.dumps(setup), run_id))

    def finish_run(self, run_id: str, status: str, reason: str) -> None:
        self._w("UPDATE runs SET status=?, stop_reason=?, ended_at=? WHERE run_id=?",
                (status, reason, time.time(), run_id))

    def write_intent(self, run_id, step, elapsed, subgoal, obs_ref, action, expected, counts, model, learned) -> None:
        self._w(
            "INSERT INTO steps(run_id,step,ts_intent,elapsed_s,subgoal,observation_ref,action_json,expected_outcome,"
            "counts_as_action,model_used,learned) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, step, time.time(), elapsed, subgoal, obs_ref, json.dumps(action), expected, int(counts), model, learned),
        )

    def write_result(self, run_id, step, result_ref, observed, classification, evidence, errors, verify_model, seconds) -> None:
        self._w(
            "UPDATE steps SET ts_result=?, result_ref=?, observed_result=?, classification=?, evidence=?, errors=?,"
            "verify_model=?, step_seconds=? WHERE run_id=? AND step=?",
            (time.time(), result_ref, observed, classification, evidence, errors, verify_model, seconds, run_id, step),
        )

    def event(self, run_id: str, step: int, elapsed: float, kind: str, detail: dict) -> None:
        self._w("INSERT INTO events(run_id,step,ts,elapsed_s,kind,detail_json) VALUES(?,?,?,?,?,?)",
                (run_id, step, time.time(), elapsed, kind, json.dumps(detail, default=str)))

    def model_call(self, run_id, step, role, model, tin, tout, cost, latency) -> None:
        self._w("INSERT INTO model_calls(run_id,step,role,model,tokens_in,tokens_out,cost_usd,latency_s,ts) VALUES(?,?,?,?,?,?,?,?,?)",
                (run_id, step, role, model, tin, tout, cost, latency, time.time()))

    # ---- reads ----
    def run_row(self, run_id: str):
        return self.db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()

    def steps(self, run_id: str) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM steps WHERE run_id=? ORDER BY step", (run_id,)).fetchall()

    def events(self, run_id: str, kind: str | None = None) -> list[sqlite3.Row]:
        q, a = "SELECT * FROM events WHERE run_id=?", [run_id]
        if kind:
            q, a = q + " AND kind=?", a + [kind]
        return self.db.execute(q + " ORDER BY id", a).fetchall()

    def dangling_intents(self, run_id: str) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM steps WHERE run_id=? AND ts_result IS NULL ORDER BY step", (run_id,)).fetchall()

    def max_step(self, run_id: str) -> int:
        r = self.db.execute(
            "SELECT MAX(m) FROM (SELECT MAX(step) m FROM steps WHERE run_id=? UNION ALL "
            "SELECT MAX(step) FROM events WHERE run_id=? UNION ALL SELECT MAX(step) FROM model_calls WHERE run_id=?)",
            (run_id, run_id, run_id)).fetchone()
        return r[0] or 0

    def max_elapsed(self, run_id: str) -> float:
        r = self.db.execute(
            "SELECT MAX(m) FROM (SELECT MAX(elapsed_s) m FROM steps WHERE run_id=? UNION ALL "
            "SELECT MAX(elapsed_s) FROM events WHERE run_id=?)", (run_id, run_id)).fetchone()
        return r[0] or 0.0

    def actions_used(self, run_id: str) -> int:
        return self.db.execute(
            "SELECT COALESCE(SUM(counts_as_action),0) FROM steps WHERE run_id=?", (run_id,)).fetchone()[0]

    def totals(self, run_id: str) -> dict:
        r = self.db.execute(
            "SELECT COALESCE(SUM(cost_usd),0) c, COALESCE(SUM(tokens_in),0) ti, COALESCE(SUM(tokens_out),0) "
            "t_o, COUNT(*) n FROM model_calls WHERE run_id=?", (run_id,)).fetchone()
        return {"cost_usd": r["c"], "tokens_in": r["ti"], "tokens_out": r["t_o"], "model_calls": r["n"]}
