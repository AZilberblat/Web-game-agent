# gameagent

LangGraph agent that tries to complete level 1 of an unfamiliar turn-based browser game from screenshots + click/key tools,
with code-enforced budget and permissions and a durable per-step log. Implements sections 2–9 of the brief.

```
pip install -e '.[dev]'
export ANTHROPIC_API_KEY=...            # CHROME_BINARY=/path/to/chrome if Chrome is not auto-detected
python -m gameagent run --name tiles --url http://localhost:8000/tile_game.html --cue "heading LEVEL 1 COMPLETE" --dom-text "LEVEL 1 COMPLETE"
python -m gameagent report <run_id>      # what the agent did, why it stopped, where policy blocked it
pytest                                   # fake game + scripted brain, no Chrome/API needed
```

## Layout
| file | role |
|---|---|
| `graph.py` | LangGraph loop: setup (once) → guard → capture → decide → gate → intent → execute → verify → record |
| `policy.py` | single registry of allowed tools/params + pre-execution gate + URL allowlist |
| `tools.py` | the ToolNode (the allowlist at execution time); refuses to build with a tool lacking a policy entry |
| `browser.py` | Selenium browser, settle-wait, network-layer block (Chrome `--host-resolver-rules`) |
| `brain.py` | cheap-first / escalate model layer (Anthropic SDK called from inside graph nodes) |
| `runlog.py` | SQLite log: `steps` (intent before dispatch, result after), `model_calls`, `events` |
| `run.py` | `run_game`, `resume_run` (crash semantics) |
| `report.py`, `evaluate.py` | per-run report; fresh/repeated/interleaved version comparison + switch rule |

## How the brief's rules map to code
- **Budget guard** runs in `guard` before every iteration *and* again in `gate` after the model call (model latency counts). Only game actions (`press_mouse`, `press_key`) count; `wait` does not.
- **Intent/result**: `intent` writes the row before `execute`; `verify` fills the result columns. A row with no result = outcome unknown.
- **Resume** (`resume_run`): dangling intents are closed as `unknown`, still count toward the 100 actions, are never re-dispatched; the agent takes a fresh screenshot and is told that outcome is unknown.
- **Unchanged screen ≠ failure**: bounded settle, one extra bounded wait, and a `failure` verdict with no pixel change is downgraded to `unknown`.
- **Success** needs: verifier claims the cue → strong model re-checks → DOM text (`GameSpec.dom_success_text`) if configured. Model and screenshot are not independent, so configure a DOM signal where one exists.
- **Safety**: model output is only a proposal; `Policy.check` rejects unknown tools/extra params/out-of-viewport coords/key combos; the ToolNode can only run the 3 tools; Chrome cannot resolve non-game hosts, and the URL is checked after every action (`network_block` = stopped, `leak` = off-site content loaded). Blocked attempts and leaks are separate event kinds. `safety_alert` from the model is logged only; nothing depends on it.

## Open choices (section 10) — defaults picked, change with a reason
- Models: `claude-haiku-5-5` first, `claude-sonnet-5-5` on unparseable output, self-reported `low` confidence, or ≥3 consecutive non-success steps (then strong directly). Prices in `Config.prices` are **placeholders**.
- Log: SQLite + PNG files under `runs/<run_id>/`.
- Webdriver: Selenium; `press_mouse` coords are **screenshot pixels** (the brief's `[5,3]` example reads like grid cells; if a game needs cell coords the model maps them itself).
- History slice: last 5 steps + last 3 successes, plus a compact tried-action→outcome-count table and ≤10 learned facts.
- Thresholds: 6 consecutive failure/unknown → stop; 5 blocked attempts per run → safety stop; 3 consecutive runtime errors → stop; 250-iteration backstop for iterations that spend no action.
- Out of scope: memory across games, parallel runs.

## Known limitations / unverified
- **`SeleniumBrowser` has not been run against real Chrome here** (container had chromedriver 147 vs Chrome 141, no download access). `tests/test_selenium_integration.py` is skipped when no matching driver exists; run it where one does. The `AnthropicBrain` prompts have only been tested with a stubbed client.
- Resume via a *new* browser loses game state (the agent re-observes and adapts); the library call accepts a live browser.
- Verification after the final in-budget action is skipped if the 5-minute budget is already spent, so a win on the very last moment can go undetected.
