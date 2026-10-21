"""Run configuration. Defaults are the brief's fixed constraints plus flagged simple defaults (section 10)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from urllib.parse import urlparse


@dataclass(frozen=True)
class GameSpec:
    """Per-game inputs the team supplies: where the game is and what 'level 1 done' looks like."""

    name: str
    start_url: str
    success_cue: str  # explicit visual cue for level 1 complete, in words
    instructions: str | None = None  # opening instructions; if None the agent reads them from the first screen
    allowed_hosts: tuple[str, ...] = ()  # defaults to the host of start_url
    dom_success_text: str | None = None  # optional extra DOM evidence: text that must be on the page at success
    viewport: tuple[int, int] = (1024, 768)

    def hosts(self) -> tuple[str, ...]:
        return self.allowed_hosts or (urlparse(self.start_url).hostname or "",)


@dataclass(frozen=True)
class Config:
    # --- fixed by the brief ---
    max_actions: int = 100
    max_seconds: float = 300.0
    # --- section 10 defaults (flagged; change with a reason) ---
    cheap_model: str = "claude-haiku-5-5"
    strong_model: str = "claude-sonnet-5-5"
    # "not good enough" = unparseable output, self-reported low confidence, or this many consecutive
    # steps without a success classification (then the strong model is used directly).
    escalate_after_no_progress: int = 3
    max_consecutive_failures: int = 6  # consecutive failure/unknown classifications -> stop
    max_blocked_attempts: int = 5  # policy rejections + network blocks, whole run -> safety stop
    max_runtime_errors: int = 3  # consecutive runtime errors -> stop
    max_iterations: int = 250  # backstop for iterations that spend no game action (waits, errors)
    settle_timeout: float = 3.0
    settle_poll: float = 0.4
    unchanged_extra_wait: float = 2.0  # one extra bounded wait when the screen did not change
    max_wait_action: float = 5.0
    history_recent: int = 5  # history slice: last N steps ...
    history_successes: int = 3  # ... plus the last M successful ones
    runs_dir: str = "runs"
    # USD per million tokens (input, output). PLACEHOLDERS: verify against current pricing before trusting cost.
    prices: dict = field(
        default_factory=lambda: {
            "claude-haiku-5-5": (1.0, 5.0),
            "claude-sonnet-5-5": (3.0, 15.0),
        }
    )

    def as_json(self) -> dict:
        return asdict(self)
