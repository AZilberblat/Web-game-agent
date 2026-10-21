"""A fake game (rendered with PIL), fake browser and scripted brain. No Chrome / API needed."""
from __future__ import annotations

import io

from PIL import Image, ImageDraw

from gameagent.brain import Proposal, SetupInfo, Usage, Verdict
from gameagent.config import GameSpec

SPEC = GameSpec("fake", "http://game.test/", "a banner saying LEVEL 1 COMPLETE", dom_success_text="LEVEL 1 COMPLETE",
                viewport=(400, 300))
START_BTN, TARGET, LINK = (200, 150), (100, 100), (350, 280)


class CrashNow(BaseException):
    """Simulates the process dying mid-dispatch (not caught by the runtime-error handling)."""


class FakeBrowser:
    def __init__(self, hits_needed: int = 2, lag_frames: int = 0, dom_text_on_complete: bool = True):
        self.state, self.hits, self.need, self.lag = "start", 0, hits_needed, lag_frames
        self.pending: list[tuple[int, str]] = []  # (frames_left, new_state)
        self.url, self.clicks, self.keys, self.shots = "http://game.test/", [], [], 0
        self.crash_on_click: int | None = None
        self.dom_text_on_complete = dom_text_on_complete
        self.fail_screenshots = 0

    def screenshot(self) -> bytes:
        if self.fail_screenshots:
            self.fail_screenshots -= 1
            raise RuntimeError("screenshot failed")
        self.shots += 1
        self.pending = [(n - 1, s) for n, s in self.pending]
        for n, s in [p for p in self.pending if p[0] <= 0]:
            self.state = s
        self.pending = [p for p in self.pending if p[0] > 0]
        im = Image.new("RGB", (400, 300), "white")
        d = ImageDraw.Draw(im)
        d.text((10, 10), f"{self.state} hits={self.hits}", fill="black")
        if self.state == "complete":
            d.text((150, 140), "LEVEL 1 COMPLETE", fill="green")
        buf = io.BytesIO()
        im.save(buf, "PNG")
        return buf.getvalue()

    def _set(self, state: str) -> None:
        if self.lag:
            self.pending.append((self.lag, state))
        else:
            self.state = state

    def click(self, x: int, y: int) -> None:
        self.clicks.append((x, y))
        if self.crash_on_click is not None and len(self.clicks) == self.crash_on_click:
            raise CrashNow()
        near = lambda p: abs(x - p[0]) < 15 and abs(y - p[1]) < 15  # noqa: E731
        if self.state == "start" and near(START_BTN):
            self._set("board")
        elif self.state == "board" and near(TARGET):
            self.hits += 1
            if self.hits >= self.need:
                self._set("complete")
        elif self.state == "board" and near(LINK):
            self.url = "chrome-error://chromewebdata/"

    def press_key(self, key: str) -> None:
        self.keys.append(key)

    def current_url(self) -> str:
        return self.url

    def page_text(self) -> str:
        return "LEVEL 1 COMPLETE" if self.state == "complete" and self.dom_text_on_complete else "playing"

    def restore(self) -> None:
        self.url = "http://game.test/"

    def close(self) -> None:
        pass


def act(tool="press_mouse", expected="something changes", **kw) -> dict:
    return {"action": tool, "expected_outcome": expected, **kw}


class ScriptedBrain:
    """decide() plays a list of proposals (last one repeats). verify() = pixel change -> success else unknown."""

    def __init__(self, browser: FakeBrowser, script: list, clock=None, latency: float = 0.0, verdict_override=None):
        self.b, self.script, self.i, self.clock, self.latency = browser, script, 0, clock, latency
        self.verdict_override = verdict_override
        self.decide_contexts: list = []
        self.confirm_result = True

    def setup(self, shot, spec):
        return SetupInfo("click start, then hit the target", act(coord=list(START_BTN), expected="board appears")), []

    def decide(self, ctx):
        self.decide_contexts.append(ctx)
        if self.clock:
            self.clock.t += self.latency
        p = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return Proposal(p, "scripted", [Usage("decide", "scripted", 10, 5, 0.001, self.latency)])

    def verify(self, c):
        if self.verdict_override:
            v = self.verdict_override
            return Verdict(v, "", "override", False, [Usage("verify", "scripted", 1, 1, 0.0005)], "scripted")
        done = self.b.state == "complete"
        cls = "success" if c.screen_changed else "unknown"
        return Verdict(cls, f"state={self.b.state}", "pixel diff", done, [Usage("verify", "scripted", 1, 1, 0.0005)], "scripted")

    def confirm_success(self, shot, spec):
        return self.confirm_result, "banner visible", [Usage("confirm", "scripted", 1, 1, 0.001)]


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


WIN = [act(coord=list(START_BTN), expected="board appears")] + [act(coord=list(TARGET), expected="hit counter increases")]
