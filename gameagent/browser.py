"""Browser layer: protocol the graph depends on, a settle helper, and the Selenium implementation.

Network-layer blocking: Chrome is started with a host-resolver rule that makes every host except the allowed
game hosts unresolvable (`MAP * ~NOTFOUND, EXCLUDE <host>`), so an allowed click that opens an external
site or follows a redirect cannot load it. The graph additionally inspects the URL after every action,
logs the block (or a leak if off-site content did load) and navigates back.
"""
from __future__ import annotations

import hashlib
import os
import time
from typing import Callable, Protocol


class Browser(Protocol):
    def screenshot(self) -> bytes: ...
    def click(self, x: int, y: int) -> None: ...
    def press_key(self, key: str) -> None: ...
    def current_url(self) -> str: ...
    def page_text(self) -> str: ...  # DOM signal, only used as extra evidence for success
    def restore(self) -> None: ...  # navigate back to the game page
    def close(self) -> None: ...


def digest(png: bytes) -> str:
    return hashlib.sha256(png).hexdigest()


def settle(browser: Browser, timeout: float, poll: float, sleep: Callable[[float], None] = time.sleep,
           clock: Callable[[], float] = time.monotonic) -> tuple[bytes, bool]:
    """Wait (bounded) until two consecutive screenshots are identical. Returns (last screenshot, settled)."""
    end = clock() + max(0.0, timeout)
    prev = browser.screenshot()
    while clock() < end:
        sleep(poll)
        cur = browser.screenshot()
        if digest(cur) == digest(prev):
            return cur, True
        prev = cur
    return prev, False


_KEYS = {"ArrowUp": "ARROW_UP", "ArrowDown": "ARROW_DOWN", "ArrowLeft": "ARROW_LEFT", "ArrowRight": "ARROW_RIGHT",
         "Space": "SPACE", "Enter": "ENTER", "Escape": "ESCAPE"}


class SeleniumBrowser:
    def __init__(self, start_url: str, allowed_hosts: tuple[str, ...], viewport: tuple[int, int] = (1024, 768),
                 headless: bool = True, binary: str | None = None):
        from selenium import webdriver

        o = webdriver.ChromeOptions()
        if headless:
            o.add_argument("--headless=new")
        binary = binary or os.environ.get("CHROME_BINARY")
        if binary:
            o.binary_location = binary
        o.add_argument("--no-sandbox")
        o.add_argument(f"--window-size={viewport[0]},{viewport[1]}")
        rules = "MAP * ~NOTFOUND" + "".join(f", EXCLUDE {h}" for h in allowed_hosts)
        o.add_argument(f"--host-resolver-rules={rules}")
        o.add_argument("--disable-popup-blocking=false")
        self.start_url = start_url
        self.d = webdriver.Chrome(options=o)
        self.d.set_window_size(*viewport)
        self.d.get(start_url)

    def screenshot(self) -> bytes:
        return self.d.get_screenshot_as_png()

    def click(self, x: int, y: int) -> None:
        from selenium.webdriver.common.actions.action_builder import ActionBuilder

        ab = ActionBuilder(self.d)
        ab.pointer_action.move_to_location(x, y)
        ab.pointer_action.click()
        ab.perform()

    def press_key(self, key: str) -> None:
        from selenium.webdriver import ActionChains
        from selenium.webdriver.common.keys import Keys

        ch = getattr(Keys, _KEYS[key]) if key in _KEYS else key
        ActionChains(self.d).send_keys(ch).perform()

    def current_url(self) -> str:
        return self.d.current_url

    def page_text(self) -> str:
        return self.d.execute_script("return document.body ? document.body.innerText : ''") or ""

    def restore(self) -> None:
        # A new window/tab opened by the game would be an extra handle: close extras, then go back.
        handles = self.d.window_handles
        for h in handles[1:]:
            self.d.switch_to.window(h)
            self.d.close()
        self.d.switch_to.window(handles[0])
        if self.d.current_url != self.start_url:
            self.d.back()

    def close(self) -> None:
        self.d.quit()
