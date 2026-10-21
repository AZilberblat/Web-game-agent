"""Real Chrome + real SeleniumBrowser against games/tile_game.html with a scripted (non-model) brain."""
import http.server
import socketserver
import sys
import threading
from functools import partial
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from fakes import Clock, ScriptedBrain, act  # noqa: E402

from gameagent.config import Config, GameSpec  # noqa: E402
from gameagent.run import run_game  # noqa: E402
from gameagent.runlog import RunLog  # noqa: E402

ROOT = Path(__file__).parent.parent / "games"


def test_real_chrome_end_to_end(tmp_path):
    from gameagent.browser import SeleniumBrowser

    srv = socketserver.TCPServer(("127.0.0.1", 0), partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/tile_game.html"
    spec = GameSpec("tile", url, "LEVEL 1 COMPLETE heading", dom_success_text="LEVEL 1 COMPLETE", viewport=(800, 600))
    try:
        try:
            SeleniumBrowser(url, spec.hosts(), spec.viewport).close()
        except Exception as e:
            pytest.skip(f"no usable chrome/chromedriver (set CHROME_BINARY): {e}")

        brain = ScriptedBrain(None, [act(coord=[140, 140]), act(coord=[340, 240])])

        def verify(c):  # same as the fake brain but level_complete comes from the real DOM
            v = ScriptedBrain.verify(brain, c)
            v.level_complete = "LEVEL 1 COMPLETE" in brain.b.page_text()
            return v

        def factory():
            brain.b = SeleniumBrowser(url, spec.hosts(), spec.viewport)
            return brain.b

        brain.verify = verify
        brain.b = type("S", (), {"state": ""})()
        res = run_game(spec, Config(runs_dir=str(tmp_path)), factory, brain, run_id="chrome1")
        assert res.reached, RunLog(tmp_path).steps("chrome1")
        assert res.actions == 4
    finally:
        srv.shutdown()
