"""
Tests for the Test tab's team feedback: the six-gesture starter model as the
default (downloaded once, checked, saved as a float32 model folder) and a
maze move for every class of the loaded model.

Run:  python -m pytest tests/test_starter_model.py -v
"""

import functools
import hashlib
import http.server
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PyQt6 import QtWidgets

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

from core import starter_model as SM
import ui.test_tab as T


def test_every_class_has_a_default_move():
    assert [k for k, _ in T.MAZE_ACTIONS] == ["forward", "left", "right", "around", "none"]
    got = [T.default_maze_action(c) for c in
           ("idle", "push", "swipe_down", "swipe_left", "swipe_right", "swipe_up", "wave")]
    assert got == ["none", "forward", "around", "left", "right", "forward", "none"]


def test_the_maze_turns_around_and_obeys_any_move():
    maze = T.MazeWidget(3, 3)
    maze._walls = [[0] * 3 for _ in range(3)]
    assert maze.apply_action("around", "swipe_down").startswith("Turned around! Now facing West")
    assert maze.apply_action("around").startswith("Turned around! Now facing East")
    assert maze.apply_action("forward", "swipe_up").startswith("Moved East")
    assert maze.apply_action("none", "wave").startswith("Wave does nothing in the maze")
    assert maze.apply_gesture("idle").startswith("Idle")
    assert maze.apply_gesture("swipe_left").startswith("Turned left")


def test_chosen_moves_are_remembered_per_set_of_classes(monkeypatch):
    monkeypatch.setattr(T, "_MOVES_FILE", os.path.join(tempfile.mkdtemp(), "maze_moves.json"))
    classes = ["idle", "push", "swipe_up"]
    moves = T.load_maze_moves(classes)
    assert moves == {"idle": "none", "push": "forward", "swipe_up": "forward"}
    moves["swipe_up"] = "around"
    T.save_maze_moves(classes, moves)
    assert T.load_maze_moves(classes)["swipe_up"] == "around"
    assert T.load_maze_moves(["swipe_up"])["swipe_up"] == "forward", "another model starts from the defaults"


def _fake_site(tmp):
    """A tiny float16 model published the way the web version publishes the starter model."""
    from safetensors.numpy import save_file
    site = Path(tmp, "site")
    site.mkdir()
    w = {"a.weight": np.arange(12, dtype=np.float16).reshape(3, 4) / 4, "b.bias": np.array([-1.5, 2.0], np.float16)}
    save_file(w, str(site / "whole.safetensors"), metadata={"format": "pt"})
    blob = (site / "whole.safetensors").read_bytes()
    parts = []
    for i in range(0, len(blob), 100):
        chunk = blob[i:i + 100]
        name = f"model.safetensors.part{len(parts):02d}"
        (site / name).write_bytes(chunk)
        parts.append({"file": name, "bytes": len(chunk), "sha256": hashlib.sha256(chunk).hexdigest()})
    (site / "manifest.json").write_text(json.dumps({"id": "test", "weights": {
        "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest(), "parts": parts}}))
    (site / "config.json").write_text(json.dumps({"id2label": {"0": "idle", "1": "push"}}))
    (site / "preprocessor_config.json").write_text("{}")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site))
    handler.log_message = lambda *a: None
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, site, w


def test_the_starter_model_downloads_once_and_is_saved_as_float32(monkeypatch):
    from safetensors.numpy import load_file
    tmp = tempfile.mkdtemp()
    server, site, weights = _fake_site(tmp)
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        dest = os.path.join(tmp, "models", "starter_six_class")
        monkeypatch.setattr(SM, "STARTER_DIR", dest)
        monkeypatch.setattr(SM, "_bundled_dir", lambda: os.path.join(tmp, "none"))
        assert SM.find_starter() is None
        seen = []
        folder = SM.download_starter(lambda d, t: seen.append((d, t)), base_url=url, dest=dest)
        assert folder == dest and SM.find_starter() == dest
        assert seen and seen[-1][0] == seen[-1][1]
        got = load_file(os.path.join(dest, "model.safetensors"))
        for k, v in weights.items():
            assert got[k].dtype == np.float32 and np.array_equal(got[k], v.astype(np.float32))
        # a damaged part is refused and nothing is replaced
        (site / "model.safetensors.part00").write_bytes(b"x" * 100)
        try:
            SM.download_starter(base_url=url, dest=dest)
            raise AssertionError("a damaged download was accepted")
        except RuntimeError as e:
            assert "damaged" in str(e)
        assert np.array_equal(load_file(os.path.join(dest, "model.safetensors"))["b.bias"], [-1.5, 2.0])
    finally:
        server.shutdown()


def test_a_bundled_starter_model_is_used_first(monkeypatch):
    tmp = tempfile.mkdtemp()
    bundled = os.path.join(tmp, "bundle", "models", "starter_six_class")
    os.makedirs(bundled)
    for f in ("config.json", "preprocessor_config.json", "model.safetensors"):
        Path(bundled, f).write_text("x")
    monkeypatch.setattr(SM, "_bundled_dir", lambda: bundled)
    assert SM.find_starter() == bundled
