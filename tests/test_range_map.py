"""
Tests for the range map: the two scripts stay exact copies, the shared MTI and
distance settings, and the Visualize tab's range view.

Run:  python -m pytest tests/test_range_map.py -v
"""

import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PyQt6 import QtWidgets

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

from core import range_map as RM

# The scripts as written, never edited
SCRIPT_SHA256 = {
    "range_map_v1.py": "9f49f89d2d6d7a43416b8d36798e9c021b891c80c6a6e037923de0415029d806",
    "range_map_live.py": "f1f04fc30c3b595ba6c24bf789d29ac4501291eb1fbb5ad07d6b1283466c629b",
}


def _restore():
    RM.set_settings(mti=False, min_m=0.0, max_m=RM.RANGE_MAX_M)


def _pump(ms):
    end = time.time() + ms / 1000
    while time.time() < end:
        APP.processEvents()
        time.sleep(0.005)


def test_range_map_scripts_are_unmodified():
    for name, digest in SCRIPT_SHA256.items():
        source = (ROOT / "core" / name).read_bytes()
        assert hashlib.sha256(source).hexdigest() == digest, (
            f"core/{name} has been edited; it must stay an exact copy of the script")


def test_the_scripts_import_from_inside_core():
    v1, live = RM.import_range_map()
    assert live.frame_profile is v1.frame_profile
    r = RM.range_axis()
    assert r.size == 512 and r[0] == 0.0 and abs(r[-1] - 3.4809) < 1e-3


def test_settings_are_clamped_and_heard():
    heard = []
    fn = lambda s, changed: heard.append((s, changed))
    RM.on_settings_changed(fn)
    try:
        RM.set_settings(mti=True)
        assert RM.mode() == "mti" and heard[-1][1] == {"mti": True, "distance": False}
        RM.set_settings(min_m=2.0, max_m=1.0)        # To below From
        s = RM.get_settings()
        assert s["min"] == 2.0 and abs(s["max"] - 2.1) < 1e-9
        RM.set_settings(min_m=-1, max_m=99)
        s = RM.get_settings()
        assert s["min"] == 0.0 and s["max"] == RM.RANGE_MAX_M
        n = len(heard)
        RM.set_settings(min_m=0.0)                   # no change, nobody told
        assert len(heard) == n
    finally:
        RM.remove_listener(fn)
        _restore()


def test_shown_bins_and_colours_follow_the_distance():
    r = RM.range_axis()
    lo, hi = RM.shown_bins(r, 0.5, 2.0)
    assert r[lo] >= 0.5 > r[lo - 1] and r[hi] <= 2.0 < r[hi + 1]
    hist = np.full((10, 512), -60.0)
    hist[:, 20] = 0.0                                # a strong still reflector near 0.14 m
    hist[:, 200] = -30.0                             # the target near 1.36 m
    rgb = RM.range_rgb(hist, (lo, hi), -50.0)
    assert rgb.shape == (10, 512, 3)
    from matplotlib import colormaps
    top = np.array(colormaps["jet"](1.0, bytes=True)[:3])
    assert np.all(rgb[:, 200] == top), "the target is the brightest thing shown"


def test_visualize_switches_to_the_range_map():
    from ui.spectrogram_widget import VisualizeTab
    _restore()
    tab = VisualizeTab()
    tab.resize(1100, 700)
    tab.show()
    APP.processEvents()
    try:
        assert tab._range_opts.isHidden()
        tab._view_combo.setCurrentIndex(tab._view_combo.findData("range"))
        APP.processEvents()
        assert tab._range_view is not None and tab._range_view.is_ready
        assert tab._reference is None
        assert not tab._range_opts.isHidden() and tab._combo.isHidden()
        assert "range_map_live.py" in tab._readout.text()

        rng = np.random.default_rng(1)
        for _ in range(5):
            tab.on_raw_frame(rng.standard_normal((3, 128, 256)).astype(np.float32))
        _pump(600)
        assert tab._range_view._last is not None, "no frame was drawn"

        # The checkbox reaches the shared setting, and MTI rebuilds the view
        old = tab._range_view
        tab._mti_check.setChecked(True)
        APP.processEvents()
        assert RM.mode() == "mti" and tab._range_view is not old
        assert tab._range_view.mode == "mti"

        # A new distance rebuilds only the plot, with the bins inside it
        RM.set_settings(min_m=0.5, max_m=1.5)
        APP.processEvents()
        assert abs(tab._dist_from.value() - 0.5) < 1e-9
        lo, hi = tab._range_view._lo_hi
        r = RM.range_axis()
        assert r[lo] >= 0.5 and r[hi] <= 1.5
        assert "0.50 to 1.50 m" in tab._readout.text()

        tab._view_combo.setCurrentIndex(tab._view_combo.findData("doppler"))
        APP.processEvents()
        assert tab._range_view is None
    finally:
        tab.stop_if_running()
        _restore()
