"""
Tests for the Analysis tab's team feedback: velocity span, Doppler velocities
on the chirp spacing a sample was recorded with, and the selected sample drawn
the way the Visualize tab draws it (micro-Doppler and range map).

Run:  python -m pytest tests/test_sample_features.py -v
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PyQt6 import QtWidgets

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

from core import physical_features as PF

C = 299792458.0
SLOPE = 5.5e9 / 256e-6
LAM = C / 60.75e9


def _swing(amp=0.15, f=0.7, n_frame=20, chirp_s=0.0003, seed=0):
    """A target 0.8 m away swinging toward and away from the radar."""
    rng = np.random.default_rng(seed)
    s = np.arange(256)
    cube = np.zeros((n_frame, 3, 128, 256), np.float32)
    for i in range(n_frame):
        for c in range(128):
            t = i * 0.1 + c * chirp_s
            r = 0.8 + amp * np.sin(2 * np.pi * f * t)
            fb = 2 * SLOPE * r / C
            cube[i, :, c] = (0.5 + 0.02 * np.cos(2 * np.pi * fb * s / 1e6 + 4 * np.pi * r / LAM)
                             + 0.002 * rng.standard_normal(256))
    return cube


def _pump(ms):
    end = time.time() + ms / 1000
    while time.time() < end:
        APP.processEvents()
        time.sleep(0.005)


def test_velocity_span_is_exposed_and_measures_the_swing():
    assert "vel_span_ms" in PF.SUMMARY_KEYS and "vel_upper_ms" in PF.FRAME_KEYS
    assert PF.feature_label("vel_span_ms") == "Velocity span (m/s)"
    summary, series = PF.extract(_swing())
    true_span = 2 * 0.15 * 2 * np.pi * 0.7            # +/- peak radial velocity
    assert abs(summary["vel_span_ms"] - true_span) < 0.3, summary["vel_span_ms"]
    assert summary["vel_span_ms"] == np.nanmax(series["vel_upper_ms"]) - np.nanmin(series["vel_lower_ms"])
    still, _ = PF.extract(_swing(amp=0.0))
    assert still["vel_span_ms"] < 0.2, still["vel_span_ms"]
    noise = np.random.default_rng(4).standard_normal((20, 3, 128, 256)).astype(np.float32)
    assert PF.extract(noise)[0]["vel_span_ms"] == 0.0, "receiver noise has no signature"


def test_velocities_follow_the_chirp_spacing():
    assert PF.CHIRP_REP_S == 0.0003 and abs(PF.MAX_VELOCITY - 4.115) < 1e-3
    old = _swing(chirp_s=0.0002)
    right, _ = PF.extract(old, chirp_rep_s=0.0002)
    wrong, _ = PF.extract(old)                       # read as if 0.3 ms
    true_span = 2 * 0.15 * 2 * np.pi * 0.7
    assert abs(right["vel_span_ms"] - true_span) < 0.3
    assert abs(wrong["vel_span_ms"] / right["vel_span_ms"] - 2 / 3) < 0.05


def test_folders_say_which_chirp_they_were_recorded_with():
    d = tempfile.mkdtemp()
    assert PF.chirp_for_folder(d) == PF.CHIRP_REP_S
    Path(d, "capture_info.json").write_text(json.dumps({"chirp_repetition_time_s": 0.0002}))
    assert PF.chirp_for_folder(d) == 0.0002
    from ui.collect_tab import write_capture_info
    write_capture_info(d, "infineon", True)
    assert json.loads(Path(d, "capture_info.json").read_text())["chirp_repetition_time_s"] == PF.CHIRP_REP_S


def test_the_preview_is_drawn_by_the_visualize_plots():
    from core import sample_preview as SP
    from core import range_map as RM
    cube = _swing()
    pre = SP.compute(cube, mode="mti")
    assert pre["doppler"].shape == (20, 512) and pre["range_map"].shape == (20, 512)
    # The same numbers the Visualize tab's processor gives frame by frame
    ref = __import__("core.reference_image", fromlist=["x"]).import_reference()
    proc = ref.LiveDopplerProcessor(n_sample=256, n_chirp=128, history_length=20)
    for i in range(20):
        hist, _ = proc.process_frame(cube[i, 0])
    assert np.array_equal(pre["doppler"], hist)
    v1, _ = RM.import_range_map()
    want, _ = v1.compute_range_map(cube, antenna=0, mode="mti")
    assert np.array_equal(pre["range_map"], want)
    for render in (SP.render_doppler, SP.render_range):
        png = render(pre)
        assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 5000


def test_the_tab_shows_both_previews_for_a_clicked_sample():
    from ui.features_tab import FeaturesTab
    root = tempfile.mkdtemp()
    gdir = Path(root, "Alex", "push")
    gdir.mkdir(parents=True)
    np.save(gdir / "sample_001_raw.npy", _swing())
    tab = FeaturesTab()
    tab.show()
    record = {"student": "Alex", "gesture": "push", "name": "sample_001_raw.npy",
              "path": str(gdir / "sample_001_raw.npy"),
              "summary": {k: 0.0 for k in PF.SUMMARY_KEYS},
              "series": {"time_s": np.arange(20) * 0.1}}
    tab._records = [record]
    tab._show_record(record)
    for _ in range(400):
        _pump(20)
        if tab._preview_png.get("range"):
            break
    assert set(tab._preview_png) == {"doppler", "range"}
    for label in tab._preview_labels.values():
        assert not label.pixmap().isNull()
    tab.hide()
