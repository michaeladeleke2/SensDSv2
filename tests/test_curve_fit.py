"""
Tests for tracing a curve over a spectrogram and fitting a model to it.

The Curve Fit tab exists so a student can check a fitted equation against
something they can measure, a pendulum they can put a ruler to. That only
works if the fit recovers the motion that was traced, so these tests pin the
cleanup of freehand tracing, the snap onto the signal, each model recovering
known parameters, and failure coming back as a message rather than a crash.

Run:  python -m pytest tests/test_curve_fit.py -v
"""

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PyQt6 import QtCore, QtGui, QtWidgets

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

from core import curve_fit as CF


# ── trace cleanup ────────────────────────────────────────────────────────────

def test_clean_trace_sorts_by_time_and_averages_repeats():
    t, v = CF.clean_trace([(1.0, 2.0), (0.0, 1.0), (1.0, 4.0)])
    assert np.allclose(t, [0.0, 1.0])
    assert np.allclose(v, [1.0, 3.0])          # the two at t=1 averaged


def test_clean_trace_collapses_near_duplicate_times():
    """Freehand tracing lands many samples on almost the same instant."""
    pts = [(0.0, 0.0), (1.0, 2.0), (1.0 + 1e-9, 4.0), (2.0, 1.0)]
    t, v = CF.clean_trace(pts)
    assert t.size == 3
    assert np.isclose(v[1], 3.0)


def test_clean_trace_handles_empty_and_single():
    t, v = CF.clean_trace([])
    assert t.size == 0 and v.size == 0
    t, v = CF.clean_trace([(1.0, 2.0)])
    assert np.allclose(t, [1.0]) and np.allclose(v, [2.0])


# ── snapping ─────────────────────────────────────────────────────────────────

def _ridge_spectrogram(n_bins=100, n_cols=50, peak_row=60):
    spec = np.full((n_bins, n_cols), -20.0)
    spec[peak_row, :] = 0.0
    return spec


def test_snap_to_peak_pulls_points_onto_the_ridge():
    spec = _ridge_spectrogram()
    vel_scale, time_scale = 0.1, 0.1
    # Traced five bins below the ridge, as a wobbling hand would.
    traced = [(c * time_scale, (55 - 50) * vel_scale) for c in range(5)]
    snapped = CF.snap_to_peak(traced, spec, time_scale, vel_scale,
                              window_bins=20)
    for (_, v) in snapped:
        assert np.isclose(v, (60 - 50) * vel_scale)


def test_snap_to_peak_leaves_a_point_alone_when_the_ridge_is_out_of_reach():
    """A tie inside a flat window must not drag the point to the window edge."""
    spec = _ridge_spectrogram()
    vel_scale, time_scale = 0.1, 0.1
    traced = [(0.0, (20 - 50) * vel_scale)]     # 40 bins from the ridge
    snapped = CF.snap_to_peak(traced, spec, time_scale, vel_scale,
                              window_bins=5)
    assert np.isclose(snapped[0][1], traced[0][1])


def test_snap_to_peak_survives_an_empty_spectrogram():
    traced = [(0.0, 1.0), (0.1, 1.1)]
    assert CF.snap_to_peak(traced, np.empty((0, 0)), 0.1, 0.1) == traced


# ── models ───────────────────────────────────────────────────────────────────

def _sine_points(A=1.2, f=0.5, phi=0.7, c=0.1, noise=0.01, n=120, span=5.0):
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, span, n)
    v = CF.sinusoid(t, A, f, phi, c) + rng.normal(0.0, noise, n)
    return list(zip(t, v))


def test_sinusoid_recovers_the_motion_that_was_traced():
    result = CF.fit_curve(_sine_points(), CF.MODEL_SINUSOID)
    assert result.ok, result.message
    A, f, phi, c = result.params
    assert np.isclose(abs(A), 1.2, atol=0.05)
    assert np.isclose(f, 0.5, atol=0.01)
    assert np.isclose(c, 0.1, atol=0.05)
    assert result.r_squared > 0.99
    assert "sin" in result.equation


def test_damped_sinusoid_recovers_the_decay():
    t = np.linspace(0.0, 6.0, 200)
    v = CF.damped_sinusoid(t, 1.5, 2.0, 0.8, 0.0, 0.0)
    result = CF.fit_curve(list(zip(t, v)), CF.MODEL_DAMPED)
    assert result.ok, result.message
    A, tau, f, phi, c = result.params
    assert np.isclose(tau, 2.0, rtol=0.1)
    assert np.isclose(f, 0.8, atol=0.01)
    assert result.r_squared > 0.99


def test_polynomial_and_linear_recover_their_coefficients():
    t = np.linspace(0.0, 3.0, 40)
    quad = CF.fit_curve(list(zip(t, 2.0 * t ** 2 - 3.0 * t + 1.0)),
                        CF.MODEL_POLYNOMIAL, degree=2)
    assert quad.ok, quad.message
    assert np.allclose(quad.params, [2.0, -3.0, 1.0], atol=1e-6)
    assert quad.r_squared > 0.999

    line = CF.fit_curve(list(zip(t, 4.0 * t - 2.0)), CF.MODEL_LINEAR)
    assert line.ok, line.message
    assert np.allclose(line.params, [4.0, -2.0], atol=1e-6)
    assert "t" in line.equation


def test_a_fit_that_cannot_converge_comes_back_as_a_message():
    """curve_fit raises on bad input; the tab must not take the app down."""
    bad = [(float(i), float("nan")) for i in range(20)]
    result = CF.fit_curve(bad, CF.MODEL_SINUSOID)
    assert not result.ok
    assert "did not converge" in result.message


def test_too_few_points_is_refused_by_name():
    result = CF.fit_curve([(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)],
                          CF.MODEL_SINUSOID)
    assert not result.ok
    assert "at least 4" in result.message


# ── physics readout ──────────────────────────────────────────────────────────

def test_pendulum_readout_reports_period_and_length():
    text = CF.pendulum_readout(0.5)
    expected = CF.G / (4 * np.pi ** 2 * 0.25)     # 0.994 m
    assert "0.5 Hz" in text
    assert "2 s" in text
    assert f"{expected:.3g}" in text
    assert "simple pendulum" in text


def test_physics_is_only_shown_for_the_sinusoid_models():
    sine = CF.fit_curve(_sine_points(), CF.MODEL_SINUSOID)
    assert "Predicted length" in sine.physics
    t = np.linspace(0.0, 3.0, 40)
    poly = CF.fit_curve(list(zip(t, t ** 2)), CF.MODEL_POLYNOMIAL, degree=2)
    assert poly.physics == ""


def test_numbers_are_rounded_to_three_significant_figures():
    assert CF._g(1.23456) == "1.23"
    assert CF._g(0.000123456) == "0.000123"


# ── the tab ──────────────────────────────────────────────────────────────────

def _batch(tab, value):
    return np.full((tab._freq_bins, 4), value, dtype=np.float32)


def test_freezing_stops_the_buffer_and_resuming_starts_it():
    from ui.curve_fit_tab import CurveFitTab
    tab = CurveFitTab()
    tab.on_spectrogram_frame(_batch(tab, -5.0))
    before = tab._display_array().copy()

    tab._freeze()
    tab.on_spectrogram_frame(_batch(tab, -1.0))
    assert np.array_equal(tab._display_array(), before), "frozen buffer moved"

    tab._resume()
    tab.on_spectrogram_frame(_batch(tab, -1.0))
    assert not np.array_equal(tab._display_array(), before)


def test_fit_button_waits_for_ten_points():
    from ui.curve_fit_tab import MIN_TRACE_POINTS, CurveFitTab
    tab = CurveFitTab()
    assert not tab._fit_btn.isEnabled()
    tab._points = [(0.1 * i, 0.0) for i in range(MIN_TRACE_POINTS - 1)]
    tab._sync_buttons()
    assert not tab._fit_btn.isEnabled()
    tab._points.append((1.0, 0.0))
    tab._sync_buttons()
    assert tab._fit_btn.isEnabled()


def test_dragging_on_the_plot_traces_a_path():
    from ui.curve_fit_tab import CurveFitTab
    tab = CurveFitTab()
    tab.resize(900, 700)
    tab.show()
    APP.processEvents()
    tab._trace_btn.setChecked(True)
    assert tab.is_frozen(), "tracing should freeze the display first"

    viewport = tab._plot_widget.viewport()
    kinds = [QtCore.QEvent.Type.MouseButtonPress,
             QtCore.QEvent.Type.MouseMove,
             QtCore.QEvent.Type.MouseMove,
             QtCore.QEvent.Type.MouseButtonRelease]
    for i, kind in enumerate(kinds):
        pos = QtCore.QPointF(120.0 + 20 * i, 200.0 + 5 * i)
        buttons = (QtCore.Qt.MouseButton.LeftButton
                   if kind != QtCore.QEvent.Type.MouseButtonRelease
                   else QtCore.Qt.MouseButton.NoButton)
        APP.sendEvent(viewport, QtGui.QMouseEvent(
            kind, pos, pos, QtCore.Qt.MouseButton.LeftButton, buttons,
            QtCore.Qt.KeyboardModifier.NoModifier))

    assert len(tab.trace_points()) == 4
    xs, _ = tab._trace_curve.getData()
    assert len(xs) == 4
    tab._clear_trace()
    assert tab.trace_points() == []
    tab.hide()


def test_polynomial_degree_only_shows_for_polynomial():
    from ui.curve_fit_tab import CurveFitTab
    tab = CurveFitTab()
    tab.show()
    APP.processEvents()
    assert tab._degree_row.isHidden()
    for btn in tab._model_group.buttons():
        if btn.property("model_key") == CF.MODEL_POLYNOMIAL:
            btn.setChecked(True)
    assert not tab._degree_row.isHidden()
    tab.hide()


def test_a_failed_fit_shows_a_message_instead_of_raising():
    from ui.curve_fit_tab import CurveFitTab
    tab = CurveFitTab()
    tab._points = [(float(i), float("nan")) for i in range(20)]
    tab._sync_buttons()
    tab._on_fit()                       # must not raise
    assert "did not converge" in tab._status.text()


# ── wiring ───────────────────────────────────────────────────────────────────

class _FakeBridge(QtCore.QObject):
    """Enough of RadarBridge for _on_connect, with no radar attached."""
    frame_ready = QtCore.pyqtSignal(np.ndarray)
    raw_frame_ready = QtCore.pyqtSignal(np.ndarray)
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.started = []
        self.processor = type("P", (), {"reset": lambda self: None})()

    def start_stream(self, with_display=False):
        self.started.append(with_display)

    def stop_stream(self):
        self.started.append("stop")


def test_main_window_places_curve_fit_after_analysis_and_feeds_it():
    from ui import main_window as MW
    win = MW.MainWindow()
    try:
        tabs = win._tabs
        idx = tabs.indexOf(win._curve_fit_tab)
        assert idx == tabs.indexOf(win._analysis_tab) + 1
        assert "Curve Fit" in tabs.tabText(idx)

        real, MW.RadarBridge = MW.RadarBridge, _FakeBridge
        try:
            win._on_connect()
        finally:
            MW.RadarBridge = real

        # The tab keeps its own buffer, fed by frame_ready.
        before = win._curve_fit_tab._display_array().copy()
        win._bridge.frame_ready.emit(
            np.full((win._curve_fit_tab._freq_bins, 4), -3.0, dtype=np.float32))
        assert not np.array_equal(win._curve_fit_tab._display_array(), before)

        # Opening the tab needs the display stream running.
        win._bridge.started.clear()
        win._apply_stream_for_tab(idx)
        assert win._bridge.started == [True]
    finally:
        win._visualize_tab.stop_if_running()


def _raw_frames(n=12, seed=5):
    """Frames shaped like the radar's, so the Doppler axis comes out at 512."""
    rng = np.random.default_rng(seed)
    frames = rng.standard_normal((n, 3, 128, 64)).astype(np.float32)
    s = np.arange(64)[None, None, None, :]
    c = np.arange(128)[None, None, :, None]
    t = np.arange(n)[:, None, None, None]
    frames += 6.0 * np.cos(2 * np.pi * (0.1 * s + 0.01 * (1 + t) * c))
    return frames


def test_freezing_draws_the_reference_image_for_infineon():
    from core import processing as P
    from core import reference_image as R
    from ui.curve_fit_tab import CurveFitTab
    P.set_method(P.METHOD_INFINEON)
    try:
        tab = CurveFitTab()
        frames = _raw_frames()
        for frame in frames:
            tab.on_raw_frame(frame)
        tab._freeze()
        assert tab._reference_frozen, tab._status.text()

        img = tab._img.image
        assert img.ndim == 3 and img.shape[2] == 3, "not an RGB image"
        assert img.shape[0] == tab._width, "one column per frame expected"

        spec = R.reference_spectrogram(frames)
        pad = np.full((tab._width - spec.shape[0], spec.shape[1]),
                      float(spec.min()))
        spec = np.vstack([pad, spec])
        assert np.array_equal(img, R.reference_rgb(spec)[::-1].transpose(1, 0, 2))
        # Snapping reads this array by velocity, so it has to carry the same
        # flip as the picture or points snap to its mirror image.
        assert np.array_equal(tab._frozen_spec, spec.T[::-1])
    finally:
        P.set_method(P.METHOD_STFT)


def test_stft_freeze_falls_back_to_the_app_drawing():
    from core import processing as P
    from ui.curve_fit_tab import CurveFitTab
    P.set_method(P.METHOD_STFT)
    tab = CurveFitTab()
    for frame in _raw_frames(4):
        tab.on_raw_frame(frame)
    tab._freeze()
    assert not tab._reference_frozen
    assert tab._img.image.ndim == 2
    assert "app view" in tab._spec_heading.text()


def test_resume_returns_to_the_live_drawing():
    from core import processing as P
    from ui.curve_fit_tab import CurveFitTab
    P.set_method(P.METHOD_INFINEON)
    try:
        tab = CurveFitTab()
        for frame in _raw_frames():
            tab.on_raw_frame(frame)
        tab._freeze()
        assert tab._img.image.ndim == 3
        tab._resume()
        assert tab._img.image.ndim == 2, "live view should be the app drawing"
        assert not tab._reference_frozen
        assert tab._spec_heading.text() == "Live Spectrogram"
    finally:
        P.set_method(P.METHOD_STFT)


def test_raw_frames_stop_arriving_while_frozen():
    from ui.curve_fit_tab import CurveFitTab
    tab = CurveFitTab()
    for frame in _raw_frames(4):
        tab.on_raw_frame(frame)
    held = len(tab._raw_frames)
    tab._freeze()
    tab.on_raw_frame(_raw_frames(1)[0])
    assert len(tab._raw_frames) == held


# ── team feedback: exponential, Auto Fit, the range map, the student panel ──

def test_exponential_recovers_decay_and_growth():
    for A, k, c in ((0.8, -0.5, 0.1), (0.05, 0.3, -0.2), (1.5, -0.8, 1.0)):
        t = 2 + 0.08 * np.arange(60)
        v = A * np.exp(k * t) + c + 0.002 * np.sin(41 * t)
        r = CF.fit_curve(list(zip(t, v)), CF.MODEL_EXP)
        assert r.ok, r.message
        A2, k2, c2 = r.params
        assert abs(A2 - A) < 0.03 * max(1, abs(A)) and abs(k2 - k) < 0.03 and abs(c2 - c) < 0.03
        assert r.r_squared > 0.99
        assert ("Half-life" if k < 0 else "Doubling time") in r.physics


def test_range_quantity_writes_r_of_t_in_metres():
    t = 0.1 * np.arange(50)
    v = 1.2 + 0.1 * np.sin(2 * np.pi * 0.6 * t + 0.4)
    r = CF.fit_curve(list(zip(t, v)), CF.MODEL_SINUSOID, quantity="range")
    assert r.ok and r.equation.startswith("r(t) = ") and r.param_text.endswith("c = 1.2 m")
    assert abs(r.params[1] - 0.6) < 1e-6


def _ridge(fn, rows=512, cols=80, ts=0.1, scale=2 * 4.112380411522634 / 512, offset=256, gaps=()):
    rng = np.random.default_rng(3)
    spec = -80 + 4 * rng.standard_normal((rows, cols))
    for c in range(cols):
        if c in gaps:
            continue
        row = int(round(fn((c + 0.5) * ts) / scale + offset))
        for d in range(-2, 3):
            if 0 <= row + d < rows:
                spec[row + d, c] = -20 - 3 * abs(d)
    return spec


def test_auto_trace_follows_the_ridge_and_skips_empty_columns():
    fn = lambda t: 1.5 * np.sin(2 * np.pi * 0.5 * t)
    scale = 2 * 4.112380411522634 / 512
    pts = CF.auto_trace(_ridge(fn, gaps=(10, 11, 12)), 0.1, scale, row_offset=256,
                        y_range=(-3, 3), dead_zone=0.1)
    assert 60 <= len(pts) <= 77
    assert not any(abs(t - 1.15) < 0.01 for t, _ in pts)
    for t, v in pts:
        assert abs(v - fn(t)) < 0.05 or abs(fn(t)) < 0.15
    r = CF.fit_curve(pts, CF.MODEL_SINUSOID)
    assert abs(r.params[1] - 0.5) < 0.01 and r.r_squared > 0.98


def test_auto_trace_on_a_range_map_keeps_to_the_distance_shown():
    step = 0.0068
    fn = lambda t: 1.0 + 0.15 * np.sin(2 * np.pi * 0.7 * t)
    spec = _ridge(fn, scale=step, offset=0)
    spec[int(round(0.3 / step)), :] = -5.0          # a stronger still reflector
    pts = CF.auto_trace(spec, 0.1, step, row_offset=0, y_range=(0.6, 2.0))
    assert len(pts) > 70 and all(abs(v - fn(t)) < 0.02 for t, v in pts)
    near = CF.auto_trace(spec, 0.1, step, row_offset=0, y_range=(0.0, 3.48))
    assert all(abs(v - 0.3) < 0.01 for _, v in near)


def test_snap_to_peak_on_a_range_axis():
    spec = _ridge(lambda t: 1.0, rows=300, cols=20, scale=0.01, offset=0)
    out = CF.snap_to_peak([(0.55, 1.08), (1.05, 0.93)], spec, 0.1, 0.01, row_offset=0)
    assert all(abs(v - 1.0) < 1e-9 for _, v in out)


def test_peaks_give_the_swing_and_the_decay():
    f, A, c, phi = 0.8, 0.6, 0.2, 0.9
    t = 0.05 * np.arange(120)
    v = A * np.sin(2 * np.pi * f * t + phi) + c + 0.02 * np.sin(37 * t)
    peaks = CF.find_peaks(t, v)
    assert len(peaks["positive"]) >= 4 and len(peaks["negative"]) >= 4
    signs = [p[2] for p in peaks["all"]]
    assert all(a != b for a, b in zip(signs, signs[1:])), "peaks alternate"
    s = CF.peak_start(CF.MODEL_SINUSOID, peaks)
    assert abs(s[1] - f) < 0.05 and abs(s[0] - A) < 0.08 and abs(s[3] - c) < 0.05
    vd = A * np.exp(-t / 2.5) * np.sin(2 * np.pi * f * t + phi)
    sd = CF.peak_start(CF.MODEL_DAMPED, CF.find_peaks(t, vd))
    assert abs(sd[1] - 2.5) < 0.4
    assert CF.peak_start(CF.MODEL_EXP, peaks) is None


def test_every_letter_is_explained():
    from core.curve_explain import explain_fit, explain_peaks, fit_quality, g
    t = 0.05 * np.arange(80)
    wave = lambda tau: 0.5 * np.exp(-t / tau) * np.sin(2 * np.pi * 0.9 * t + 0.3) + 0.05
    cases = [(CF.MODEL_SINUSOID, wave(1e9), ["A", "f", "φ", "c"]),
             (CF.MODEL_DAMPED, wave(3), ["A", "τ", "f", "φ", "c"]),
             (CF.MODEL_EXP, 0.5 * np.exp(-t / 1.5) + 0.05, ["A", "k", "c"]),
             (CF.MODEL_LINEAR, 0.3 * t + 0.1, ["a1", "a0"]),
             (CF.MODEL_POLYNOMIAL, 0.2 * t ** 2 - t + 0.5, ["a2", "a1", "a0"])]
    for model, v, letters in cases:
        r = CF.fit_curve(list(zip(t, v)), model)
        assert r.ok, (model, r.message)
        x = explain_fit(r)
        import re
        assert [re.sub(r"<[^>]+>", "", p["sym"]) for p in x["params"]] == letters
        for p in x["params"]:
            assert p["name"] and p["meaning"] and p["unit"]
        assert x["general"].startswith("<i>v</i>(<i>t</i>)")
        if model in (CF.MODEL_SINUSOID, CF.MODEL_DAMPED):
            assert any(q["name"] == "Pendulum length" for q in x["results"])
    assert fit_quality(0.97) == ("Excellent fit", "ok") and fit_quality(0.3)[1] == "bad"
    assert g(32412.7) == "32,400" and g(-0.0000123) == "−1.23 × 10<sup>−5</sup>" and g(-0.691) == "−0.691"
    peaks = CF.find_peaks(t, wave(1e9))
    assert "high peak" in explain_peaks(peaks, "m/s")


def test_the_tab_auto_fits_the_frozen_spectrogram_and_shows_cards():
    from core import processing as P
    from ui.curve_fit_tab import CurveFitTab
    P.set_method(P.METHOD_STFT)
    tab = CurveFitTab()
    tab.resize(1300, 850)
    tab.show()
    APP.processEvents()
    # A ridge in the app's own buffer, row 0 the most negative velocity
    fn = lambda t: 1.5 * np.sin(2 * np.pi * 0.6 * t)
    for col in range(tab._width):
        column = np.full(tab._freq_bins, -60.0, dtype=np.float32)
        row = int(round(fn((col + 0.5) * tab._time_scale) / tab._vel_scale + tab._freq_bins / 2))
        column[max(0, row - 2):row + 3] = -5.0
        tab.on_spectrogram_frame(column[:, None])
    tab._auto_fit()
    assert tab.is_frozen() and tab._auto
    assert "Auto Fit" in tab._status.text(), tab._status.text()
    assert tab._peaks and tab._peaks["positive"] and tab._peaks["negative"]
    body = tab._results_area.widget()
    texts = " ".join(l.text() for l in body.findChildren(QtWidgets.QLabel))
    for word in ("Your fitted curve", "AMPLITUDE", "FREQUENCY", "PENDULUM LENGTH", "high peak"):
        assert word in texts, word
    # Drawing by hand afterwards replaces what Auto Fit found
    tab._trace_btn.setChecked(True)
    tab._add_point(QtCore.QPointF(200, 150))
    assert not tab._auto and len(tab._points) == 1
    tab.shutdown()
    tab.hide()


def test_the_tab_draws_the_range_map_with_the_distance_shown():
    from core import range_map as RM
    from ui.curve_fit_tab import CurveFitTab
    tab = CurveFitTab()
    tab.show()
    APP.processEvents()
    try:
        assert tab._range_opts.isHidden()
        tab._kind_combo.setCurrentIndex(tab._kind_combo.findData("range"))
        APP.processEvents()
        assert tab._kind == "range" and not tab._range_opts.isHidden()
        assert tab._range_worker is not None
        hist = np.full((50, 512), -60.0)
        hist[:, 150] = -10.0
        tab._on_range_history(hist, 50, 50)
        assert tab._img.image.shape == (50, 512, 3)
        RM.set_settings(min_m=0.5, max_m=1.5)
        APP.processEvents()
        lo, hi = tab._plot.vb.viewRange()[1]
        assert abs(lo - 0.5) < 1e-6 and abs(hi - 1.5) < 1e-6
        tab._freeze()
        assert tab._frozen_spec.shape == (512, 50), "rows from the nearest bin"
        tab._kind_combo.setCurrentIndex(tab._kind_combo.findData("doppler"))
        APP.processEvents()
        assert tab._range_worker is None and not tab.is_frozen()
    finally:
        tab.shutdown()
        RM.set_settings(mti=False, min_m=0.0, max_m=RM.RANGE_MAX_M)
        tab.hide()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
    print("\nAll checks passed.")
