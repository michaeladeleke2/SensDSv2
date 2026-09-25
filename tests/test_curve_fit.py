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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
    print("\nAll checks passed.")
