"""
Tests for the reference spectrogram views.

The Infineon SDK display is drawn by core/doppler_spectrogram_live.py, a
verbatim copy of the reference script that must never be edited. These tests
pin that the vendored file is still byte-identical, that the capture view draws
exactly what the reference computes, and that each tab puts the reference in
front for the Infineon SDK method and steps it aside for STFT.

Run:  python -m pytest tests/test_reference_views.py -v
"""

import hashlib
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PyQt6 import QtCore, QtWidgets

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

from core import processing as P

VENDORED_SHA256 = (
    "5138ad8cb5b6d1c9889c0d6062196556cddbdbf1ba441fed283dfc9cfebb1b8c"
)


def _settle():
    """Let pool threads finish and their queued results reach the GUI thread."""
    QtCore.QThreadPool.globalInstance().waitForDone(30000)
    for _ in range(5):
        APP.processEvents()


def _cube(seed, n_frame=12):
    """A small raw capture with a moving target, (n_frame, n_ant, chirp, sample)."""
    rng = np.random.default_rng(seed)
    cube = rng.standard_normal((n_frame, 3, 32, 64)).astype(np.float32)
    t = np.arange(n_frame)[:, None, None, None]
    s = np.arange(64)[None, None, None, :]
    c = np.arange(32)[None, None, :, None]
    cube += 5.0 * np.cos(2 * np.pi * (0.12 * s + (0.05 + 0.01 * t) * c))
    return cube


def _stft():
    P.set_method(P.METHOD_STFT)


def test_vendored_reference_script_is_unmodified():
    source = (ROOT / "core" / "doppler_spectrogram_live.py").read_bytes()
    assert hashlib.sha256(source).hexdigest() == VENDORED_SHA256, (
        "core/doppler_spectrogram_live.py has been edited; it must stay an "
        "exact copy of the reference script"
    )


def test_capture_view_draws_exactly_what_the_reference_computes():
    from ui.reference_view import (
        REF_JET_VMIN, ReferenceRecordedView, _import_reference)
    ref = _import_reference()
    view = ReferenceRecordedView()
    try:
        cube = _cube(1)
        view.show_capture(cube)
        _settle()
        assert view.figure is not None, "nothing was drawn"
        image = view.figure.axes[0].images[0]
        expected, _ = ref.compute_recorded(cube, antenna=0)
        assert np.array_equal(np.asarray(image.get_array()), expected.T)
        vmin, vmax = ref._jet_clim(expected.T, REF_JET_VMIN)
        assert (image.norm.vmin, image.norm.vmax) == (vmin, vmax)
    finally:
        view.shutdown()


def test_rapid_captures_show_only_the_latest_and_do_not_pile_up():
    import matplotlib.pyplot as plt
    from ui.reference_view import ReferenceRecordedView, _import_reference
    ref = _import_reference()
    view = ReferenceRecordedView()
    before = len(plt.get_fignums())
    try:
        first, second = _cube(2), _cube(3)
        view.show_capture(first)
        view.show_capture(second)    # queued before the first can be drawn
        _settle()
        expected, _ = ref.compute_recorded(second, antenna=0)
        drawn = np.asarray(view.figure.axes[0].images[0].get_array())
        assert np.array_equal(drawn, expected.T), "a stale capture was drawn"

        view.show_capture(first)
        _settle()
        assert len(plt.get_fignums()) == before + 1, "old figures were not closed"
    finally:
        view.shutdown()
    assert len(plt.get_fignums()) == before


def test_visualize_puts_the_reference_in_front_for_infineon():
    from ui.spectrogram_widget import VisualizeTab
    _stft()
    tab = VisualizeTab()
    try:
        tab._combo.setCurrentIndex(tab._combo.findData(P.METHOD_INFINEON))
        assert tab._reference is not None and tab._reference.is_ready
        assert tab.spectrogram.isHidden()
        for w in (tab._sensds_axes, tab._sensds_image, tab._zoom_row,
                  tab._flip_check):
            assert w.isHidden(), "a SensDS-only control is showing"
        assert not tab._alongside_check.isHidden()

        tab._alongside_check.setChecked(True)
        assert not tab.spectrogram.isHidden()
        assert not tab._sensds_axes.isHidden()
        assert tab._reference is not None

        tab._combo.setCurrentIndex(tab._combo.findData(P.METHOD_STFT))
        assert tab._reference is None
        assert not tab.spectrogram.isHidden()
        assert tab._alongside_check.isHidden()
    finally:
        tab.stop_if_running()
        _stft()


def test_visualize_follows_a_method_change_made_on_another_tab():
    """
    Regression: the Collect tab can switch the method. Visualize used to keep
    its combo and buffer on the old one, so SpectrogramWidget.update_frame
    dropped every batch as the wrong shape and the live view froze silently.
    """
    from ui.spectrogram_widget import VisualizeTab
    _stft()
    tab = VisualizeTab()
    try:
        assert tab._combo.currentData() == P.METHOD_STFT
        P.set_method(P.METHOD_INFINEON)          # as the Collect tab does
        tab.show()
        APP.processEvents()
        assert tab._combo.currentData() == P.METHOD_INFINEON
        assert tab.spectrogram.freq_bins() == P.method_freq_bins(P.METHOD_INFINEON)
        assert tab._reference is not None
    finally:
        tab.stop_if_running()
        tab.hide()
        _stft()


def test_hidden_sensds_view_still_keeps_its_buffer_current():
    """Skipping the repaint while hidden must not skip the data."""
    from ui.spectrogram_widget import SpectrogramWidget
    _stft()
    w = SpectrogramWidget()                     # never shown
    batch = np.full((w.freq_bins(), 2), -5.0, dtype=np.float32)
    w.update_frame(batch)
    assert np.all(w._buffer[:, :2] == -5.0)


def test_collect_preview_is_drawn_by_the_reference_for_infineon():
    from ui.collect_tab import CollectTab
    _stft()
    tab = CollectTab()
    try:
        assert tab._preview_stack.currentWidget() is tab._preview_widget
        tab._method_combo.setCurrentIndex(
            tab._method_combo.findData(P.METHOD_INFINEON))
        assert tab._ref_preview is not None
        assert tab._preview_stack.currentWidget() is tab._ref_preview
        tab._method_combo.setCurrentIndex(
            tab._method_combo.findData(P.METHOD_STFT))
        assert tab._preview_stack.currentWidget() is tab._preview_widget
    finally:
        if tab._ref_preview is not None:
            tab._ref_preview.shutdown()
        _stft()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
    print("\nAll checks passed.")
