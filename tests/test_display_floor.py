"""
Tests for the absolute floor under the Infineon display window.

The window is normally 40 dB below the frame's own peak. That works while
something is moving, but on a still scene the brightest thing is static clutter
only ~20 dB above the receiver noise, so the window reached past the noise
floor and painted it mid-scale: idle captures came out solid green instead of
the empty blue chart they should be.

RDM_DISPLAY_FLOOR_DB clamps the bottom of the window. These tests pin the two
halves of that bargain: a quiet frame's noise gets pushed below the visible
range, and a frame with a real return is left exactly as it was.

Run:  python -m pytest tests/test_display_floor.py -v
      python tests/test_display_floor.py            # standalone
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import processing as P

# Levels measured from 50 real captures; see RDM_DISPLAY_FLOOR_DB.
IDLE_PEAK_DB = -38.1
PUSH_PEAK_DB = -7.4
NOISE_DB = -58.0


def _frame(peak_db, noise_db=NOISE_DB, shape=(512, 20)):
    """A flat noise field with one bright row, in absolute dB."""
    spec = np.full(shape, noise_db, dtype=np.float32)
    spec[shape[0] // 2] = peak_db
    return spec


def test_quiet_frame_renders_as_empty():
    """Idle: noise must land at the very bottom of the display range."""
    out = P.doppler_to_display_db(_frame(IDLE_PEAK_DB))
    noise = out[0]                      # a row that is pure noise
    assert np.allclose(noise, P.DB_MIN), f"noise mapped to {noise[0]}, want {P.DB_MIN}"


def test_quiet_frame_keeps_its_clutter_line():
    """The zero-velocity clutter is real and should still be visible."""
    spec = _frame(IDLE_PEAK_DB)
    out = P.doppler_to_display_db(spec)
    assert np.allclose(out[out.shape[0] // 2], P.DB_MAX)
    # And it is the only thing lit.
    assert float(np.mean(out > P.DB_MIN + 0.5)) < 0.05


def test_frame_with_a_real_return_is_unchanged():
    """
    A push peaks ~30 dB above idle, so peak - 40 already clears the floor and
    the mapping must be untouched.
    """
    spec = _frame(PUSH_PEAK_DB)
    floored = P.doppler_to_display_db(spec)

    saved = P.RDM_DISPLAY_FLOOR_DB
    try:
        P.RDM_DISPLAY_FLOOR_DB = -1e9        # effectively no floor
        unfloored = P.doppler_to_display_db(spec)
    finally:
        P.RDM_DISPLAY_FLOOR_DB = saved
    assert np.array_equal(floored, unfloored)


def test_the_floor_sits_inside_the_idle_to_push_gap():
    """
    The whole scheme depends on one number falling in a gap. If the floor ever
    drifts out of it, one of the two behaviours above silently breaks.
    """
    assert IDLE_PEAK_DB < P.RDM_DISPLAY_FLOOR_DB + P.RDM_DYNAMIC_RANGE_DB
    assert P.RDM_DISPLAY_FLOOR_DB > NOISE_DB      # noise falls below the window
    assert PUSH_PEAK_DB - P.RDM_DYNAMIC_RANGE_DB > P.RDM_DISPLAY_FLOOR_DB


def test_output_stays_inside_the_display_range():
    for peak in (IDLE_PEAK_DB, PUSH_PEAK_DB, 0.0, 20.0):
        out = P.doppler_to_display_db(_frame(peak))
        assert out.min() >= P.DB_MIN - 1e-4
        assert out.max() <= P.DB_MAX + 1e-4


def test_a_flat_frame_does_not_divide_by_zero():
    """Peak equal to floor collapses the window; it must not produce NaNs."""
    flat = np.full((64, 8), P.RDM_DISPLAY_FLOOR_DB, dtype=np.float32)
    out = P.doppler_to_display_db(flat)
    assert np.all(np.isfinite(out))


def test_explicit_vmax_still_gets_the_floor():
    """The live view passes its own held peak; the clamp must apply there too."""
    spec = _frame(IDLE_PEAK_DB)
    out = P.doppler_to_display_db(spec, vmax=IDLE_PEAK_DB)
    assert np.allclose(out[0], P.DB_MIN)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
    print("\nAll checks passed.")
