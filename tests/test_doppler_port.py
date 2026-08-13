"""
Verifies core.processing.doppler_spectrogram_from_frames() against a LITERAL
transcription of the original doppler_spectrogram.py loops.

The shipped version is vectorised (batched numpy FFTs instead of the original's
19 200 sequential 1-D FFTs per 3 s epoch).  This test is the proof that the
optimisation did not change the numbers.

Run:  python -m pytest tests/test_doppler_port.py -v
      python tests/test_doppler_port.py            # standalone
"""

import sys
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.processing import (  # noqa: E402
    doppler_spectrogram_from_frames,
    RDM_CLIPPING_VALUE, RDM_SPECT_THRESHOLD, RDM_SMOOTH_WINDOW,
)

CLIPPING_VALUE = RDM_CLIPPING_VALUE
SPECT_THRESHOLD = RDM_SPECT_THRESHOLD
SMOOTH_WINDOW = RDM_SMOOTH_WINDOW


def reference_compute(data: np.ndarray, antenna: int = 0) -> np.ndarray:
    """
    Literal transcription of doppler_spectrogram.py:compute(), minus the
    matplotlib plotting.  Loops exactly as the original does.
    """
    n_frame, n_ant, n_chirp, n_sample = data.shape

    range_fft_size = n_sample * 4
    doppler_fft_size = n_chirp * 4
    n_range_bins = range_fft_size // 2

    try:
        range_window = signal.blackmanharris(n_sample)
        doppler_window = signal.chebwin(n_chirp, at=100.0)
    except AttributeError:
        range_window = signal.windows.blackmanharris(n_sample)
        doppler_window = signal.windows.chebwin(n_chirp, at=100.0)
    doppler_window = doppler_window / np.sum(doppler_window)

    clip_db = 20.0 * np.log10(CLIPPING_VALUE)
    rdm_cube = np.zeros((n_frame, n_range_bins, doppler_fft_size), dtype=np.float64)

    for frame_idx in range(n_frame):
        frame = data[frame_idx, antenna].astype(np.float64, copy=False)
        rdm_complex = np.zeros((n_range_bins, n_chirp), dtype=np.complex128)

        for chirp_idx in range(n_chirp):
            chirp = frame[chirp_idx]
            x = chirp - np.mean(chirp)
            x = x * range_window
            buf = np.zeros(range_fft_size, dtype=np.complex128)
            buf[:n_sample] = x
            spectrum = np.fft.fft(buf)
            rdm_complex[:, chirp_idx] = spectrum[:n_range_bins]

        for range_idx in range(n_range_bins):
            slow_time = rdm_complex[range_idx] - np.mean(rdm_complex[range_idx])
            slow_time = slow_time * doppler_window
            buf = np.zeros(doppler_fft_size, dtype=np.complex128)
            buf[:n_chirp] = slow_time
            shifted = np.fft.fftshift(np.fft.fft(buf))
            power = np.abs(shifted) ** 2
            db = np.empty(doppler_fft_size, dtype=np.float64)
            above = power >= SPECT_THRESHOLD ** 2
            db[above] = 10.0 * np.log10(power[above])
            db[~above] = clip_db
            rdm_cube[frame_idx, range_idx] = db

    per_frame_energy = np.argmax(rdm_cube.sum(axis=2), axis=1).astype(np.int32)
    window = max(1, SMOOTH_WINDOW | 1)
    smoothed = median_filter(per_frame_energy.astype(np.float64), size=window)
    range_bin = np.clip(np.round(smoothed), 0, n_range_bins - 1).astype(np.int32)

    spectrogram = np.zeros((n_frame, doppler_fft_size), dtype=np.float64)
    for i in range(n_frame):
        spectrogram[i] = rdm_cube[i, range_bin[i], :]
    return spectrogram.T


def _synthetic_frames(n_frame=6, n_ant=3, n_chirp=32, n_sample=64, seed=0):
    """
    Synthetic radar-like data: a moving target (range walks with time, with a
    Doppler shift) plus static clutter and noise.  Small dims keep the O(n^2)
    reference loop fast while exercising every code path.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_sample)
    out = np.zeros((n_frame, n_ant, n_chirp, n_sample))
    for fi in range(n_frame):
        beat = 3.0 + 0.35 * fi                      # range walks across frames
        for ai in range(n_ant):
            for ci in range(n_chirp):
                phase = 2 * np.pi * 0.012 * ci      # Doppler across slow time
                out[fi, ai, ci] = (
                    np.sin(2 * np.pi * beat * t / n_sample + phase)
                    + 0.4 * np.sin(2 * np.pi * 1.0 * t / n_sample)   # clutter
                    + 0.05 * rng.standard_normal(n_sample)
                )
    return out


# Tolerance note
# ──────────────
# Batched numpy FFTs sum in a different order than the original's sequential
# 1-D FFTs, so results agree to floating-point rounding rather than bit-for-bit
# (observed max ~3e-13 dB on values of magnitude ~120 dB, i.e. ~1e-15 relative).
# This tolerance is also a strong check on the RANGE-BIN SELECTION: adjacent
# range bins differ by many dB, so if the vectorised code ever picked a
# different bin the error would be O(1) dB, thousands of times above ATOL.
ATOL_DB = 1e-9


def test_matches_reference_float64():
    """Vectorised float64 path must match the original loops to fp rounding."""
    data = _synthetic_frames()
    expected = reference_compute(data)
    actual = doppler_spectrogram_from_frames(data, cube_dtype=np.float64)

    assert actual.shape == expected.shape
    max_diff = float(np.max(np.abs(actual - expected)))
    assert max_diff < ATOL_DB, f"max abs diff = {max_diff} dB"


def test_range_bin_selection_matches():
    """
    Pin the range-bin tracking specifically, independent of the dB values:
    every emitted column must be an exact row of the reference RDM cube.
    """
    data = _synthetic_frames()
    expected = reference_compute(data)
    actual = doppler_spectrogram_from_frames(data, cube_dtype=np.float64)

    # Per-column error must stay at rounding level for EVERY frame, which can
    # only happen if the same range bin was selected for each one.
    per_col = np.max(np.abs(actual - expected), axis=0)
    assert np.all(per_col < ATOL_DB), f"columns diverged: {per_col}"


def test_chunking_is_invariant():
    """Chunk size must not affect the result beyond fp rounding."""
    data = _synthetic_frames(n_frame=7)
    ref = doppler_spectrogram_from_frames(data, chunk=64, cube_dtype=np.float64)
    for c in (1, 2, 3, 8):
        got = doppler_spectrogram_from_frames(data, chunk=c, cube_dtype=np.float64)
        max_diff = float(np.max(np.abs(got - ref)))
        assert max_diff < ATOL_DB, f"chunk={c} differs by {max_diff} dB"


def test_float32_cube_is_visually_identical():
    """
    The shipped default stores the dB cube as float32 to halve memory.
    Confirm that costs us nothing that could ever be seen: same range-bin
    picks, and dB error far below display resolution.
    """
    data = _synthetic_frames()
    exact = doppler_spectrogram_from_frames(data, cube_dtype=np.float64)
    packed = doppler_spectrogram_from_frames(data, cube_dtype=np.float32)

    assert exact.shape == packed.shape
    max_diff = float(np.max(np.abs(exact - packed)))
    # Display spans 40 dB over 256 colour levels -> 0.16 dB per level.
    assert max_diff < 1e-3, f"float32 cube diverged by {max_diff} dB"


def test_antenna_selection():
    """Different antennas must give different output, and index is validated."""
    data = _synthetic_frames()
    data[:, 1] *= 2.5                       # make antenna 1 clearly different
    a0 = doppler_spectrogram_from_frames(data, antenna=0, cube_dtype=np.float64)
    a1 = doppler_spectrogram_from_frames(data, antenna=1, cube_dtype=np.float64)
    assert not np.allclose(a0, a1)
    assert np.max(np.abs(a1 - reference_compute(data, antenna=1))) < ATOL_DB

    try:
        doppler_spectrogram_from_frames(data, antenna=99)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for out-of-range antenna")


def test_single_frame_3d_input():
    """A 3-D (n_ant, n_chirp, n_sample) input is treated as one frame."""
    data = _synthetic_frames(n_frame=1)
    out3d = doppler_spectrogram_from_frames(data[0], cube_dtype=np.float64)
    out4d = doppler_spectrogram_from_frames(data, cube_dtype=np.float64)
    assert np.array_equal(out3d, out4d)   # same code path, must be exact
    assert out3d.shape[1] == 1          # exactly one column per frame


def test_output_shape_for_real_config():
    """Real radar config: 128 chirps / 256 samples -> 512 bins, 1 col per frame."""
    data = _synthetic_frames(n_frame=3, n_ant=3, n_chirp=128, n_sample=256)
    out = doppler_spectrogram_from_frames(data)
    assert out.shape == (512, 3), out.shape




# ── display / noise-floor behaviour ──────────────────────────────────────────

def _idle_frames(n_frame=8, seed=0):
    """Static clutter + noise, no moving target."""
    import numpy as _np
    rng = _np.random.default_rng(seed)
    t = _np.arange(256)
    out = _np.zeros((n_frame, 3, 128, 256))
    for f in range(n_frame):
        for a in range(3):
            out[f, a] = (0.7 * _np.sin(2 * _np.pi * 2.0 * t / 256)
                         + 0.05 * rng.standard_normal((128, 256)))
    return out


def _gesture_frames(n_frame=8, seed=1):
    """Moving target on top of the same clutter + noise."""
    import numpy as _np
    rng = _np.random.default_rng(seed)
    t = _np.arange(256)
    ci = _np.arange(128)[:, None]
    out = _np.zeros((n_frame, 3, 128, 256))
    for f in range(n_frame):
        for a in range(3):
            out[f, a] = (_np.sin(2 * _np.pi * 11 * t / 256 + 2 * _np.pi * 0.04 * ci)
                         + 0.7 * _np.sin(2 * _np.pi * 2.0 * t / 256)
                         + 0.05 * rng.standard_normal((128, 256)))
    return out


def test_idle_stays_dark_after_gesture():
    """
    Regression: the live view lit up with noise whenever nobody gestured.

    With a few frames and no target, the peak IS the noise floor, so a
    peak-relative colour scale stretched noise across the whole colormap.
    The held peak must keep quiet periods dark.
    """
    import core.processing as P
    prev = P.get_method()
    P.set_method(P.METHOD_INFINEON)
    try:
        proc = P.SpectrogramProcessor(streaming=True)
        for f in _gesture_frames(12):
            proc.push_frame_raw(f)
        proc.get_streaming_result(n_cols=4, n_frames=8)

        idle = _idle_frames(60)
        worst = 1.0
        for i, f in enumerate(idle):
            proc.push_frame_raw(f)
            r = proc.get_streaming_result(n_cols=4, n_frames=8)
            if r is not None and i > 10:          # let the buffer flush
                worst = min(worst, float((r <= P.DB_MIN + 2).mean()))
        assert worst > 0.95, f"idle view only {worst:.0%} dark — noise is showing"
    finally:
        P.set_method(prev)


def test_dynamic_range_setting_changes_contrast():
    """A tighter dynamic range must darken more of the image."""
    import core.processing as P
    raw = doppler_spectrogram_from_frames(_gesture_frames())
    tight = P.doppler_to_display_db(raw, dynamic_range_db=15)
    wide = P.doppler_to_display_db(raw, dynamic_range_db=55)
    dark_tight = float((tight <= P.DB_MIN + 2).mean())
    dark_wide = float((wide <= P.DB_MIN + 2).mean())
    assert dark_tight > dark_wide, (dark_tight, dark_wide)


def test_dynamic_range_setter_roundtrip():
    import core.processing as P
    prev = P.get_dynamic_range_db()
    try:
        P.set_dynamic_range_db(25)
        assert P.get_dynamic_range_db() == 25.0
        P.set_dynamic_range_db(1)          # clamped to a sane minimum
        assert P.get_dynamic_range_db() >= 5.0
    finally:
        P.set_dynamic_range_db(prev)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
    print("\nAll checks passed.")
