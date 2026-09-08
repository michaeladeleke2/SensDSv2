"""
Tests that saved training images and live inference images stay the same way up.

The Collect tab writes the PNGs a model is trained on; the Test and VEX AIM
tabs build the images that model is asked to classify. Those are three separate
code paths, and if their row order ever drifts apart the model is fed upside
down pictures at inference time and simply predicts badly — with no error
anywhere. All three go through core.processing.image_rows() for exactly that
reason, and these tests pin it.

A vertical flip is cosmetic to the model as long as the whole dataset agrees;
it swaps toward for away, so a dataset that mixes both orientations cannot
separate an approaching gesture from a receding one.

Run:  python -m pytest tests/test_image_orientation.py -v
      python tests/test_image_orientation.py            # standalone
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import processing as P


def _spectrogram_with_marker(row: int, bins: int = 512, cols: int = 30):
    """A dB spectrogram whose only bright row is `row`."""
    spec = np.full((bins, cols), P.DB_MIN, dtype=np.float32)
    spec[row] = P.DB_MAX
    return spec


def _restore():
    P.set_method(P.METHOD_STFT)
    P.set_image_velocity_flipped(False)


def test_default_puts_positive_velocity_at_the_top():
    """
    Row 0 is the most negative velocity after fftshift, so unflipped it must
    end up at the bottom of the image.
    """
    P.set_image_velocity_flipped(False)
    try:
        rows = P.image_rows(_spectrogram_with_marker(0))
        assert int(np.argmax(rows[:, 0])) == rows.shape[0] - 1
    finally:
        _restore()


def test_flipped_puts_negative_velocity_at_the_top():
    P.set_image_velocity_flipped(True)
    try:
        rows = P.image_rows(_spectrogram_with_marker(0))
        assert int(np.argmax(rows[:, 0])) == 0
    finally:
        _restore()


def test_flip_is_an_exact_mirror():
    spec = _spectrogram_with_marker(37)
    try:
        P.set_image_velocity_flipped(False)
        unflipped = P.image_rows(spec)
        P.set_image_velocity_flipped(True)
        flipped = P.image_rows(spec)
        assert np.array_equal(unflipped[::-1], flipped)
    finally:
        _restore()


def test_colored_three_channel_arrays_flip_over_rows_only():
    """Call sites pass the RGB array, not the dB one; channels must survive."""
    colored = np.zeros((8, 4, 3), dtype=np.uint8)
    colored[0] = [10, 20, 30]
    try:
        P.set_image_velocity_flipped(False)
        out = P.image_rows(colored)
        assert out.shape == colored.shape
        assert list(out[-1, 0]) == [10, 20, 30]
    finally:
        _restore()


def test_collect_and_inference_agree_on_row_order():
    """
    The heart of it: the Collect tab's PNG and the Test tab's inference image
    must come out the same way up, under both settings.
    """
    from ui.collect_tab import _apply_jet_colormap
    from ui.test_tab import _apply_jet

    normalized = np.zeros((64, 10), dtype=np.float32)
    normalized[3] = 1.0
    try:
        for flipped in (False, True):
            P.set_image_velocity_flipped(flipped)
            saved = P.image_rows(_apply_jet_colormap(normalized))
            inferred = P.image_rows(_apply_jet(normalized))
            bright_saved = int(np.argmax(saved[:, 0, 0]))
            bright_inferred = int(np.argmax(inferred[:, 0, 0]))
            assert bright_saved == bright_inferred, (
                f"flipped={flipped}: saved row {bright_saved} but "
                f"inference row {bright_inferred}"
            )
    finally:
        _restore()


def test_vex_aim_uses_the_same_helper():
    """The third image path must not have been left behind on its own flip."""
    source = (Path(__file__).resolve().parent.parent
              / "ui" / "vex_aim_tab.py").read_text()
    assert "image_rows(_apply_jet(normalized))" in source
    assert "_apply_jet(normalized)[::-1]" not in source


def test_both_methods_still_report_their_shapes():
    """Switching method from the Collect tab must not disturb the geometry."""
    try:
        P.set_method(P.METHOD_INFINEON)
        assert P.method_freq_bins() == P.RDM_DOPPLER_BINS
        assert P.method_cols_per_frame() == P.RDM_COLS_PER_FRAME
        P.set_method(P.METHOD_STFT)
        assert P.method_freq_bins() == P.STFT_NFFT
        assert P.method_cols_per_frame() == P.COLS_PER_FRAME
    finally:
        _restore()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
    print("\nAll checks passed.")
