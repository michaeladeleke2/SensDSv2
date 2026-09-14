"""
Tests for the Infineon SDK training image, drawn the reference script's way.

Students are meant to see exactly what the model learns from. That rests on
what these tests pin: the image is made from the reference script's own pieces
and matches the colors its plot assigns, the image the Test and VEX AIM tabs
hand the model live is identical to the one the Collect tab saves, and the
Reduce noise option does what it says without quietly mixing datasets.

Run:  python -m pytest tests/test_reference_image.py -v
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PyQt6 import QtWidgets

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

from core import processing as P
from core import reference_image as R


def _cube(seed, n_frame=12):
    """A small raw capture with a moving target, (n_frame, n_ant, chirp, sample)."""
    rng = np.random.default_rng(seed)
    cube = rng.standard_normal((n_frame, 3, 32, 64)).astype(np.float32)
    t = np.arange(n_frame)[:, None, None, None]
    s = np.arange(64)[None, None, None, :]
    c = np.arange(32)[None, None, :, None]
    cube += 5.0 * np.cos(2 * np.pi * (0.12 * s + (0.05 + 0.01 * t) * c))
    return cube


def _restore():
    R.set_reduce_noise(False)
    P.set_method(P.METHOD_STFT)


def test_app_opens_on_the_infineon_method():
    """Checked in a fresh interpreter: other tests change the method."""
    out = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
         "from core import processing as P; print(P.get_method())"],
        capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == P.METHOD_INFINEON


def test_colors_match_what_his_plot_assigns():
    """Cell for cell, at both color floors, against his own imshow."""
    import matplotlib.pyplot as plt
    ref = R.import_reference()
    spec, _ = ref.compute_recorded(_cube(1), antenna=0)
    for floor in (R.REF_JET_VMIN, R.REDUCED_NOISE_JET_VMIN):
        was = plt.isinteractive()
        plt.ioff()
        try:
            fig = ref.plot_recorded_spectrogram(
                spec, max_speed_m_s=R.REF_MAX_SPEED_M_S, jet_vmin=floor)
        finally:
            if was:
                plt.ion()
        try:
            im = fig.axes[0].images[0]
            expected = im.to_rgba(im.get_array(), bytes=True)[..., :3]
            assert np.array_equal(R.reference_rgb(spec, jet_vmin=floor), expected)
        finally:
            plt.close(fig)


def test_spectrogram_is_his_compute_recorded():
    ref = R.import_reference()
    cube = _cube(4)
    expected, _ = ref.compute_recorded(cube, antenna=0)
    assert np.array_equal(R.reference_spectrogram(cube), expected)


def test_top_row_is_the_most_negative_velocity():
    """His orientation: fftshift bin 0, the most negative velocity, on top."""
    spec = np.full((10, 512), -60.0)
    spec[:, 0] = 0.0
    rgb = R.reference_rgb(spec, jet_vmin=-20.0)
    assert int(rgb[0, 0, 0]) > int(rgb[256, 0, 0])      # red on top row only


def test_saved_and_live_inference_images_are_identical():
    from PIL import Image
    from ui import test_tab, vex_aim_tab
    P.set_method(P.METHOD_INFINEON)
    d = tempfile.mkdtemp()
    try:
        cube = _cube(2, n_frame=20)
        saved = R.training_image(R.reference_spectrogram(cube))
        path = os.path.join(d, "sample_001.png")
        saved.save(path)
        reloaded = np.array(Image.open(path).convert("RGB"))
        assert np.array_equal(reloaded, np.array(saved)), "PNG is not lossless"
        assert reloaded.shape == (300, 400, 3)
        for tab in (test_tab, vex_aim_tab):
            live = tab._frames_to_pil(list(cube))
            assert live is not None, f"{tab.__name__} produced no image"
            assert np.array_equal(np.array(live), reloaded), (
                f"{tab.__name__} shows the model a different picture")
    finally:
        shutil.rmtree(d)
        _restore()


def test_reduce_noise_follows_the_option():
    try:
        R.set_reduce_noise(True)
        assert R.current_jet_vmin() == R.REDUCED_NOISE_JET_VMIN
        R.set_reduce_noise(False)
        assert R.current_jet_vmin() == R.REF_JET_VMIN
    finally:
        _restore()


def test_reduce_noise_keeps_a_still_scene_dark():
    """Idle as measured: a clutter line near -38 dB over noise near -58 dB."""
    from matplotlib import colormaps
    rng = np.random.default_rng(0)
    spec = -58.0 + rng.normal(0.0, 2.0, (20, 512))
    spec[:, 256] = -38.0
    darkest = np.array(colormaps["jet"](0.0, bytes=True)[:3])

    def dark_fraction(rgb):
        return float(np.mean(np.all(rgb == darkest, axis=-1)))

    assert dark_fraction(R.reference_rgb(spec, jet_vmin=-20.0)) < 0.2, (
        "the reference default should show idle as noise")
    assert dark_fraction(R.reference_rgb(spec, jet_vmin=-50.0)) > 0.9, (
        "Reduce noise should leave idle dark but for the clutter line")


def test_folders_drawn_the_old_way_are_flagged():
    from ui.collect_tab import capture_mismatch, write_capture_info
    d = tempfile.mkdtemp()
    try:
        Path(d, "sample_001.npy").write_bytes(b"")
        # As written before the renderer was recorded.
        Path(d, "capture_info.json").write_text(
            '{"spectrogram_method": "infineon", "velocity_flipped": true}')
        assert "coloring" in capture_mismatch(d, P.METHOD_INFINEON, True)

        write_capture_info(d, P.METHOD_INFINEON, True)
        assert capture_mismatch(d, P.METHOD_INFINEON, True) == ""

        R.set_reduce_noise(True)
        assert "Reduce noise" in capture_mismatch(d, P.METHOD_INFINEON, True)
    finally:
        shutil.rmtree(d)
        _restore()


def test_collect_saves_the_reference_drawing():
    from PIL import Image
    from ui.collect_tab import CollectTab, read_capture_info
    P.set_method(P.METHOD_INFINEON)
    tab = CollectTab()
    d = tempfile.mkdtemp()
    try:
        cube = _cube(3, n_frame=20)
        spec = R.reference_spectrogram(cube)
        tab._save_dir = d
        tab._samples_collected = 0
        tab._on_sample_done(spec, 20, cube, P.METHOD_INFINEON)
        saved = np.array(Image.open(os.path.join(d, "sample_001.png")).convert("RGB"))
        assert np.array_equal(saved, np.array(R.training_image(spec)))
        info = read_capture_info(d)
        assert info["renderer"] == "reference"
        assert info["jet_vmin"] == R.REF_JET_VMIN
        assert tab._ref_preview is not None and tab._ref_preview.figure is not None
    finally:
        if tab._ref_preview is not None:
            tab._ref_preview.shutdown()
        shutil.rmtree(d)
        _restore()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
    print("\nAll checks passed.")
