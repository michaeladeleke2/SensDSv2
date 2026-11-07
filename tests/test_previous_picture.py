"""
The previous spectrogram (core/previous_picture.py), which the Test tab shows
the starter model.

fixtures/previous_picture/expected_ratio1.png was drawn by SensDSv2's own code
from before the change note (commit 418a1ae^: the previous range bin rule and
colour limits) from synthetic_capture(), so with the previous chirp (no
stretch) the picture must match it byte for byte.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")

from core import previous_picture as PP

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "previous_picture", "expected_ratio1.png")


def synthetic_capture(n_frame=20, n_chirp=128, n_sample=256):
    """A target at a fixed range whose speed sweeps across the capture."""
    c = np.arange(n_chirp)[:, None]
    s = np.arange(n_sample)[None, :]
    frames = []
    for f in range(n_frame):
        v = 0.04 * np.sin(2 * np.pi * f / n_frame)
        x = 0.5 + 0.1 * np.cos(2 * np.pi * (s * 40 / 1024 + c * v)) + 0.02 * np.cos(2 * np.pi * s * 90 / 1024)
        frames.append(np.stack([x, x, x]))
    return np.asarray(frames, dtype=np.float64)


def test_speed_ratio():
    assert PP.speed_ratio(0.0003) == 1.5
    assert PP.speed_ratio(0.0002) == 1.0


def test_matches_the_drawing_before_the_change_note():
    from PIL import Image
    got = np.asarray(PP.previous_image(synthetic_capture(), ratio=1))
    want = np.asarray(Image.open(FIXTURE).convert("RGB"))
    assert got.shape == (300, 400, 3)
    assert np.array_equal(got, want)


def test_stretch_onto_the_previous_speed_range():
    row = np.array([[-80, -70, -60, -50, 0, -50, -60, -70]], dtype=np.float64)   # zero speed at bin 4
    out = PP.to_previous_speed_range(row, 1.5)
    # bin k reads current bin 4 + (k - 4) * 1.5 (bin 3 -> 2.5, between -60 and -50);
    # past the ends (bins 0, 1 and 7), the column's median (-60)
    assert out[0].tolist() == [-60, -60, -70, -55, 0, -55, -70, -60]


def test_stretched_picture_differs_from_the_unstretched_one():
    cube = synthetic_capture()
    a = np.asarray(PP.previous_image(cube))
    b = np.asarray(PP.previous_image(cube, ratio=1))
    assert a.shape == b.shape and not np.array_equal(a, b)
