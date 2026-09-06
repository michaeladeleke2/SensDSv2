"""
Tests for locating the spectrogram PNG that belongs to a raw radar cube.

The Features tab plots one point per raw cube and shows that recording's
spectrogram when the point is clicked, so the raw -> PNG mapping has to hold
for the trio the Collect tab writes (sample_NNN.npy, sample_NNN_raw.npy,
sample_NNN.png) and has to return None rather than a bad path when the image
is absent — older captures predate the raw cube, and folders get pruned.

Run:  python -m pytest tests/test_sample_png_lookup.py -v
      python tests/test_sample_png_lookup.py            # standalone
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import physical_features as PF


def _dataset():
    """A miniature capture folder; returns its root."""
    root = Path(tempfile.mkdtemp(prefix="sensds_png_"))
    gdir = root / "Vincent" / "push"
    gdir.mkdir(parents=True)
    for i in (1, 2, 3):
        (gdir / f"sample_{i:03d}_raw.npy").write_bytes(b"")
        (gdir / f"sample_{i:03d}.npy").write_bytes(b"")
    # Sample 1 and 2 have images; sample 3 does not.
    (gdir / "sample_001.png").write_bytes(b"")
    (gdir / "sample_002.png").write_bytes(b"")
    return root


def test_png_found_beside_raw_cube():
    root = _dataset()
    try:
        raw = root / "Vincent" / "push" / "sample_001_raw.npy"
        assert PF.png_for_sample(str(raw)) == str(
            root / "Vincent" / "push" / "sample_001.png"
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_missing_png_returns_none():
    root = _dataset()
    try:
        raw = root / "Vincent" / "push" / "sample_003_raw.npy"
        assert PF.png_for_sample(str(raw)) is None
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_raw_suffix_is_stripped_not_just_the_extension():
    """
    sample_001_raw.npy -> sample_001.png, never sample_001_raw.png.

    Dropping only the extension would look for a file that is never written,
    so every point would report a missing image.
    """
    root = _dataset()
    try:
        raw = root / "Vincent" / "push" / "sample_001_raw.npy"
        found = PF.png_for_sample(str(raw))
        assert found is not None
        assert not Path(found).name.endswith("_raw.png")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_scanned_samples_all_resolve():
    """Every path scan_samples() hands the Features tab is a usable input."""
    root = _dataset()
    try:
        samples = PF.scan_samples(str(root))
        assert len(samples) == 3
        resolved = [PF.png_for_sample(s["path"]) for s in samples]
        assert sum(1 for p in resolved if p) == 2
        assert all(p is None or Path(p).is_file() for p in resolved)
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
    print("\nAll checks passed.")
