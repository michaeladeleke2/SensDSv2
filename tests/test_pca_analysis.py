"""
Tests for core/pca_analysis.py.

PCA is checked against an independent eigendecomposition and silhouette
against a hand-computed case, since neither has a library reference in this
project (no scikit-learn by design).

Run:  python tests/test_pca_analysis.py
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import pca_analysis as PA  # noqa: E402


# ── synthetic radar ──────────────────────────────────────────────────────────

def make_cube(n_frame=8, n_ant=3, n_chirp=32, n_sample=64, beat=8.0, dopp=0.04,
              seed=0):
    """Real-valued ADC-like data with a target at a given range and velocity."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_sample)
    ci = np.arange(n_chirp)[:, None]
    out = np.zeros((n_frame, n_ant, n_chirp, n_sample), dtype=np.float32)
    for f in range(n_frame):
        for a in range(n_ant):
            out[f, a] = (
                np.sin(2 * np.pi * beat * t / n_sample + 2 * np.pi * dopp * ci)
                + 0.4 * np.sin(2 * np.pi * 2.0 * t / n_sample)
                + 0.05 * rng.standard_normal((n_chirp, n_sample))
            )
    return out


# ── shapes / dtypes ──────────────────────────────────────────────────────────

def test_range_fft_shape():
    cube = make_cube()
    r = PA.range_fft(cube)
    n_frame, _, n_chirp, n_sample = cube.shape
    assert r.shape == (n_frame, n_sample * 4 // 2, n_chirp), r.shape
    assert np.iscomplexobj(r)


def test_doppler_db_shape_and_floor():
    cube = make_cube()
    n_chirp = cube.shape[2]
    rdm = PA.doppler_db(PA.range_fft(cube), n_chirp)
    assert rdm.shape == (cube.shape[0], 64 * 4 // 2, n_chirp * 4), rdm.shape
    assert rdm.min() >= 20.0 * np.log10(PA.CLIP_VALUE) - 1e-3


def test_doppler_db_chunk_invariant():
    cube = make_cube()
    r = PA.range_fft(cube)
    ref = PA.doppler_db(r, cube.shape[2], chunk=64)
    for c in (1, 2, 3):
        got = PA.doppler_db(r, cube.shape[2], chunk=c)
        assert np.array_equal(got, ref), f"chunk={c} differs"


def test_extract_spectrogram_shape():
    cube = make_cube()
    rdm = PA.doppler_db(PA.range_fft(cube), cube.shape[2])
    spec = PA.extract_spectrogram(rdm)
    assert spec.shape == (cube.shape[0], rdm.shape[2]), spec.shape


def test_feature_dims():
    cube = make_cube()
    assert PA.spectrogram_features(cube).shape == (1024,)
    assert PA.range_fft_features(cube).shape == (2048,)


def test_features_are_finite():
    cube = make_cube()
    for fn in (PA.spectrogram_features, PA.range_fft_features):
        v = fn(cube)
        assert np.all(np.isfinite(v)), f"{fn.__name__} produced non-finite values"


# ── PCA ──────────────────────────────────────────────────────────────────────

def test_pca_matches_eigendecomposition():
    """
    Cross-check the SVD implementation against an independent eigendecomposition
    of the covariance matrix. (The shipped code deliberately avoids forming that
    matrix; here it is only a reference.)
    """
    rng = np.random.default_rng(3)
    X = rng.standard_normal((40, 12)) @ rng.standard_normal((12, 12))
    proj, evr = PA.pca(X, k=2)

    Xc = X - X.mean(axis=0)
    cov = np.cov(Xc, rowvar=False, bias=True)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]

    ref_evr = vals[:2] / vals.sum()
    assert np.allclose(evr, ref_evr, atol=1e-10), (evr, ref_evr)

    ref_proj = Xc @ vecs[:, :2]
    # Sign of a component is arbitrary; compare magnitudes.
    assert np.allclose(np.abs(proj), np.abs(ref_proj), atol=1e-8)


def test_pca_variance_ordering_and_bounds():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((30, 8))
    _, evr = PA.pca(X, k=2)
    assert evr[0] >= evr[1]
    assert 0.0 <= evr[1] and evr[0] <= 1.0 + 1e-12
    assert evr.sum() <= 1.0 + 1e-12


def test_pca_recovers_dominant_direction():
    """Data stretched along one axis should put almost all variance in PC1."""
    rng = np.random.default_rng(5)
    X = rng.standard_normal((60, 5))
    X[:, 0] *= 50.0
    _, evr = PA.pca(X, k=2)
    assert evr[0] > 0.95, evr


def test_pca_rank_deficient_is_padded():
    """More features than samples: still returns k components without error."""
    rng = np.random.default_rng(6)
    X = rng.standard_normal((3, 500))
    proj, evr = PA.pca(X, k=2)
    assert proj.shape == (3, 2)
    assert evr.shape == (2,)
    assert np.all(np.isfinite(proj)) and np.all(np.isfinite(evr))


# ── silhouette ───────────────────────────────────────────────────────────────

def test_silhouette_hand_computed():
    """
    Four points, two classes:
        A: (0,0) (0,1)      B: (10,0) (10,1)
    For every point a = 1, b = mean(10, sqrt(101)) = 10.0249..., so
    score = (b - a) / b, identical for all four.
    """
    proj = np.array([[0., 0.], [0., 1.], [10., 0.], [10., 1.]])
    labels = np.array(['a', 'a', 'b', 'b'])
    a = 1.0
    b = (10.0 + np.sqrt(101.0)) / 2.0
    expected = (b - a) / max(a, b)
    assert abs(PA.silhouette(proj, labels) - expected) < 1e-12


def test_silhouette_well_separated_beats_overlapping():
    rng = np.random.default_rng(7)
    far = np.vstack([rng.normal(0, .3, (25, 2)), rng.normal(30, .3, (25, 2))])
    near = np.vstack([rng.normal(0, 5, (25, 2)), rng.normal(1, 5, (25, 2))])
    labels = np.array(['a'] * 25 + ['b'] * 25)
    s_far = PA.silhouette(far, labels)
    s_near = PA.silhouette(near, labels)
    assert s_far > 0.9, s_far
    assert s_far > s_near
    assert -1.0 <= s_near <= 1.0


def test_silhouette_edge_cases():
    proj = np.array([[0., 0.], [1., 1.]])
    assert PA.silhouette(proj, np.array(['a', 'a'])) == 0.0     # one class
    assert PA.silhouette(np.zeros((1, 2)), np.array(['a'])) == 0.0
    # singleton classes score 0, so the mean over two singletons is 0
    assert PA.silhouette(proj, np.array(['a', 'b'])) == 0.0


def test_silhouette_identical_points():
    """Coincident points must not divide by zero."""
    proj = np.zeros((4, 2))
    s = PA.silhouette(proj, np.array(['a', 'a', 'b', 'b']))
    assert np.isfinite(s) and s == 0.0


# ── dataset scan ─────────────────────────────────────────────────────────────

def test_scan_dataset():
    root = tempfile.mkdtemp(prefix="pca_scan_")
    try:
        for student, gestures in (("Ana", ("push", "idle")), ("Bo", ("push",))):
            for g in gestures:
                d = os.path.join(root, student, g)
                os.makedirs(d)
                for i in range(2):
                    np.save(os.path.join(d, f"sample_{i:03d}_raw.npy"),
                            np.zeros((2, 3, 4, 4), np.float32))
                    # processed files must be ignored
                    np.save(os.path.join(d, f"sample_{i:03d}.npy"), np.zeros((2, 2)))
        # stray checkpoint dir must be ignored
        os.makedirs(os.path.join(root, "model"))
        os.makedirs(os.path.join(root, "models", "run_v1"))

        paths, labels = PA.scan_dataset(root)
        assert len(paths) == 6, len(paths)
        assert all(p.endswith("_raw.npy") for p in paths)
        assert sorted(set(labels)) == ["idle", "push"]
        assert labels.count("push") == 4 and labels.count("idle") == 2
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_scan_missing_root():
    assert PA.scan_dataset("/nonexistent/path/xyz") == ([], [])


# ── end to end ───────────────────────────────────────────────────────────────

def test_distinct_gestures_separate():
    """
    Two clearly different gestures should give a positive silhouette in at
    least one representation — the whole point of the feature.
    """
    cubes, labels = [], []
    for i in range(5):
        cubes.append(make_cube(beat=6.0, dopp=0.05, seed=i))
        labels.append("push")
    for i in range(5):
        cubes.append(make_cube(beat=20.0, dopp=-0.05, seed=100 + i))
        labels.append("swipe")

    spec = np.vstack([PA.spectrogram_features(c) for c in cubes])
    rf = np.vstack([PA.range_fft_features(c) for c in cubes])
    labels = np.array(labels)

    sp, sv = PA.pca(spec, 2)
    rp, rv = PA.pca(rf, 2)
    s_sil = PA.silhouette(sp, labels)
    r_sil = PA.silhouette(rp, labels)

    assert sp.shape == (10, 2) and rp.shape == (10, 2)
    assert max(s_sil, r_sil) > 0.3, (s_sil, r_sil)
    for v in (sv, rv):
        assert v[0] >= v[1] and v.sum() <= 1.0 + 1e-12


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
    print("\nAll checks passed.")
