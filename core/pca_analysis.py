"""
core/pca_analysis.py

Compares two representations of the same radar capture by projecting both to
2D with PCA and scoring how well each separates the gesture classes.

  Doppler domain — the spectrogram: how fast things moved.
  Range domain   — the range FFT profile: how far away things were.

Raw samples are sample_NNN_raw.npy written by the Collect tab, with shape
(n_frame, n_ant, n_chirp, n_sample) and dtype float32 — the radar returns
real-valued ADC samples, not complex.

No scikit-learn and no matplotlib: PCA is SVD from first principles and the
silhouette score is computed directly.
"""

import glob
import os

import numpy as np
from scipy.ndimage import median_filter, zoom
from scipy.signal.windows import blackmanharris, chebwin

DATA_ROOT = os.path.join(os.path.expanduser("~"), "SensDSv2_data")
EXCLUDE_DIRS = {"models", "model", "predict_temp", "temp", "test", "checkpoints"}

RANGE_FFT_MULT = 4
DOPPLER_FFT_MULT = 4
CLIP_VALUE = 1e-6
SMOOTH_WINDOW = 7
FEATURE_SIZE = 32

_RANGE_WIN_CACHE: dict = {}
_DOPPLER_WIN_CACHE: dict = {}


def _range_window(n_sample: int) -> np.ndarray:
    win = _RANGE_WIN_CACHE.get(n_sample)
    if win is None:
        win = blackmanharris(n_sample).astype(np.float64)
        _RANGE_WIN_CACHE[n_sample] = win
    return win


def _doppler_window(n_chirp: int) -> np.ndarray:
    """chebwin is expensive to build, so cache it; normalized to unit sum."""
    win = _DOPPLER_WIN_CACHE.get(n_chirp)
    if win is None:
        w = chebwin(n_chirp, at=100.0).astype(np.float64)
        win = w / np.sum(w)
        _DOPPLER_WIN_CACHE[n_chirp] = win
    return win


def range_fft(cube: np.ndarray, antenna: int = 0) -> np.ndarray:
    """(n_frame, n_ant, n_chirp, n_sample) -> (n_frame, n_bins, n_chirp) complex."""
    cube = np.asarray(cube)
    if cube.ndim != 4:
        raise ValueError(f"expected 4-D cube, got {cube.shape}")
    n_frame, n_ant, n_chirp, n_sample = cube.shape
    if antenna >= n_ant:
        raise ValueError(f"antenna {antenna} out of range (n_ant={n_ant})")

    fft_size = n_sample * RANGE_FFT_MULT
    n_bins = fft_size // 2
    rw = _range_window(n_sample)

    x = cube[:, antenna].astype(np.float64, copy=False)
    x = x - np.mean(x, axis=-1, keepdims=True)
    x = x * rw
    spec = np.fft.fft(x, n=fft_size, axis=-1)[..., :n_bins]
    return np.swapaxes(spec, 1, 2)


def doppler_db(rfft: np.ndarray, n_chirp: int, chunk: int = 4) -> np.ndarray:
    """(n_frame, n_bins, n_chirp) -> (n_frame, n_bins, dfft_size) dB."""
    n_frame, n_bins, _ = rfft.shape
    dfft_size = n_chirp * DOPPLER_FFT_MULT
    dw = _doppler_window(n_chirp)

    floor_db = 20.0 * np.log10(CLIP_VALUE)
    thr = CLIP_VALUE ** 2
    out = np.empty((n_frame, n_bins, dfft_size), dtype=np.float32)

    # Chunked over frames: the complex intermediate is ~4 MB per frame at the
    # app's 128x256 config, so a full batch would allocate far more than needed.
    for start in range(0, n_frame, chunk):
        stop = min(start + chunk, n_frame)
        block = rfft[start:stop]
        slow = block - np.mean(block, axis=-1, keepdims=True)
        slow = slow * dw
        spec = np.fft.fftshift(np.fft.fft(slow, n=dfft_size, axis=-1), axes=-1)
        power = np.abs(spec) ** 2
        db = np.full(power.shape, floor_db, dtype=np.float64)
        above = power >= thr
        db[above] = 10.0 * np.log10(power[above])
        out[start:stop] = db
    return out


def extract_spectrogram(rdm: np.ndarray) -> np.ndarray:
    """(n_frame, n_bins, dfft) -> (n_frame, dfft), tracking the strongest range bin."""
    n_frame, n_bins, _ = rdm.shape
    energy = np.argmax(rdm.sum(axis=2), axis=1).astype(np.float64)
    smoothed = median_filter(energy, size=max(1, SMOOTH_WINDOW | 1))
    bins = np.clip(np.round(smoothed), 0, n_bins - 1).astype(np.int32)
    return rdm[np.arange(n_frame), bins, :]


def _resize(arr: np.ndarray, size: int = FEATURE_SIZE) -> np.ndarray:
    """Bilinear resize to exactly (size, size)."""
    arr = np.asarray(arr, dtype=np.float64)
    h, w = arr.shape
    if (h, w) == (size, size):
        return arr
    out = zoom(arr, (size / h, size / w), order=1)
    # zoom can land one pixel off; force the exact shape.
    if out.shape != (size, size):
        fixed = np.zeros((size, size), dtype=np.float64)
        r = min(size, out.shape[0])
        c = min(size, out.shape[1])
        fixed[:r, :c] = out[:r, :c]
        out = fixed
    return out


def spectrogram_features(cube: np.ndarray) -> np.ndarray:
    """Doppler-domain feature: resized dB spectrogram, 1024-dim."""
    n_chirp = cube.shape[2]
    rfft = range_fft(cube)
    rdm = doppler_db(rfft, n_chirp)
    spec = extract_spectrogram(rdm)
    return _resize(spec).ravel()


def range_fft_features(cube: np.ndarray) -> np.ndarray:
    """Range-domain feature: chirp-averaged profile, magnitude + phase, 2048-dim."""
    rfft = range_fft(cube)
    profile = rfft.mean(axis=2)                     # (n_frame, n_bins) complex
    mag = _resize(np.abs(profile)).ravel()
    phase = _resize(np.angle(profile)).ravel()
    return np.concatenate([mag, phase])


def pca(X: np.ndarray, k: int = 2):
    """
    PCA via SVD on the mean-centered matrix.

    A covariance matrix is never formed — with 1024/2048-dim features and only
    a couple dozen samples that would be a 2048x2048 matrix built from far too
    few points, and squaring the data costs precision. SVD on the centered
    matrix gives the same components directly.

    Returns (proj (n, k), explained_variance_ratio (k,)).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected 2-D feature matrix, got {X.shape}")
    n = X.shape[0]
    Xc = X - X.mean(axis=0, keepdims=True)

    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    total = float(np.sum(S ** 2))
    evr = (S ** 2 / total) if total > 0 else np.zeros_like(S)

    k_eff = min(k, S.shape[0])
    proj = U[:, :k_eff] * S[:k_eff]
    if k_eff < k:                                   # pad if rank-deficient
        proj = np.hstack([proj, np.zeros((n, k - k_eff))])
        evr = np.concatenate([evr, np.zeros(k - evr.shape[0])])
    return proj, evr[:k]


def silhouette(proj: np.ndarray, labels) -> float:
    """
    Mean silhouette score, computed directly.

    Per point: a = mean distance to others in its own class, b = the smallest
    mean distance to any other class, score = (b - a) / max(a, b).
    Points in a class of size 1 score 0 by convention.
    Returns 0.0 when there are fewer than two classes.
    """
    proj = np.asarray(proj, dtype=np.float64)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    n = proj.shape[0]
    if classes.shape[0] < 2 or n < 2:
        return 0.0

    diff = proj[:, None, :] - proj[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))

    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        own = labels == labels[i]
        own_count = int(own.sum())
        if own_count <= 1:
            scores[i] = 0.0
            continue
        a = (dist[i][own].sum()) / (own_count - 1)   # exclude self (dist 0)

        b = np.inf
        for c in classes:
            if c == labels[i]:
                continue
            mask = labels == c
            if not mask.any():
                continue
            b = min(b, float(dist[i][mask].mean()))
        if not np.isfinite(b):
            scores[i] = 0.0
            continue

        denom = max(a, b)
        scores[i] = 0.0 if denom == 0 else (b - a) / denom
    return float(scores.mean())


def scan_dataset(data_root: str | None = None):
    """Return (paths, labels) for every sample_*_raw.npy under data_root."""
    root = data_root or DATA_ROOT
    paths: list[str] = []
    labels: list[str] = []
    if not os.path.isdir(root):
        return paths, labels

    for student in sorted(os.listdir(root)):
        sdir = os.path.join(root, student)
        if not os.path.isdir(sdir) or student in EXCLUDE_DIRS or student.startswith("."):
            continue
        for gesture in sorted(os.listdir(sdir)):
            gdir = os.path.join(sdir, gesture)
            if not os.path.isdir(gdir) or gesture.startswith("."):
                continue
            for p in sorted(glob.glob(os.path.join(gdir, "sample_*_raw.npy"))):
                paths.append(p)
                labels.append(gesture)
    return paths, labels
