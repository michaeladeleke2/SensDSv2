import os, glob
import numpy as np
from scipy import signal
from scipy.ndimage import zoom, median_filter

DATA_ROOT = os.path.expanduser("~/SensDSv2_data")
ANTENNA = 0
TARGET = (32, 32)
CLIP = 1e-6
SMOOTH = 7


def range_fft(cube, antenna=ANTENNA):
    n_frame, n_ant, n_chirp, n_sample = cube.shape
    rfft_size = n_sample * 4
    n_bins = rfft_size // 2
    win = signal.windows.blackmanharris(n_sample)
    out = np.zeros((n_frame, n_bins, n_chirp), dtype=np.complex128)
    for f in range(n_frame):
        frame = cube[f, antenna].astype(np.float64)
        for c in range(n_chirp):
            x = (frame[c] - frame[c].mean()) * win
            buf = np.zeros(rfft_size, dtype=np.complex128)
            buf[:n_sample] = x
            out[f, :, c] = np.fft.fft(buf)[:n_bins]
    return out


def doppler_db(rfft, n_chirp):
    n_frame, n_bins, _ = rfft.shape
    dfft_size = n_chirp * 4
    win = signal.windows.chebwin(n_chirp, at=100.0)
    win = win / win.sum()
    clip_db = 20.0 * np.log10(CLIP)
    cube = np.zeros((n_frame, n_bins, dfft_size))
    for f in range(n_frame):
        for r in range(n_bins):
            st = (rfft[f, r] - rfft[f, r].mean()) * win
            buf = np.zeros(dfft_size, dtype=np.complex128)
            buf[:n_chirp] = st
            p = np.abs(np.fft.fftshift(np.fft.fft(buf))) ** 2
            db = np.full(dfft_size, clip_db)
            m = p >= CLIP ** 2
            db[m] = 10.0 * np.log10(p[m])
            cube[f, r] = db
    return cube


def extract_spectrogram(rdm):
    n_frame, n_bins, dfft = rdm.shape
    peak = np.argmax(rdm.sum(axis=2), axis=1).astype(float)
    smooth = median_filter(peak, size=SMOOTH)
    bins = np.clip(np.round(smooth), 0, n_bins - 1).astype(int)
    return np.array([rdm[i, bins[i], :] for i in range(n_frame)])


def resize_flat(a, shape=TARGET):
    z = (shape[0] / a.shape[0], shape[1] / a.shape[1])
    return zoom(a, z, order=1).flatten()


def pca(X, k=2):
    mean = X.mean(axis=0)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    proj = Xc @ Vt[:k].T
    var = (S ** 2) / (S ** 2).sum()
    return proj, var[:k]


def silhouette(proj, labels):
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return 0.0
    scores = []
    for i in range(len(proj)):
        same = proj[(labels == labels[i])]
        same = same[~np.all(same == proj[i], axis=1)]
        if len(same) == 0:
            continue
        a = np.linalg.norm(same - proj[i], axis=1).mean()
        b = min(np.linalg.norm(proj[labels == u] - proj[i], axis=1).mean()
                for u in uniq if u != labels[i])
        scores.append((b - a) / max(a, b))
    return float(np.mean(scores)) if scores else 0.0


print("Loading samples...")
files, labels = [], []
for student in sorted(os.listdir(DATA_ROOT)):
    sdir = os.path.join(DATA_ROOT, student)
    if not os.path.isdir(sdir) or student in ("models", "results"):
        continue
    for gesture in sorted(os.listdir(sdir)):
        gdir = os.path.join(sdir, gesture)
        if not os.path.isdir(gdir):
            continue
        for f in sorted(glob.glob(os.path.join(gdir, "*_raw.npy"))):
            files.append(f)
            labels.append(gesture)

labels = np.array(labels)
print(f"{len(files)} samples, classes: {np.unique(labels)}\n")

spec_feats, rfft_feats = [], []
for i, f in enumerate(files):
    print(f"  [{i+1}/{len(files)}] {os.path.basename(os.path.dirname(f))}/{os.path.basename(f)}")
    cube = np.load(f)
    n_chirp = cube.shape[2]
    rf = range_fft(cube)
    rdm = doppler_db(rf, n_chirp)
    spec_feats.append(resize_flat(extract_spectrogram(rdm)))
    profile = rf.mean(axis=2)
    small = zoom(np.abs(profile), (TARGET[0] / profile.shape[0], TARGET[1] / profile.shape[1]), order=1)
    ang = zoom(np.angle(profile), (TARGET[0] / profile.shape[0], TARGET[1] / profile.shape[1]), order=1)
    rfft_feats.append(np.concatenate([small.flatten(), ang.flatten()]))

spec_X = np.array(spec_feats)
rfft_X = np.array(rfft_feats)

print("\n" + "=" * 55)
for name, X in [("SPECTROGRAM (dB)", spec_X), ("RANGE FFT (complex)", rfft_X)]:
    proj, var = pca(X, 2)
    sil = silhouette(proj, labels)
    print(f"\n{name}")
    print(f"  feature dim:     {X.shape[1]}")
    print(f"  PC1 variance:    {var[0]:.1%}")
    print(f"  PC2 variance:    {var[1]:.1%}")
    print(f"  silhouette:      {sil:+.3f}")
    for g in np.unique(labels):
        p = proj[labels == g]
        print(f"    {g:14s} centroid ({p[:,0].mean():8.1f}, {p[:,1].mean():8.1f})")
print("=" * 55)