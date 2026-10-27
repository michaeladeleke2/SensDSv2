"""
core/physical_features.py

Extracts interpretable physical quantities — distance, speed, acceleration,
Doppler spread — from the raw radar cubes saved by the Collect tab, and writes
them as CSV for external tools such as CODAP.

Two shapes are produced from the same extraction:

  summary   one row per gesture sample; the natural shape for scatterplots
  frames    one row per radar frame; keeps the trajectory so speed can be
            plotted against time within a single gesture

Everything here is derived from the radar configuration in core/radar.py, so
the numbers carry real units rather than arbitrary feature indices.

Doppler velocities depend on the chirp spacing the sample was recorded with.
The radar now uses 0.3 ms (the visualization change note; it was 0.2 ms), so
the Collect tab records it in capture_info.json and samples are read with the
spacing their folder says, or the current one when the folder does not say.
"""

import csv
import glob
import json
import os

import numpy as np
from scipy.signal.windows import blackmanharris, chebwin

DATA_ROOT = os.path.join(os.path.expanduser("~"), "SensDSv2_data")
EXCLUDE_DIRS = {"models", "model", "predict_temp", "temp", "test", "checkpoints"}

# ── Radar constants (must match core/radar.py build_config) ───────────────────
C = 3e8
F_START = 58.0e9
F_END = 63.5e9
BANDWIDTH = F_END - F_START                 # 5.5 GHz
F_CENTER = (F_START + F_END) / 2
WAVELENGTH = C / F_CENTER
CHIRP_REP_S = 0.0003                        # the radar's chirp spacing now (was 0.0002)
PRF = 1.0 / CHIRP_REP_S                     # 3.33 kHz
FRAME_S = 0.10                              # 10 fps
MAX_VELOCITY = PRF * WAVELENGTH / 4         # ±4.11 m/s


def max_velocity(chirp_rep_s: float = CHIRP_REP_S) -> float:
    """The Doppler axis edge, m/s, for a chirp spacing in seconds."""
    return WAVELENGTH / (4.0 * chirp_rep_s)
RANGE_RES_M = C / (2 * BANDWIDTH)           # 2.73 cm
N_SAMPLES_CFG = 256
MAX_RANGE_M = N_SAMPLES_CFG * C / (4 * BANDWIDTH)   # 3.49 m

RANGE_FFT_MULT = 4
DOPPLER_FFT_MULT = 4
CLIP_VALUE = 1e-6

# A frame counts as "moving" when the tracked speed exceeds this. Tuned so an
# idle capture reports near-zero motion rather than counting noise crossings.
MOTION_SPEED_MS = 0.25

# The micro-Doppler signature of a frame: the Doppler bins at least
# SIGNATURE_DB above the frame's own noise floor (its median), so receiver
# noise never counts and a frame with nothing moving has no signature, and
# within SIGNATURE_PEAK_DB of the frame's strongest bin, so the faint skirts of
# a strong return do not stretch it.
SIGNATURE_DB = 15.0
SIGNATURE_PEAK_DB = 10.0

_RANGE_WIN: dict = {}
_DOPPLER_WIN: dict = {}


def _range_window(n):
    w = _RANGE_WIN.get(n)
    if w is None:
        w = blackmanharris(n).astype(np.float64)
        _RANGE_WIN[n] = w
    return w


def _doppler_window(n):
    w = _DOPPLER_WIN.get(n)
    if w is None:
        v = chebwin(n, at=100.0).astype(np.float64)
        w = v / np.sum(v)
        _DOPPLER_WIN[n] = w
    return w


# ── Feature catalogue ────────────────────────────────────────────────────────
# (key, human label, unit). Order drives the UI dropdowns and the CSV columns.

# Everything summarize() and frame_series() compute. summarize() still returns
# all of these — only the exposed lists below drive the UI dropdowns and the
# CSV columns, so re-enabling one is a matter of moving it into the list.
ALL_SUMMARY_FEATURES = [
    ("range_mean_m",        "Distance from radar",    "m"),
    ("range_travel_m",      "Distance traveled",      "m"),
    ("range_net_m",         "Net displacement",       "m"),
    ("range_toward_m",      "Movement toward radar",  "m"),
    ("radial_speed_max_ms", "Peak speed (toward/away)",  "m/s"),
    ("radial_speed_mean_ms", "Mean speed (toward/away)", "m/s"),
    ("vel_peak_ms",         "Peak Doppler velocity",  "m/s"),
    ("vel_mean_ms",         "Mean Doppler velocity",  "m/s"),
    ("vel_std_ms",          "Velocity variability",   "m/s"),
    ("vel_span_ms",         "Velocity span",          "m/s"),
    ("accel_peak_ms2",      "Peak acceleration",      "m/s²"),
    ("accel_mean_ms2",      "Mean |acceleration|",    "m/s²"),
    ("doppler_spread_ms",   "Doppler spread",         "m/s"),
    ("doppler_spread_max_ms", "Peak Doppler spread",  "m/s"),
    ("direction_changes",   "Direction changes",      "count"),
    ("motion_duration_s",   "Motion duration",        "s"),
    ("energy_peak",         "Peak return energy",     "a.u."),
    ("energy_mean",         "Mean return energy",     "a.u."),
    ("n_frames",            "Frames captured",        "count"),
]

ALL_FRAME_FEATURES = [
    ("time_s",            "Time",             "s"),
    ("range_m",           "Distance",         "m"),
    ("radial_speed_ms",   "Speed (toward/away)", "m/s"),
    ("velocity_ms",       "Doppler velocity", "m/s"),
    ("speed_ms",          "Doppler speed",    "m/s"),
    ("accel_ms2",         "Acceleration",     "m/s²"),
    ("doppler_spread_ms", "Doppler spread",   "m/s"),
    ("energy",            "Return energy",    "a.u."),
    ("vel_upper_ms",      "Highest velocity in the signature", "m/s"),
    ("vel_lower_ms",      "Lowest velocity in the signature",  "m/s"),
]

# ── Exposed set ──────────────────────────────────────────────────────────────
# Trimmed to distance and speed only: two of each, both measured the same way
# (from how the tracked range changes between frames), so the whole tab can be
# explained in one sentence. "toward/away" is literal — the radar measures
# radial motion, so a purely sideways swipe registers very little speed.
#
# Velocity span (requested by Dr. Hazel) is the one Doppler feature: the
# highest velocity in the gesture's micro-Doppler signature minus the lowest,
# over the whole sample. In each frame the signature is the Doppler bins well
# above that frame's noise and near its strongest return (SIGNATURE_DB,
# SIGNATURE_PEAK_DB); its top and bottom are the frame's highest and lowest
# velocities. It is not mean speed: mean speed averages
# how fast the tracked distance changes, while the span is how wide a range
# of velocities the gesture covers. A push and pull swings from toward to
# away and has a wide span; a hand held still has none.
SUMMARY_FEATURES = [
    ("range_mean_m",         "Distance from radar",       "m"),
    ("range_travel_m",       "Distance traveled",         "m"),
    ("radial_speed_max_ms",  "Peak speed (toward/away)",  "m/s"),
    ("radial_speed_mean_ms", "Mean speed (toward/away)",  "m/s"),
    ("vel_span_ms",          "Velocity span",             "m/s"),
]

FRAME_FEATURES = [
    ("time_s",          "Time",                 "s"),
    ("range_m",         "Distance",             "m"),
    ("radial_speed_ms", "Speed (toward/away)",  "m/s"),
    ("vel_upper_ms",    "Highest velocity in the signature", "m/s"),
    ("vel_lower_ms",    "Lowest velocity in the signature",  "m/s"),
]

SUMMARY_KEYS = [k for k, _, _ in SUMMARY_FEATURES]
FRAME_KEYS = [k for k, _, _ in FRAME_FEATURES]


def feature_label(key: str) -> str:
    for k, label, unit in ALL_SUMMARY_FEATURES + ALL_FRAME_FEATURES:
        if k == key:
            return f"{label} ({unit})" if unit else label
    return key


def feature_unit(key: str) -> str:
    for k, _, unit in ALL_SUMMARY_FEATURES + ALL_FRAME_FEATURES:
        if k == key:
            return unit
    return ""


# ── Extraction ───────────────────────────────────────────────────────────────

def _range_profile(cube: np.ndarray) -> np.ndarray:
    """(n_frame, n_ant, n_chirp, n_sample) -> (n_ant, n_frame, n_bins) complex."""
    n_frame, n_ant, n_chirp, n_sample = cube.shape
    fft_size = n_sample * RANGE_FFT_MULT
    n_bins = fft_size // 2
    win = _range_window(n_sample)

    x = cube.astype(np.float64, copy=False)
    x = x - np.mean(x, axis=-1, keepdims=True)
    x = x * win
    spec = np.fft.fft(x, n=fft_size, axis=-1)[..., :n_bins]   # (fr, ant, chirp, bin)
    prof = spec.mean(axis=2)                                   # average over chirps
    return np.swapaxes(prof, 0, 1)                             # (ant, fr, bin)


def _doppler_at_bins(cube: np.ndarray, bins: np.ndarray, antenna: int = 0,
                     chirp_rep_s: float = CHIRP_REP_S):
    """
    Doppler spectrum for one range bin per frame.

    Only the tracked bin is transformed rather than the whole Range-Doppler
    cube — 30 FFTs instead of 30 x 512, which is what makes extraction fast
    enough to run over a whole dataset.
    """
    n_frame, n_ant, n_chirp, n_sample = cube.shape
    fft_size = n_sample * RANGE_FFT_MULT
    n_bins = fft_size // 2
    dfft = n_chirp * DOPPLER_FFT_MULT

    rwin = _range_window(n_sample)
    dwin = _doppler_window(n_chirp)

    x = cube[:, antenna].astype(np.float64, copy=False)        # (fr, chirp, samp)
    x = x - np.mean(x, axis=-1, keepdims=True)
    x = x * rwin
    rspec = np.fft.fft(x, n=fft_size, axis=-1)[..., :n_bins]   # (fr, chirp, bin)

    idx = np.clip(bins, 0, n_bins - 1)
    slow = rspec[np.arange(n_frame), :, idx]                   # (fr, chirp)
    slow = slow - np.mean(slow, axis=-1, keepdims=True)
    slow = slow * dwin
    spec = np.fft.fftshift(np.fft.fft(slow, n=dfft, axis=-1), axes=-1)

    power = np.abs(spec) ** 2
    floor = 20.0 * np.log10(CLIP_VALUE)
    db = np.full(power.shape, floor, dtype=np.float64)
    above = power >= CLIP_VALUE ** 2
    db[above] = 10.0 * np.log10(power[above])
    vmax = max_velocity(chirp_rep_s)
    return db, np.linspace(-vmax, vmax, dfft)


def frame_series(cube: np.ndarray, chirp_rep_s: float = CHIRP_REP_S) -> dict:
    """Per-frame physical quantities. Returns a dict of equal-length arrays."""
    cube = np.asarray(cube)
    if cube.ndim != 4:
        raise ValueError(f"expected (n_frame, n_ant, n_chirp, n_sample), got {cube.shape}")
    n_frame = cube.shape[0]

    prof = _range_profile(cube)                       # (ant, fr, bin)
    n_bins = prof.shape[2]
    # Static clutter removal: anything constant across frames is not the hand.
    moving = prof - prof.mean(axis=1, keepdims=True)
    mag = np.abs(moving).mean(axis=0)                 # (fr, bin)

    peak_bin = np.argmax(mag, axis=1)
    rng_m = peak_bin * (MAX_RANGE_M / n_bins)
    energy = mag.max(axis=1)

    db, vel_axis = _doppler_at_bins(cube, peak_bin, chirp_rep_s=chirp_rep_s)
    # Energy-weighted Doppler centroid and spread, per frame.
    w = 10.0 ** (db / 10.0)
    w = w - w.min(axis=1, keepdims=True)
    total = w.sum(axis=1, keepdims=True)
    total[total <= 0] = 1.0
    w = w / total
    velocity = (vel_axis[None, :] * w).sum(axis=1)
    spread = np.sqrt((((vel_axis[None, :] - velocity[:, None]) ** 2) * w).sum(axis=1))

    # The signature's top and bottom, NaN in a frame with no signature
    strong = db >= np.maximum(np.median(db, axis=1, keepdims=True) + SIGNATURE_DB,
                              db.max(axis=1, keepdims=True) - SIGNATURE_PEAK_DB)
    has = strong.any(axis=1)
    upper = np.full(n_frame, np.nan)
    lower = np.full(n_frame, np.nan)
    upper[has] = np.max(np.where(strong, vel_axis[None, :], -np.inf), axis=1)[has]
    lower[has] = np.min(np.where(strong, vel_axis[None, :], np.inf), axis=1)[has]

    dt = FRAME_S
    radial = np.zeros(n_frame)
    if n_frame > 1:
        radial[1:] = np.diff(rng_m) / dt
    accel = np.zeros(n_frame)
    if n_frame > 1:
        accel[1:] = np.diff(velocity) / dt

    return {
        "time_s": np.arange(n_frame) * dt,
        "range_m": rng_m,
        "velocity_ms": velocity,
        "speed_ms": np.abs(velocity),
        "radial_speed_ms": radial,
        "accel_ms2": accel,
        "doppler_spread_ms": spread,
        "energy": energy,
        "vel_upper_ms": upper,
        "vel_lower_ms": lower,
    }


def summarize(series: dict) -> dict:
    """Collapse a per-frame series into one row of summary features."""
    rng = series["range_m"]
    vel = series["velocity_ms"]
    radial = series["radial_speed_ms"]
    accel = series["accel_ms2"]
    spread = series["doppler_spread_ms"]
    energy = series["energy"]
    n = len(rng)

    moving = np.abs(vel) > MOTION_SPEED_MS
    if moving.sum() > 1:
        signs = np.sign(vel[moving])
        direction_changes = int(np.sum(np.diff(signs) != 0))
    else:
        direction_changes = 0

    # Negative range change = approaching the radar.
    steps = np.diff(rng) if n > 1 else np.zeros(1)
    toward = float(-steps[steps < 0].sum()) if np.any(steps < 0) else 0.0

    return {
        "range_mean_m": float(rng.mean()),
        "range_travel_m": float(rng.max() - rng.min()),
        "range_net_m": float(rng[-1] - rng[0]) if n > 1 else 0.0,
        "range_toward_m": toward,
        "radial_speed_max_ms": float(np.abs(radial).max()),
        "radial_speed_mean_ms": float(np.abs(radial).mean()),
        "vel_peak_ms": float(np.abs(vel).max()),
        "vel_mean_ms": float(vel.mean()),
        "vel_std_ms": float(vel.std()),
        "vel_span_ms": _span(series),
        "accel_peak_ms2": float(np.abs(accel).max()),
        "accel_mean_ms2": float(np.abs(accel).mean()),
        "doppler_spread_ms": float(spread.mean()),
        "doppler_spread_max_ms": float(spread.max()),
        "direction_changes": float(direction_changes),
        "motion_duration_s": float(moving.sum() * FRAME_S),
        "energy_peak": float(energy.max()),
        "energy_mean": float(energy.mean()),
        "n_frames": float(n),
    }


def _span(series: dict) -> float:
    """Highest minus lowest signature velocity over the sample; 0 with no signature."""
    upper = series.get("vel_upper_ms")
    lower = series.get("vel_lower_ms")
    if upper is None or not np.any(np.isfinite(upper)):
        return 0.0
    return float(np.nanmax(upper) - np.nanmin(lower))


def _csv_value(x) -> str:
    """.6g, with a frame that has no value left blank (CODAP reads it as missing)."""
    return f"{x:.6g}" if np.isfinite(x) else ""


def extract(cube: np.ndarray, chirp_rep_s: float | None = None):
    """Returns (summary_dict, frame_series_dict)."""
    series = frame_series(cube, CHIRP_REP_S if chirp_rep_s is None else chirp_rep_s)
    return summarize(series), series


def chirp_for_folder(folder: str) -> float:
    """
    The chirp spacing a gesture folder's samples were recorded with, from its
    capture_info.json, or the radar's current spacing when it does not say.
    """
    try:
        with open(os.path.join(folder, "capture_info.json")) as f:
            value = float(json.load(f).get("chirp_repetition_time_s", CHIRP_REP_S))
        return value if value > 0 else CHIRP_REP_S
    except Exception:
        return CHIRP_REP_S


# ── Dataset scanning ─────────────────────────────────────────────────────────

def png_for_sample(raw_path: str) -> str | None:
    """
    The training PNG the Collect tab saved beside a raw cube, or None.

    Captures are written as a trio — sample_NNN.npy (the spectrogram),
    sample_NNN_raw.npy (the cube these features come from) and sample_NNN.png
    (the image the model is trained on). Older captures predate the cube, and
    a folder can be pruned by hand, so the PNG is never assumed to exist.
    """
    suffix = "_raw.npy"
    if raw_path.endswith(suffix):
        candidate = raw_path[: -len(suffix)] + ".png"
    else:
        candidate = os.path.splitext(raw_path)[0] + ".png"
    return candidate if os.path.isfile(candidate) else None


def scan_samples(data_root: str | None = None) -> list:
    """Every raw sample as {path, student, gesture, name}, sorted."""
    root = data_root or DATA_ROOT
    out = []
    if not os.path.isdir(root):
        return out
    for student in sorted(os.listdir(root)):
        sdir = os.path.join(root, student)
        if (not os.path.isdir(sdir) or student in EXCLUDE_DIRS
                or student.startswith(".")):
            continue
        for gesture in sorted(os.listdir(sdir)):
            gdir = os.path.join(sdir, gesture)
            if not os.path.isdir(gdir) or gesture.startswith("."):
                continue
            for p in sorted(glob.glob(os.path.join(gdir, "sample_*_raw.npy"))):
                out.append({
                    "path": p,
                    "student": student,
                    "gesture": gesture,
                    "name": os.path.basename(p),
                    "chirp_rep_s": chirp_for_folder(gdir),
                })
    return out


# ── CSV writers ──────────────────────────────────────────────────────────────

SUMMARY_ID_COLS = ["sample_id", "student", "gesture", "file"]
FRAME_ID_COLS = ["sample_id", "student", "gesture", "frame"]


def write_summary_csv(records: list, path: str) -> int:
    """
    One row per gesture sample.

    records: [{"student","gesture","name","summary"}]
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SUMMARY_ID_COLS + SUMMARY_KEYS)
        for i, r in enumerate(records, 1):
            w.writerow(
                [i, r["student"], r["gesture"], r["name"]]
                + [_csv_value(r['summary'][k]) for k in SUMMARY_KEYS]
            )
    return len(records)


def write_frames_csv(records: list, path: str) -> int:
    """
    One row per radar frame, keeping each gesture's trajectory.

    records: [{"student","gesture","name","series"}]
    """
    rows = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FRAME_ID_COLS + FRAME_KEYS)
        for i, r in enumerate(records, 1):
            s = r["series"]
            for j in range(len(s["time_s"])):
                w.writerow(
                    [i, r["student"], r["gesture"], j]
                    + [_csv_value(s[k][j]) for k in FRAME_KEYS]
                )
                rows += 1
    return rows
