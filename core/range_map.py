"""
core/range_map.py

The range map: how far away things are over time.

core/range_map_v1.py and core/range_map_live.py are exact copies of the two
range map scripts and are never edited (tests/test_range_map.py checks their
hashes). This module only imports them and holds the settings the Visualize
and Curve Fit tabs share:

  MTI       on: moving targets only (Doppler power without the zero-velocity
            bin); off: the mean range-FFT magnitude, walls and furniture stay.
            Starts off. RANGE_MAP_MODE below, or the SENSDS_RANGE_MODE
            environment variable (mti or raw), sets the starting value; the
            MTI checkbox changes it from there.
  distance  the part of the range axis shown, in metres. Only the bins inside
            it are drawn and the colours are scaled to them, so a strong still
            object outside it does not wash out the target.

The scripts assume 2 MHz sampling and the radar now samples at 1 MHz; the
sample rate cancels out of the range axis (bin k is k * n_sample / n_fft *
c / (2 * bandwidth)), so 0 to 3.48 m holds for both.
"""

import os
import sys

import numpy as np

# Starting value of the MTI checkbox: "raw" (MTI off) or "mti".
RANGE_MAP_MODE = "raw"

N_SAMPLE = 256
N_CHIRP = 128
RANGE_FFT_SIZE = N_SAMPLE * 4
N_RANGE_BINS = RANGE_FFT_SIZE // 2
FRAME_TIME_S = 0.1
ANTENNA = 0

_listeners = []


def _starting_mode() -> str:
    env = os.environ.get("SENSDS_RANGE_MODE", "").strip().lower()
    return env if env in ("mti", "raw") else RANGE_MAP_MODE


def import_range_map():
    """
    The two scripts, as (range_map_v1, range_map_live).

    range_map_live.py imports range_map_v1 by its bare name, as it does beside
    it on disk; registering the package copy under that name lets it load
    unedited from inside core/.
    """
    import matplotlib
    matplotlib.use("QtAgg", force=False)
    from core import range_map_v1 as v1
    sys.modules.setdefault("range_map_v1", v1)
    from core import range_map_live as live
    return v1, live


def range_axis() -> np.ndarray:
    """Bin centres in metres for the radar's frame (512 bins, 0 to 3.48 m)."""
    v1, _ = import_range_map()
    return v1.range_axis_m(
        N_RANGE_BINS, RANGE_FFT_SIZE,
        sample_rate_hz=v1.SAMPLE_RATE_HZ,
        start_frequency_hz=v1.START_FREQUENCY_HZ,
        end_frequency_hz=v1.END_FREQUENCY_HZ,
        n_sample=N_SAMPLE,
    )


RANGE_MAX_M = 3.48


# ── shared settings ──────────────────────────────────────────────────────────

_settings = {"mti": _starting_mode() == "mti", "min": 0.0, "max": RANGE_MAX_M}


def get_settings() -> dict:
    return dict(_settings)


def mode() -> str:
    return "mti" if _settings["mti"] else "raw"


def set_settings(mti=None, min_m=None, max_m=None):
    """Change MTI and/or the distance shown; listeners hear (settings, changed)."""
    new = dict(_settings)
    if mti is not None:
        new["mti"] = bool(mti)
    if min_m is not None:
        new["min"] = float(min_m)
    if max_m is not None:
        new["max"] = float(max_m)
    new["min"] = max(0.0, min(RANGE_MAX_M - 0.1, new["min"]))
    new["max"] = min(RANGE_MAX_M, max(new["min"] + 0.1, new["max"]))
    changed = {
        "mti": new["mti"] != _settings["mti"],
        "distance": (new["min"], new["max"]) != (_settings["min"], _settings["max"]),
    }
    if not (changed["mti"] or changed["distance"]):
        return
    _settings.update(new)
    for fn in list(_listeners):
        try:
            fn(get_settings(), changed)
        except RuntimeError:
            # A listener's widget was deleted; forget it.
            _listeners.remove(fn)


def on_settings_changed(fn):
    _listeners.append(fn)


def remove_listener(fn):
    if fn in _listeners:
        _listeners.remove(fn)


# ── drawing helpers ──────────────────────────────────────────────────────────

def shown_bins(range_m, min_m=None, max_m=None):
    """(lo, hi) inclusive: the bins whose centres lie inside the distance shown."""
    range_m = np.asarray(range_m)
    lo, hi = 0, range_m.size - 1
    if min_m is not None:
        while lo < hi and range_m[lo] < min_m - 1e-9:
            lo += 1
    if max_m is not None:
        while hi > lo and range_m[hi] > max_m + 1e-9:
            hi -= 1
    return lo, hi


def range_rgb(history, n_bins_lo_hi, jet_vmin):
    """
    The range map as the scripts colour it, (n_frames, n_range_bins, 3) uint8.

    history: (n_frames, n_range_bins) dB. Colour limits come from the script's
    own _jet_clim over the bins inside (lo, hi); matplotlib's jet with
    Normalize(clip=True), as its imshow applies it.
    """
    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    v1, _ = import_range_map()
    lo, hi = n_bins_lo_hi
    history = np.asarray(history, dtype=np.float64)
    vmin, vmax = v1._jet_clim(history[:, lo:hi + 1], jet_vmin)
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    return colormaps["jet"](norm(history), bytes=True)[..., :3]
