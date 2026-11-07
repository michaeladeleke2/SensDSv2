"""
core/previous_picture.py

The spectrogram as SensDS drew it before the change note of 2026-10-19, for
the starter model: its training pictures were drawn that way, and a model
does best on pictures like the ones it learned from. The web version does the
same (src/test/previous_picture.js).

What the change note changed, and how this undoes it for the model only:
  range bin     back to the previous rule (the bin with the largest dB sum
                over Doppler, from bin 0), then the same 5-frame median.
  colour scale  back to a floor of -20 dB, or 40 dB below the loudest cell
                when nothing reaches it (Reduce noise off).
  chirp         the radar now sends a chirp every 0.3 ms instead of 0.2 ms,
                so the same 512 Doppler bins cover 1/1.5 of the speeds they
                used to. That is set on the radar and is not changed here;
                instead each column is stretched back onto the previous speed
                range: previous bin k shows speed (k - 256) x 1.5 of a current
                bin, read from the current bins by linear interpolation.
                Speeds the current chirp cannot see (above 4.1 m/s, the outer
                sixth at each end) are filled with that column's median, its
                background level.
Nothing else in the app changes: Visualize, Collect, Analysis and a student's
own models keep the current drawing. The reference script
(doppler_spectrogram_live.py) is used as it is; only its range bin rule and
colour limits are replaced here.
"""

import numpy as np

from core.physical_features import CHIRP_REP_S
from core.reference_image import REF_ANTENNA, TRAINING_SIZE, import_reference

PREVIOUS_CHIRP_S = 0.0002
PREVIOUS_JET_VMIN = -20.0


def speed_ratio(chirp_s: float = CHIRP_REP_S) -> float:
    """How many times more speed a Doppler bin covered with the previous chirp."""
    return round(chirp_s / PREVIOUS_CHIRP_S, 9)


def select_range_bin_raw_previous(rdm_db: np.ndarray) -> int:
    """The previous rule: argmax over range of the dB sum over Doppler."""
    return int(np.argmax(rdm_db.sum(axis=1)))


def jet_clim_previous(plot_data: np.ndarray, jet_vmin: float):
    """The previous colour limits."""
    vmax = float(np.nanmax(plot_data))
    vmin = jet_vmin if jet_vmin < vmax else vmax - 40.0
    return vmin, vmax


def compute_recorded_previous(data: np.ndarray, antenna: int = REF_ANTENNA):
    """compute_recorded with the previous range bin rule."""
    ref = import_reference()
    data = np.asarray(data)
    n_frame, _, n_chirp, n_sample = data.shape
    range_fft_size, doppler_fft_size = n_sample * 4, n_chirp * 4
    n_range_bins = range_fft_size // 2
    range_window, doppler_window = ref._range_doppler_windows(n_sample, n_chirp)
    rdm = [ref.compute_rdm_db(data[i, antenna], range_window=range_window,
                              doppler_window=doppler_window,
                              range_fft_size=range_fft_size,
                              doppler_fft_size=doppler_fft_size)
           for i in range(n_frame)]
    raw = np.array([select_range_bin_raw_previous(r) for r in rdm], dtype=np.int32)
    bins = ref.smooth_range_bins(raw, n_range_bins, ref.SMOOTH_WINDOW)
    return np.stack([rdm[i][bins[i]] for i in range(n_frame)]), bins


def to_previous_speed_range(spectrogram: np.ndarray, ratio: float | None = None) -> np.ndarray:
    """spectrogram (n_frame, D), zero speed at bin D/2, onto the previous speed range."""
    ratio = speed_ratio() if ratio is None else ratio
    spec = np.asarray(spectrogram, dtype=np.float64)
    if ratio == 1:
        return spec.copy()
    D = spec.shape[1]
    half = D // 2
    x = half + (np.arange(D) - half) * ratio
    inside = (x >= 0) & (x <= D - 1)
    out = np.empty_like(spec)
    for f, row in enumerate(spec):
        out[f] = np.median(row)
        out[f, inside] = np.interp(x[inside], np.arange(D), row)
    return out


def previous_rgb(spectrogram: np.ndarray) -> np.ndarray:
    """reference_rgb with the previous colour limits, (D, n_frame, 3) uint8."""
    from matplotlib import colormaps, colors
    plot_data = np.asarray(spectrogram, dtype=np.float64).T
    vmin, vmax = jet_clim_previous(plot_data, PREVIOUS_JET_VMIN)
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    return colormaps["jet"](norm(plot_data), bytes=True)[..., :3]


def previous_image(frames, ratio: float | None = None):
    """
    The model's picture from frames (n_frame, n_ant, n_chirp, n_sample), as
    training_image(reference_spectrogram(frames)) makes it, drawn the
    previous way. One PIL RGB image, TRAINING_SIZE.
    """
    from PIL import Image
    spectrogram, _ = compute_recorded_previous(frames)
    rgb = np.ascontiguousarray(previous_rgb(to_previous_speed_range(spectrogram, ratio)))
    return Image.fromarray(rgb, "RGB").resize(TRAINING_SIZE, Image.BILINEAR)
