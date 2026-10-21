#!/usr/bin/env python3
"""
Range–time map from a recorded FMCW cube.

One image for the whole file: time on x, range on y. The range FFT matches
microdoppler/doppler_spectrogram_live.py (mean removal, Blackman–Harris,
4× zero-pad, positive half kept).

Modes:
    mti  (default)  Doppler FFT, then sum linear power at each range
                    excluding the zero-velocity bin. Stationary returns drop out.
    raw             Mean range-FFT magnitude across chirps. Walls stay visible.

CLI:
    python range_map.py --input recording.npy
    python range_map.py --input recording.npy --mode raw --antenna 0

Live stream from the radar (same chirp as the Doppler live script):
    python range_map_live.py
    python range_map_live.py --until-close
    python range_map_live.py --mode raw --nframes 200

Input shape: (n_frame, n_ant, n_chirp, n_sample)
Dependencies: numpy, scipy, matplotlib
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy import signal

PathLike = Union[str, Path]

C_MPS = 299_792_458.0
CLIPPING_VALUE = 1e-6
SPECT_THRESHOLD = 1e-6

# Same chirp as microdoppler/doppler_spectrogram_live.py.
# Edit these if a recording used a different ramp.
START_FREQUENCY_HZ = 58.0e9
END_FREQUENCY_HZ = 63.5e9
SAMPLE_RATE_HZ = 2.0e6
FRAME_REPETITION_TIME_S = 0.1


def _windows(n_sample: int, n_chirp: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        range_window = signal.blackmanharris(n_sample)
        doppler_window = signal.chebwin(n_chirp, at=100.0)
    except AttributeError:
        range_window = signal.windows.blackmanharris(n_sample)
        doppler_window = signal.windows.chebwin(n_chirp, at=100.0)
    doppler_window = doppler_window / np.sum(doppler_window)
    return range_window, doppler_window


def range_axis_m(
    n_range_bins: int,
    range_fft_size: int,
    *,
    sample_rate_hz: float,
    start_frequency_hz: float,
    end_frequency_hz: float,
    n_sample: int,
) -> np.ndarray:
    """DFT bin centers in meters.

    bin k → beat frequency k * Fs / Nfft
    range  → beat frequency * c / (2 * slope)
    slope  = bandwidth / (n_sample / Fs)
    """
    bandwidth_hz = end_frequency_hz - start_frequency_hz
    if bandwidth_hz <= 0:
        raise ValueError("end frequency must be above start frequency")
    chirp_time_s = n_sample / sample_rate_hz
    slope_hz_s = bandwidth_hz / chirp_time_s
    bins = np.arange(n_range_bins, dtype=np.float64)
    beat_hz = bins * sample_rate_hz / range_fft_size
    return beat_hz * C_MPS / (2.0 * slope_hz_s)


def _to_db(linear: np.ndarray, *, power: bool) -> np.ndarray:
    """20*log10(magnitude) or 10*log10(power), with the Doppler-script floor."""
    if power:
        floor = SPECT_THRESHOLD**2
        scale = 10.0
    else:
        floor = CLIPPING_VALUE
        scale = 20.0
    db = np.full(linear.shape, scale * np.log10(floor), dtype=np.float64)
    above = linear >= floor
    db[above] = scale * np.log10(linear[above])
    return db


def _range_spectrum(
    frame: np.ndarray,
    range_window: np.ndarray,
    range_fft_size: int,
) -> np.ndarray:
    """(n_chirp, n_sample) → complex range spectrum (n_range_bins, n_chirp)."""
    n_chirp, n_sample = frame.shape
    n_range_bins = range_fft_size // 2
    x = frame.astype(np.float64, copy=False)
    x = x - np.mean(x, axis=1, keepdims=True)
    x = x * range_window
    buf = np.zeros((n_chirp, range_fft_size), dtype=np.complex128)
    buf[:, :n_sample] = x
    return np.fft.fft(buf, axis=1)[:, :n_range_bins].T


def _profile_raw(range_spectrum: np.ndarray) -> np.ndarray:
    """Mean |range FFT| across chirps, in dB. Static reflections stay."""
    magnitude = np.mean(np.abs(range_spectrum), axis=1)
    return _to_db(magnitude, power=False)


def _profile_mti(
    range_spectrum: np.ndarray,
    doppler_window: np.ndarray,
    doppler_fft_size: int,
) -> np.ndarray:
    """Moving-target range profile in dB.

    Slow-time mean removal, Chebyshev window, 4× Doppler FFT, fftshift.
    Sum linear power over every Doppler bin except the center (zero velocity).
    """
    n_range_bins, n_chirp = range_spectrum.shape
    slow = range_spectrum - np.mean(range_spectrum, axis=1, keepdims=True)
    slow = slow * doppler_window
    buf = np.zeros((n_range_bins, doppler_fft_size), dtype=np.complex128)
    buf[:, :n_chirp] = slow
    shifted = np.fft.fftshift(np.fft.fft(buf, axis=1), axes=1)
    power = np.abs(shifted) ** 2
    dc = doppler_fft_size // 2
    moving = np.sum(power, axis=1) - power[:, dc]
    return _to_db(moving, power=True)


def frame_profile(
    frame: np.ndarray,
    *,
    mode: str,
    range_window: np.ndarray,
    doppler_window: np.ndarray,
    range_fft_size: int,
    doppler_fft_size: int,
) -> np.ndarray:
    """One frame (n_chirp, n_sample) → range profile in dB (n_range_bins,)."""
    if mode not in ("mti", "raw"):
        raise ValueError(f"mode must be 'mti' or 'raw', got {mode!r}")
    spectrum = _range_spectrum(frame, range_window, range_fft_size)
    if mode == "raw":
        return _profile_raw(spectrum)
    return _profile_mti(spectrum, doppler_window, doppler_fft_size)


def compute_range_map(
    data: np.ndarray,
    *,
    antenna: int = 0,
    mode: str = "mti",
    sample_rate_hz: float = SAMPLE_RATE_HZ,
    start_frequency_hz: float = START_FREQUENCY_HZ,
    end_frequency_hz: float = END_FREQUENCY_HZ,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a range–time map for every frame in the recording.

    Returns
    -------
    range_map_db : (n_frame, n_range_bins)
    range_m : (n_range_bins,) bin centers in meters
    """
    if data.ndim != 4:
        raise ValueError(
            f"Expected (n_frame, n_ant, n_chirp, n_sample), got {data.shape}"
        )
    if mode not in ("mti", "raw"):
        raise ValueError(f"mode must be 'mti' or 'raw', got {mode!r}")

    n_frame, n_ant, n_chirp, n_sample = data.shape
    if antenna < 0 or antenna >= n_ant:
        raise ValueError(f"antenna {antenna} out of range (n_ant={n_ant})")

    range_fft_size = n_sample * 4
    doppler_fft_size = n_chirp * 4
    n_range_bins = range_fft_size // 2
    range_window, doppler_window = _windows(n_sample, n_chirp)

    range_map_db = np.empty((n_frame, n_range_bins), dtype=np.float64)
    for frame_idx in range(n_frame):
        range_map_db[frame_idx] = frame_profile(
            data[frame_idx, antenna],
            mode=mode,
            range_window=range_window,
            doppler_window=doppler_window,
            range_fft_size=range_fft_size,
            doppler_fft_size=doppler_fft_size,
        )

    range_m = range_axis_m(
        n_range_bins,
        range_fft_size,
        sample_rate_hz=sample_rate_hz,
        start_frequency_hz=start_frequency_hz,
        end_frequency_hz=end_frequency_hz,
        n_sample=n_sample,
    )
    return range_map_db, range_m


def _jet_clim(plot_data: np.ndarray, jet_vmin: float) -> Tuple[float, float]:
    vmax = float(np.nanmax(plot_data))
    vmin = jet_vmin if jet_vmin < vmax else vmax - 40.0
    return vmin, vmax


def plot_range_map(
    range_map_db: np.ndarray,
    range_m: np.ndarray,
    *,
    duration_s: float,
    jet_vmin: float = -20.0,
    title: str = "Range map",
) -> Figure:
    """Jet image of the full recording. Row 0 (nearest range) is at the bottom."""
    plot_data = np.asarray(range_map_db, dtype=np.float64).T
    if range_m.size > 1:
        dr = float(range_m[1] - range_m[0])
    else:
        dr = 0.0
    range_near = float(range_m[0] - 0.5 * dr)
    range_far = float(range_m[-1] + 0.5 * dr)

    vmin, vmax = _jet_clim(plot_data, jet_vmin)
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    image = ax.imshow(
        plot_data,
        cmap="jet",
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True),
        aspect="auto",
        origin="lower",
        extent=[0.0, duration_s, range_near, range_far],
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("range (m)")
    ax.set_title(title)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.7, color="white", alpha=0.55)
    ax.set_axisbelow(False)
    fig.colorbar(image, ax=ax, label="dB")
    return fig


def run(
    input_path: PathLike,
    *,
    antenna: int = 0,
    mode: str = "mti",
    frame_repetition_time_s: float = FRAME_REPETITION_TIME_S,
    jet_vmin: float = -20.0,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
    start_frequency_hz: float = START_FREQUENCY_HZ,
    end_frequency_hz: float = END_FREQUENCY_HZ,
    show_plot: bool = True,
    save_path: PathLike | None = None,
) -> Figure:
    data = np.load(input_path)
    range_map_db, range_m = compute_range_map(
        data,
        antenna=antenna,
        mode=mode,
        sample_rate_hz=sample_rate_hz,
        start_frequency_hz=start_frequency_hz,
        end_frequency_hz=end_frequency_hz,
    )

    n_frame = data.shape[0]
    duration_s = n_frame * frame_repetition_time_s
    title = "Range map (moving)" if mode == "mti" else "Range map (raw)"
    fig = plot_range_map(
        range_map_db,
        range_m,
        duration_s=duration_s,
        jet_vmin=jet_vmin,
        title=title,
    )

    print(f"Input: {input_path}")
    print(f"Shape: {data.shape}, antenna: {antenna}, mode: {mode}")
    print(
        f"Duration: {duration_s:.3f} s "
        f"({n_frame} frames × {frame_repetition_time_s} s)"
    )
    print(
        f"Range axis: {range_m[0]:.3f}–{range_m[-1]:.3f} m "
        f"({range_m.size} bins, {float(range_m[1] - range_m[0]) * 1000:.2f} mm spacing)"
    )
    print(
        f"Chirp: {start_frequency_hz / 1e9:.2f}–{end_frequency_hz / 1e9:.2f} GHz, "
        f"fs={sample_rate_hz / 1e6:.1f} MHz"
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    if show_plot:
        plt.show(block=True)
    else:
        plt.close(fig)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Range–time map from a recorded FMCW .npy cube"
    )
    parser.add_argument("--input", type=Path, required=True, help=".npy recording")
    parser.add_argument("--antenna", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("mti", "raw"),
        default="mti",
        help="mti: moving targets only (default); raw: include stationary reflections",
    )
    parser.add_argument(
        "--frame-repetition-time-s",
        type=float,
        default=FRAME_REPETITION_TIME_S,
        help="Seconds per frame for the time axis (default: 0.1)",
    )
    parser.add_argument("--jet-vmin", type=float, default=-20.0)
    parser.add_argument("--start-frequency-hz", type=float, default=START_FREQUENCY_HZ)
    parser.add_argument("--end-frequency-hz", type=float, default=END_FREQUENCY_HZ)
    parser.add_argument("--sample-rate-hz", type=float, default=SAMPLE_RATE_HZ)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    run(
        args.input,
        antenna=args.antenna,
        mode=args.mode,
        frame_repetition_time_s=args.frame_repetition_time_s,
        jet_vmin=args.jet_vmin,
        sample_rate_hz=args.sample_rate_hz,
        start_frequency_hz=args.start_frequency_hz,
        end_frequency_hz=args.end_frequency_hz,
        show_plot=not args.no_show,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
