#!/usr/bin/env python3
"""
Live / recorded Doppler spectrogram — Infineon SDK style (microdoppler viz).

Same DSP and plot style as doppler_spectrogram.py (time on x, velocity on y,
jet colormap, median-smoothed range-bin selection), with an optional live
stream from BGT60TR13C via ifxradarsdk.

CLI:
    python doppler_spectrogram_live.py --mode recorded --input recording.npy
    python doppler_spectrogram_live.py --mode live --nframes 200

Dependencies: numpy, scipy, matplotlib
Live mode also requires: ifxradarsdk
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from scipy import signal
from scipy.ndimage import median_filter

PathLike = Union[str, Path]

CLIPPING_VALUE = 1e-6
SPECT_THRESHOLD = 1e-6
SMOOTH_WINDOW = 5

# ---------------------------------------------------------------------------
# Radar configuration (live acquisition)
# ---------------------------------------------------------------------------

DEFAULT_CHIRP_CONFIG: Dict[str, Any] = {
    "start_frequency_Hz": 58000000000,
    "end_frequency_Hz": 63500000000,
    "sample_rate_Hz": 2e6,
    "num_samples": 256,
    "rx_mask": 7,
    "num_rx": 3,
    "tx_mask": 1,
    "tx_power_level": 31,
    "lp_cutoff_Hz": 500000,
    "hp_cutoff_Hz": 80000,
    "if_gain_dB": 33,
}

DEFAULT_SEQ_CONFIG: Dict[str, Any] = {
    "frame_repetition_time_s": 0.1,
    "chirp_repetition_time_s": 0.0002,
    "num_chirps": 128,
    "tdm_mimo": 0,
}


# ---------------------------------------------------------------------------
# DSP (doppler_spectrogram.py)
# ---------------------------------------------------------------------------


def _range_doppler_windows(n_sample: int, n_chirp: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        range_window = signal.blackmanharris(n_sample)
        doppler_window = signal.chebwin(n_chirp, at=100.0)
    except AttributeError:
        range_window = signal.windows.blackmanharris(n_sample)
        doppler_window = signal.windows.chebwin(n_chirp, at=100.0)
    doppler_window = doppler_window / np.sum(doppler_window)
    return range_window, doppler_window


def compute_rdm_db(
    frame: np.ndarray,
    *,
    range_window: np.ndarray,
    doppler_window: np.ndarray,
    range_fft_size: int,
    doppler_fft_size: int,
) -> np.ndarray:
    """One frame (n_chirp, n_sample) → RDM in dB (n_range_bins, doppler_fft_size).

    Vectorized equivalent of the Infineon-style nested-loop RDM used offline.
    """
    n_chirp, n_sample = frame.shape
    n_range_bins = range_fft_size // 2
    clip_db = 20.0 * np.log10(CLIPPING_VALUE)

    # Range FFT across chirps
    x = frame.astype(np.float64, copy=False)
    x = x - np.mean(x, axis=1, keepdims=True)
    x = x * range_window
    range_buf = np.zeros((n_chirp, range_fft_size), dtype=np.complex128)
    range_buf[:, :n_sample] = x
    # (n_range_bins, n_chirp)
    rdm_complex = np.fft.fft(range_buf, axis=1)[:, :n_range_bins].T

    # Doppler FFT across slow-time
    slow = rdm_complex - np.mean(rdm_complex, axis=1, keepdims=True)
    slow = slow * doppler_window
    doppler_buf = np.zeros((n_range_bins, doppler_fft_size), dtype=np.complex128)
    doppler_buf[:, :n_chirp] = slow
    shifted = np.fft.fftshift(np.fft.fft(doppler_buf, axis=1), axes=1)
    power = np.abs(shifted) ** 2

    rdm_db = np.full_like(power, clip_db, dtype=np.float64)
    above = power >= SPECT_THRESHOLD**2
    rdm_db[above] = 10.0 * np.log10(power[above])
    return rdm_db


def select_range_bin_raw(rdm_db: np.ndarray) -> int:
    """Energy-max range bin for one frame (before temporal median smooth)."""
    return int(np.argmax(rdm_db.sum(axis=1)))


def smooth_range_bins(
    raw_bins: np.ndarray, n_range_bins: int, window: int = SMOOTH_WINDOW
) -> np.ndarray:
    w = max(1, window | 1)
    smoothed = median_filter(raw_bins.astype(np.float64), size=w)
    return np.clip(np.round(smoothed), 0, n_range_bins - 1).astype(np.int32)


@dataclass
class LiveDopplerProcessor:
    """Frame-by-frame RDM + rolling spectrogram with median-smoothed range bin."""

    n_sample: int
    n_chirp: int
    history_length: int = 100
    smooth_window: int = SMOOTH_WINDOW

    def __post_init__(self) -> None:
        self.range_fft_size = self.n_sample * 4
        self.doppler_fft_size = self.n_chirp * 4
        self.n_range_bins = self.range_fft_size // 2
        self.range_window, self.doppler_window = _range_doppler_windows(
            self.n_sample, self.n_chirp
        )
        fill = 20.0 * np.log10(CLIPPING_VALUE)
        self.history = np.full(
            (self.history_length, self.doppler_fft_size), fill, dtype=np.float64
        )
        self._raw_bin_buf: Deque[int] = deque(maxlen=max(1, self.smooth_window | 1))
        self.last_range_bin: Optional[int] = None
        self._n_filled = 0

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        if frame.shape != (self.n_chirp, self.n_sample):
            raise ValueError(f"Expected ({self.n_chirp}, {self.n_sample}), got {frame.shape}")

        rdm_db = compute_rdm_db(
            frame,
            range_window=self.range_window,
            doppler_window=self.doppler_window,
            range_fft_size=self.range_fft_size,
            doppler_fft_size=self.doppler_fft_size,
        )
        raw_bin = select_range_bin_raw(rdm_db)
        self._raw_bin_buf.append(raw_bin)

        # Same idea as offline median_filter over the raw per-frame bins.
        buf = np.asarray(self._raw_bin_buf, dtype=np.float64)
        w = max(1, self.smooth_window | 1)
        if len(buf) >= w:
            smoothed = median_filter(buf, size=w)
            range_bin = int(np.clip(np.round(smoothed[-1]), 0, self.n_range_bins - 1))
        else:
            range_bin = int(np.clip(np.round(np.median(buf)), 0, self.n_range_bins - 1))

        self.last_range_bin = range_bin
        # Oldest left → newest right (matches recorded time axis).
        self.history[:-1, :] = self.history[1:, :]
        self.history[-1, :] = rdm_db[range_bin, :]
        self._n_filled = min(self._n_filled + 1, self.history_length)
        return self.history.copy(), range_bin


def compute_recorded(
    data: np.ndarray,
    *,
    antenna: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Offline spectrogram matching doppler_spectrogram.compute (no plot)."""
    if data.ndim != 4:
        raise ValueError(f"Expected (n_frame, n_ant, n_chirp, n_sample), got {data.shape}")

    n_frame, n_ant, n_chirp, n_sample = data.shape
    if antenna >= n_ant:
        raise ValueError(f"antenna {antenna} out of range (n_ant={n_ant})")

    range_fft_size = n_sample * 4
    doppler_fft_size = n_chirp * 4
    n_range_bins = range_fft_size // 2
    range_window, doppler_window = _range_doppler_windows(n_sample, n_chirp)

    rdm_cube = np.zeros((n_frame, n_range_bins, doppler_fft_size), dtype=np.float64)
    for frame_idx in range(n_frame):
        rdm_cube[frame_idx] = compute_rdm_db(
            data[frame_idx, antenna],
            range_window=range_window,
            doppler_window=doppler_window,
            range_fft_size=range_fft_size,
            doppler_fft_size=doppler_fft_size,
        )

    per_frame_energy = np.argmax(rdm_cube.sum(axis=2), axis=1).astype(np.int32)
    range_bin = smooth_range_bins(per_frame_energy, n_range_bins, SMOOTH_WINDOW)

    spectrogram = np.zeros((n_frame, doppler_fft_size), dtype=np.float64)
    for i in range(n_frame):
        spectrogram[i] = rdm_cube[i, range_bin[i], :]
    return spectrogram, range_bin


# ---------------------------------------------------------------------------
# Visualization (doppler_spectrogram.py)
# ---------------------------------------------------------------------------

# "frame_x": frame on x, velocity on y (default, microdoppler)
# "velocity_x": velocity on x, frame on y (rotated)
ORIENT_FRAME_X = "frame_x"
ORIENT_VELOCITY_X = "velocity_x"


def _jet_clim(plot_data: np.ndarray, jet_vmin: float) -> Tuple[float, float]:
    vmax = float(np.nanmax(plot_data))
    vmin = jet_vmin if jet_vmin < vmax else vmax - 40.0
    return vmin, vmax


def _add_velocity_grid(ax, orientation: str = ORIENT_FRAME_X) -> None:
    """Grid lines along the velocity axis, drawn above the spectrogram image."""
    if orientation == ORIENT_VELOCITY_X:
        ax.xaxis.grid(True, linestyle="--", linewidth=0.7, color="white", alpha=0.55)
    else:
        ax.yaxis.grid(True, linestyle="--", linewidth=0.7, color="white", alpha=0.55)
    ax.set_axisbelow(False)


def plot_recorded_spectrogram(
    spectrogram: np.ndarray,
    *,
    max_speed_m_s: float,
    jet_vmin: float,
    title: str = "Doppler Spectrogram",
    orientation: str = ORIENT_FRAME_X,
) -> Figure:
    """Jet spectrogram; default frame×velocity, or rotated velocity×frame."""
    n_frames = spectrogram.shape[0]
    vel_min, vel_max = -max_speed_m_s, max_speed_m_s

    fig = plt.figure(frameon=True)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    fig.add_axes(ax)

    if orientation == ORIENT_VELOCITY_X:
        # rows = frame (oldest bottom → newest top), cols = velocity
        plot_data = np.asarray(spectrogram, dtype=np.float64)
        vmin, vmax = _jet_clim(plot_data, jet_vmin)
        ax.imshow(
            plot_data,
            cmap="jet",
            norm=mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True),
            aspect="auto",
            origin="lower",
            extent=[vel_min, vel_max, 0, n_frames],
        )
        ax.set_xlabel("velocity (m/s)")
        ax.set_ylabel("frame")
    else:
        # rows = velocity, cols = frame
        plot_data = spectrogram.T
        vmin, vmax = _jet_clim(plot_data, jet_vmin)
        ax.imshow(
            plot_data,
            cmap="jet",
            norm=mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True),
            aspect="auto",
            extent=[0, n_frames, vel_min, vel_max],
        )
        ax.set_xlabel("frame")
        ax.set_ylabel("velocity (m/s)")

    _add_velocity_grid(ax, orientation)
    ax.set_title(title)
    return fig


class LiveSpectrogramPlot:
    """Interactive jet spectrogram. Default: frame (x) × velocity (y)."""

    def __init__(
        self,
        *,
        history_length: int,
        max_speed_m_s: float,
        jet_vmin: float = -20.0,
        title: str = "Doppler Spectrogram (live)",
        orientation: str = ORIENT_FRAME_X,
    ):
        self.history_length = history_length
        self.max_speed_m_s = max_speed_m_s
        self.jet_vmin = jet_vmin
        self.orientation = orientation
        self._title = title

        plt.ion()
        self.fig = plt.figure(frameon=True)
        self.ax = plt.Axes(self.fig, [0.08, 0.10, 0.88, 0.82])
        self.fig.add_axes(self.ax)
        self.fig.canvas.manager.set_window_title(title)
        self._image: Optional[AxesImage] = None
        self._is_open = True
        self.fig.canvas.mpl_connect("close_event", self._on_close)

        if orientation == ORIENT_VELOCITY_X:
            self.ax.set_xlabel("velocity (m/s)")
            self.ax.set_ylabel("frame")
            hint = "newest → top"
        else:
            self.ax.set_xlabel("frame")
            self.ax.set_ylabel("velocity (m/s)")
            hint = "newest → right"
        self.ax.set_title(f"{title}  |  {hint}")
        _add_velocity_grid(self.ax, orientation)

    def _on_close(self, _event=None) -> None:
        self._is_open = False

    def is_open(self) -> bool:
        return self._is_open

    def draw(self, spectrogram: np.ndarray, frame_end: int) -> None:
        vel_min, vel_max = -self.max_speed_m_s, self.max_speed_m_s
        frame_start = max(0, frame_end - self.history_length)

        if self.orientation == ORIENT_VELOCITY_X:
            # rows = frame (oldest bottom), cols = velocity; newest at top
            plot_data = np.asarray(spectrogram, dtype=np.float64)
            extent = [vel_min, vel_max, frame_start, frame_end]
            origin = "lower"
            hint = "newest → top"
        else:
            # rows = velocity, cols = frame; newest on the right
            plot_data = np.asarray(spectrogram, dtype=np.float64).T
            extent = [frame_start, frame_end, vel_min, vel_max]
            origin = "upper"
            hint = "newest → right"

        vmin, vmax = _jet_clim(plot_data, self.jet_vmin)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)

        if self._image is None:
            self._image = self.ax.imshow(
                plot_data,
                cmap="jet",
                norm=norm,
                aspect="auto",
                origin=origin,
                extent=extent,
            )
            _add_velocity_grid(self.ax, self.orientation)
        else:
            self._image.set_data(plot_data)
            self._image.set_norm(norm)
            self._image.set_extent(extent)

        self.ax.set_title(f"{self._title}  |  frame {frame_end} ({hint})")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self) -> None:
        if self.is_open():
            self._is_open = False
            plt.close(self.fig)


# ---------------------------------------------------------------------------
# Application API
# ---------------------------------------------------------------------------


@dataclass
class Options:
    antenna: int = 0
    history_length: int = 100
    n_frames: int = 200
    max_speed_m_s: float = 6.19405905
    jet_vmin: float = -20.0
    orientation: str = ORIENT_FRAME_X
    show_plot: bool = True
    save_path: Optional[PathLike] = None
    chirp_config: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_CHIRP_CONFIG))
    seq_config: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_SEQ_CONFIG))

    @property
    def frame_repetition_time_s(self) -> float:
        return float(self.seq_config["frame_repetition_time_s"])


def _build_simple_sequence_config(chirp_config: Dict[str, Any], seq_config: Dict[str, Any]):
    """Build FmcwSimpleSequenceConfig from chirp_config / seq_config dicts."""
    from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig

    config = FmcwSimpleSequenceConfig()
    config.frame_repetition_time_s = float(seq_config["frame_repetition_time_s"])
    config.chirp_repetition_time_s = float(seq_config["chirp_repetition_time_s"])
    config.num_chirps = int(seq_config["num_chirps"])
    config.tdm_mimo = bool(seq_config["tdm_mimo"])

    chirp = config.chirp
    chirp.start_frequency_Hz = float(chirp_config["start_frequency_Hz"])
    chirp.end_frequency_Hz = float(chirp_config["end_frequency_Hz"])
    chirp.sample_rate_Hz = float(chirp_config["sample_rate_Hz"])
    chirp.num_samples = int(chirp_config["num_samples"])
    chirp.rx_mask = int(chirp_config["rx_mask"])
    chirp.tx_mask = int(chirp_config["tx_mask"])
    chirp.tx_power_level = int(chirp_config["tx_power_level"])
    chirp.lp_cutoff_Hz = int(chirp_config["lp_cutoff_Hz"])
    chirp.hp_cutoff_Hz = int(chirp_config["hp_cutoff_Hz"])
    chirp.if_gain_dB = int(chirp_config["if_gain_dB"])
    return config


def run_recorded(input_path: PathLike, options: Optional[Options] = None) -> Figure:
    opts = options or Options()
    data = np.load(input_path)
    spectrogram, range_bins = compute_recorded(data, antenna=opts.antenna)

    n_frame = data.shape[0]
    duration_s = n_frame * opts.frame_repetition_time_s

    fig = plot_recorded_spectrogram(
        spectrogram,
        max_speed_m_s=opts.max_speed_m_s,
        jet_vmin=opts.jet_vmin,
        orientation=opts.orientation,
    )

    print(f"Input: {input_path}")
    print(f"Shape: {data.shape}")
    print(
        f"Frames: {n_frame} (~{duration_s:.3f} s at "
        f"frame_repetition_time_s={opts.frame_repetition_time_s}), "
        f"velocity: [{-opts.max_speed_m_s:.3f}, {opts.max_speed_m_s:.3f}] m/s"
    )
    print(
        f"Selected range bins: min={range_bins.min()}, max={range_bins.max()}, "
        f"mean={range_bins.mean():.1f}"
    )

    if opts.save_path:
        fig.savefig(opts.save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {opts.save_path}")
    if opts.show_plot:
        plt.show(block=True)
    elif not opts.save_path:
        plt.close(fig)
    return fig


def run_live(options: Optional[Options] = None) -> None:
    """Acquire from BGT60TR13C and display a live Doppler spectrogram."""
    opts = options or Options()
    try:
        from ifxradarsdk import get_version_full
        from ifxradarsdk.common.exceptions import ErrorFrameAcquisitionFailed
        from ifxradarsdk.fmcw import DeviceFmcw
    except ImportError as exc:
        raise ImportError("Live mode requires ifxradarsdk. pip install ifxradarsdk") from exc

    chirp_cfg = dict(opts.chirp_config)
    seq_cfg = opts.seq_config
    frame_period_s = float(seq_cfg["frame_repetition_time_s"])
    num_samples = int(chirp_cfg["num_samples"])
    num_chirps = int(seq_cfg["num_chirps"])
    num_rx_cfg = int(chirp_cfg.get("num_rx", bin(int(chirp_cfg["rx_mask"])).count("1")))

    if opts.antenna >= num_rx_cfg:
        raise ValueError(f"antenna {opts.antenna} out of range (n_ant={num_rx_cfg})")

    # Stream only the selected RX antenna to cut USB bandwidth (avoids frame drops).
    chirp_cfg["rx_mask"] = 1 << opts.antenna
    config = _build_simple_sequence_config(chirp_cfg, seq_cfg)

    with DeviceFmcw() as device:
        print(f"Radar SDK Version: {get_version_full()}")
        print(f"Sensor: {device.get_sensor_type()}")

        sequence = device.create_simple_sequence(config)
        device.set_acquisition_sequence(sequence)

        processor = LiveDopplerProcessor(
            n_sample=num_samples,
            n_chirp=num_chirps,
            history_length=opts.history_length,
        )
        plot = LiveSpectrogramPlot(
            history_length=opts.history_length,
            max_speed_m_s=opts.max_speed_m_s,
            jet_vmin=opts.jet_vmin,
            orientation=opts.orientation,
        )

        print(
            f"Live: {num_chirps}x{num_samples}, antenna {opts.antenna}, "
            f"rx_mask={chirp_cfg['rx_mask']}, "
            f"frame_repetition_time_s={frame_period_s} "
            f"({1.0 / frame_period_s:.2f} Hz), history={opts.history_length}"
        )
        print(
            f"Chirp: {chirp_cfg['start_frequency_Hz']/1e9:.2f}–"
            f"{chirp_cfg['end_frequency_Hz']/1e9:.2f} GHz, "
            f"fs={chirp_cfg['sample_rate_Hz']/1e6:.1f} MHz, "
            f"PRT={seq_cfg['chirp_repetition_time_s']*1e3:.3f} ms"
        )
        if opts.orientation == ORIENT_VELOCITY_X:
            print("Orientation velocity_x: newest frames appear at the TOP.")
        else:
            print("Orientation frame_x: newest frames appear on the RIGHT.")

        dropped = 0
        acquired = 0
        while acquired < opts.n_frames:
            if not plot.is_open():
                break
            try:
                # Single-antenna rx_mask → cube axis 0 has length 1.
                frame = device.get_next_frame()[0][0]
            except ErrorFrameAcquisitionFailed:
                dropped += 1
                continue
            history, _ = processor.process_frame(frame)
            acquired += 1
            plot.draw(history, frame_end=acquired)

        if dropped:
            print(f"Skipped {dropped} dropped frame(s) (USB/buffer overrun).")

        if opts.save_path and plot._image is not None:
            plot.fig.savefig(opts.save_path, dpi=150, bbox_inches="tight")
            print(f"Saved plot to {opts.save_path}")
        plot.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Doppler spectrogram (microdoppler viz) — live or recorded"
    )
    parser.add_argument("--mode", choices=("live", "recorded"), default="live")
    parser.add_argument("--input", type=Path, help=".npy path (recorded mode)")
    parser.add_argument("--antenna", type=int, default=0)
    parser.add_argument("--history", type=int, default=100)
    parser.add_argument("--nframes", type=int, default=200)
    parser.add_argument(
        "--frame-repetition-time-s",
        type=float,
        default=None,
        help="Override seq_config frame_repetition_time_s (default: 0.1)",
    )
    parser.add_argument("--max-speed", type=float, default=6.19405905)
    parser.add_argument("--jet-vmin", type=float, default=-20.0)
    parser.add_argument(
        "--orientation",
        choices=(ORIENT_FRAME_X, ORIENT_VELOCITY_X),
        default=ORIENT_FRAME_X,
        help="frame_x: frame on x / velocity on y (default); "
        "velocity_x: velocity on x / frame on y",
    )
    parser.add_argument("--save", type=Path)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    seq_config = dict(DEFAULT_SEQ_CONFIG)
    if args.frame_repetition_time_s is not None:
        seq_config["frame_repetition_time_s"] = args.frame_repetition_time_s

    opts = Options(
        antenna=args.antenna,
        history_length=args.history,
        n_frames=args.nframes,
        max_speed_m_s=args.max_speed,
        jet_vmin=args.jet_vmin,
        orientation=args.orientation,
        show_plot=not args.no_show,
        save_path=args.save,
        chirp_config=dict(DEFAULT_CHIRP_CONFIG),
        seq_config=seq_config,
    )

    if args.mode == "recorded":
        if args.input is None:
            raise SystemExit("--input is required for recorded mode")
        run_recorded(args.input, opts)
    else:
        run_live(opts)


if __name__ == "__main__":
    main()
