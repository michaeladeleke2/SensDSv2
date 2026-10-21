#!/usr/bin/env python3
"""
Live range–time map — Infineon BGT60TR13C.

Same range profile as range_map.py (mti or raw), drawn as a rolling jet image:
time on x, range on y, newest frame on the right. Acquisition matches
microdoppler/doppler_spectrogram_live.py (one RX, same chirp and frame rate).

CLI:
    python range_map_live.py
    python range_map_live.py --until-close
    python range_map_live.py --mode raw --antenna 0 --nframes 200 --history 100

Dependencies: numpy, scipy, matplotlib
Live mode also requires: ifxradarsdk
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

from range_map_v1 import (
    CLIPPING_VALUE,
    END_FREQUENCY_HZ,
    FRAME_REPETITION_TIME_S,
    SAMPLE_RATE_HZ,
    START_FREQUENCY_HZ,
    _jet_clim,
    _windows,
    frame_profile,
    range_axis_m,
)

# Same device setup as microdoppler/doppler_spectrogram_live.py.
DEFAULT_CHIRP_CONFIG: Dict[str, Any] = {
    "start_frequency_Hz": START_FREQUENCY_HZ,
    "end_frequency_Hz": END_FREQUENCY_HZ,
    "sample_rate_Hz": SAMPLE_RATE_HZ,
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
    "frame_repetition_time_s": FRAME_REPETITION_TIME_S,
    "chirp_repetition_time_s": 0.0002,
    "num_chirps": 128,
    "tdm_mimo": 0,
}


@dataclass
class LiveRangeProcessor:
    """Rolling range–time history. Each new frame is appended on the right."""

    n_sample: int
    n_chirp: int
    mode: str = "mti"
    history_length: int = 100
    sample_rate_hz: float = SAMPLE_RATE_HZ
    start_frequency_hz: float = START_FREQUENCY_HZ
    end_frequency_hz: float = END_FREQUENCY_HZ

    def __post_init__(self) -> None:
        if self.mode not in ("mti", "raw"):
            raise ValueError(f"mode must be 'mti' or 'raw', got {self.mode!r}")
        self.range_fft_size = self.n_sample * 4
        self.doppler_fft_size = self.n_chirp * 4
        self.n_range_bins = self.range_fft_size // 2
        self.range_window, self.doppler_window = _windows(self.n_sample, self.n_chirp)
        self.range_m = range_axis_m(
            self.n_range_bins,
            self.range_fft_size,
            sample_rate_hz=self.sample_rate_hz,
            start_frequency_hz=self.start_frequency_hz,
            end_frequency_hz=self.end_frequency_hz,
            n_sample=self.n_sample,
        )
        fill = 20.0 * np.log10(CLIPPING_VALUE)
        self.history = np.full((self.history_length, self.n_range_bins), fill, dtype=np.float64)
        self.n_filled = 0

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        if frame.shape != (self.n_chirp, self.n_sample):
            raise ValueError(
                f"Expected ({self.n_chirp}, {self.n_sample}), got {frame.shape}"
            )
        profile = frame_profile(
            frame,
            mode=self.mode,
            range_window=self.range_window,
            doppler_window=self.doppler_window,
            range_fft_size=self.range_fft_size,
            doppler_fft_size=self.doppler_fft_size,
        )
        self.history[:-1, :] = self.history[1:, :]
        self.history[-1, :] = profile
        self.n_filled = min(self.n_filled + 1, self.history_length)
        return self.history, self.n_filled


class LiveRangePlot:
    """Interactive jet range map. Time on x, range on y, newest frame on the right."""

    def __init__(
        self,
        *,
        history_length: int,
        range_m: np.ndarray,
        frame_repetition_time_s: float,
        jet_vmin: float = -20.0,
        title: str = "Range map (live)",
    ):
        self.history_length = history_length
        self.range_m = np.asarray(range_m, dtype=np.float64)
        self.frame_repetition_time_s = frame_repetition_time_s
        self.jet_vmin = jet_vmin
        self._title = title

        if self.range_m.size > 1:
            dr = float(self.range_m[1] - self.range_m[0])
        else:
            dr = 0.0
        self.range_near = float(self.range_m[0] - 0.5 * dr)
        self.range_far = float(self.range_m[-1] + 0.5 * dr)

        plt.ion()
        self.fig = plt.figure(figsize=(10, 5))
        self.ax = self.fig.add_axes([0.08, 0.12, 0.78, 0.78])
        self.fig.canvas.manager.set_window_title(title)
        self._image: Optional[AxesImage] = None
        self._cbar = None
        self._shown_cols = 0
        self._is_open = True
        self.fig.canvas.mpl_connect("close_event", self._on_close)

        self.ax.set_xlabel("time (s)")
        self.ax.set_ylabel("range (m)")
        self.ax.set_title(f"{title}  |  newest → right")
        self.ax.yaxis.grid(True, linestyle="--", linewidth=0.7, color="white", alpha=0.55)
        self.ax.set_axisbelow(False)

    def _on_close(self, _event=None) -> None:
        self._is_open = False

    def is_open(self) -> bool:
        return self._is_open

    def draw(self, history: np.ndarray, n_filled: int, frame_end: int) -> None:
        visible = np.asarray(history[-n_filled:], dtype=np.float64).T
        t1 = frame_end * self.frame_repetition_time_s
        t0 = (frame_end - n_filled) * self.frame_repetition_time_s
        extent = [t0, t1, self.range_near, self.range_far]
        vmin, vmax = _jet_clim(visible, self.jet_vmin)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)

        if self._image is None or self._shown_cols != n_filled:
            if self._image is not None:
                self._image.remove()
            self._image = self.ax.imshow(
                visible,
                cmap="jet",
                norm=norm,
                aspect="auto",
                origin="lower",
                extent=extent,
            )
            self._shown_cols = n_filled
            if self._cbar is None:
                self._cbar = self.fig.colorbar(self._image, ax=self.ax, label="dB")
            else:
                self._cbar.update_normal(self._image)
        else:
            self._image.set_data(visible)
            self._image.set_norm(norm)
            self._image.set_extent(extent)
            self._cbar.update_normal(self._image)

        self.ax.set_title(f"{self._title}  |  frame {frame_end} (newest → right)")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self) -> None:
        if self.is_open():
            self._is_open = False
            plt.close(self.fig)


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


def run_live(
    *,
    antenna: int = 0,
    mode: str = "mti",
    history_length: int = 100,
    n_frames: Optional[int] = 200,
    jet_vmin: float = -20.0,
    chirp_config: Optional[Dict[str, Any]] = None,
    seq_config: Optional[Dict[str, Any]] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """Acquire from BGT60TR13C and display a live range–time map."""
    try:
        from ifxradarsdk import get_version_full
        from ifxradarsdk.common.exceptions import ErrorFrameAcquisitionFailed
        from ifxradarsdk.fmcw import DeviceFmcw
    except ImportError as exc:
        raise ImportError("Live mode requires ifxradarsdk. pip install ifxradarsdk") from exc

    chirp_cfg = dict(DEFAULT_CHIRP_CONFIG if chirp_config is None else chirp_config)
    seq_cfg = dict(DEFAULT_SEQ_CONFIG if seq_config is None else seq_config)
    frame_period_s = float(seq_cfg["frame_repetition_time_s"])
    num_samples = int(chirp_cfg["num_samples"])
    num_chirps = int(seq_cfg["num_chirps"])
    num_rx_cfg = int(chirp_cfg.get("num_rx", bin(int(chirp_cfg["rx_mask"])).count("1")))

    if antenna < 0 or antenna >= num_rx_cfg:
        raise ValueError(f"antenna {antenna} out of range (n_ant={num_rx_cfg})")

    # Stream only the selected RX antenna to cut USB bandwidth (avoids frame drops).
    chirp_cfg["rx_mask"] = 1 << antenna

    processor = LiveRangeProcessor(
        n_sample=num_samples,
        n_chirp=num_chirps,
        mode=mode,
        history_length=history_length,
        sample_rate_hz=float(chirp_cfg["sample_rate_Hz"]),
        start_frequency_hz=float(chirp_cfg["start_frequency_Hz"]),
        end_frequency_hz=float(chirp_cfg["end_frequency_Hz"]),
    )
    title = "Range map (live, moving)" if mode == "mti" else "Range map (live, raw)"
    plot = LiveRangePlot(
        history_length=history_length,
        range_m=processor.range_m,
        frame_repetition_time_s=frame_period_s,
        jet_vmin=jet_vmin,
        title=title,
    )

    print(f"Opening radar…")
    config = _build_simple_sequence_config(chirp_cfg, seq_cfg)
    with DeviceFmcw() as device:
        print(f"Radar SDK Version: {get_version_full()}")
        print(f"Sensor: {device.get_sensor_type()}")

        sequence = device.create_simple_sequence(config)
        device.set_acquisition_sequence(sequence)

        print(
            f"Live range map: {num_chirps}x{num_samples}, antenna {antenna}, "
            f"rx_mask={chirp_cfg['rx_mask']}, mode={mode}, "
            f"frame_repetition_time_s={frame_period_s} "
            f"({1.0 / frame_period_s:.2f} Hz), history={history_length}"
        )
        print(
            f"Chirp: {chirp_cfg['start_frequency_Hz'] / 1e9:.2f}–"
            f"{chirp_cfg['end_frequency_Hz'] / 1e9:.2f} GHz, "
            f"fs={chirp_cfg['sample_rate_Hz'] / 1e6:.1f} MHz"
        )
        print(
            f"Range axis: {processor.range_m[0]:.3f}–{processor.range_m[-1]:.3f} m "
            f"({processor.range_m.size} bins)"
        )
        if n_frames is None:
            print("Running until the plot window is closed.")
        else:
            print(f"Acquiring {n_frames} frames. Close the plot window to stop early.")

        dropped = 0
        acquired = 0
        while plot.is_open() and (n_frames is None or acquired < n_frames):
            try:
                # Single-antenna rx_mask → cube axis 0 has length 1.
                frame = device.get_next_frame()[0][0]
            except ErrorFrameAcquisitionFailed:
                dropped += 1
                continue
            history, n_filled = processor.process_frame(frame)
            acquired += 1
            plot.draw(history, n_filled, frame_end=acquired)

        if dropped:
            print(f"Skipped {dropped} dropped frame(s) (USB/buffer overrun).")

    if save_path is not None and plot._image is not None:
        plot.fig.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    plot.close()
    return plot.fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live range–time map from a BGT60TR13C"
    )
    parser.add_argument("--antenna", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("mti", "raw"),
        default="mti",
        help="mti: moving targets only (default); raw: include stationary reflections",
    )
    parser.add_argument("--history", type=int, default=100, help="Frames kept on screen")
    parser.add_argument(
        "--nframes",
        type=int,
        default=200,
        help="Frames to acquire (default: 200). Ignored with --until-close",
    )
    parser.add_argument(
        "--until-close",
        action="store_true",
        help="Keep acquiring until the plot window is closed",
    )
    parser.add_argument(
        "--frame-repetition-time-s",
        type=float,
        default=None,
        help="Override frame period in seconds (default: 0.1)",
    )
    parser.add_argument("--jet-vmin", type=float, default=-20.0)
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()

    if args.history < 1:
        raise SystemExit("--history must be at least 1")
    if not args.until_close and args.nframes < 1:
        raise SystemExit("--nframes must be at least 1")

    seq_config = dict(DEFAULT_SEQ_CONFIG)
    if args.frame_repetition_time_s is not None:
        seq_config["frame_repetition_time_s"] = args.frame_repetition_time_s

    run_live(
        antenna=args.antenna,
        mode=args.mode,
        history_length=args.history,
        n_frames=None if args.until_close else args.nframes,
        jet_vmin=args.jet_vmin,
        seq_config=seq_config,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
