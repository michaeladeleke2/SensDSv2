import threading
import numpy as np
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig


def build_config():
    cfg = FmcwSimpleSequenceConfig()
    cfg.frame_repetition_time_s = 0.15
    cfg.chirp_repetition_time_s = 0.0005
    cfg.num_chirps = 64
    cfg.tdm_mimo = False
    cfg.chirp.start_frequency_Hz = 60.5e9
    cfg.chirp.end_frequency_Hz = 61.5e9
    cfg.chirp.sample_rate_Hz = 1e6
    cfg.chirp.num_samples = 64
    cfg.chirp.rx_mask = 7
    cfg.chirp.tx_mask = 1
    cfg.chirp.tx_power_level = 31
    cfg.chirp.lp_cutoff_Hz = 500000
    cfg.chirp.hp_cutoff_Hz = 80000
    cfg.chirp.if_gain_dB = 33
    return cfg


class RadarStream:
    def __init__(self, on_frame, on_error=None):
        self._on_frame = on_frame
        self._on_error = on_error
        self._device = None
        self._thread = None
        self._running = False

    def start(self):
        self._device = DeviceFmcw()
        cfg = build_config()
        sequence = self._device.create_simple_sequence(cfg)
        self._device.set_acquisition_sequence(sequence)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._device:
            self._device.__exit__(None, None, None)
            self._device = None

    def _loop(self):
        try:
            while self._running:
                frame = self._device.get_next_frame()
                self._on_frame(frame[0])
        except Exception as e:
            self._running = False
            if self._on_error:
                self._on_error(str(e))