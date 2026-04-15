import threading
import numpy as np

# NOTE: ifxradarsdk is imported lazily inside RadarStream.start() so that the
# app can be imported and built (PyInstaller / CI) on machines that don't have
# the Infineon SDK installed.  The Connect Radar button is the only code path
# that ever calls start(), so the ImportError surfaces naturally with a clear
# error message instead of crashing at startup.


def build_config():
    """
    Build radar config matching cfg_simo_chirp.json + cfg_simo_seq.json
    so that the live stream matches the data used for model training.
    """
    from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig
    cfg = FmcwSimpleSequenceConfig()
    # ── Sequence (cfg_simo_seq.json) ─────────────────────────────────────────
    cfg.frame_repetition_time_s = 0.10       # 10 fps
    cfg.chirp_repetition_time_s = 0.0002     # PRF = 5 000 Hz
    cfg.num_chirps              = 128
    cfg.tdm_mimo                = False
    # ── Chirp (cfg_simo_chirp.json) ──────────────────────────────────────────
    cfg.chirp.start_frequency_Hz = 58.0e9
    cfg.chirp.end_frequency_Hz   = 63.5e9   # 5.5 GHz bandwidth
    cfg.chirp.sample_rate_Hz     = 2e6
    cfg.chirp.num_samples        = 256
    cfg.chirp.rx_mask            = 7        # 3 receive antennas
    cfg.chirp.tx_mask            = 1
    cfg.chirp.tx_power_level     = 31
    cfg.chirp.lp_cutoff_Hz       = 500000
    cfg.chirp.hp_cutoff_Hz       = 80000
    cfg.chirp.if_gain_dB         = 33
    return cfg


class RadarStream:
    def __init__(self, on_frame, on_error=None):
        self._on_frame = on_frame
        self._on_error = on_error
        self._device = None
        self._thread = None
        self._running = False

    def start(self):
        # Lazy import — only runs when the user clicks "Connect Radar".
        # This allows the app to start and PyInstaller to build on machines
        # without the Infineon SDK installed.
        try:
            from ifxradarsdk.fmcw import DeviceFmcw
        except ImportError:
            raise RuntimeError(
                "Infineon Radar SDK (ifxradarsdk) is not installed.\n"
                "Install it from the SDK wheel provided with your radar hardware."
            )

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

    # Errors that are transient (dropped/skipped frame) — keep the loop alive.
    _TRANSIENT_ERRORS = ("IFX_ERROR_FRAME_ACQUISITION_FAILED",)

    def _loop(self):
        try:
            while self._running:
                try:
                    frame = self._device.get_next_frame()
                    self._on_frame(frame[0])
                except Exception as e:
                    msg = str(e)
                    if any(tag in msg for tag in self._TRANSIENT_ERRORS):
                        # Dropped frame — skip it and keep streaming.
                        continue
                    raise  # fatal error → break out to outer handler
        except Exception as e:
            self._running = False
            if self._on_error:
                self._on_error(str(e))
