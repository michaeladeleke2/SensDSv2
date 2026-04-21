"""
Radar signal processing — exact port of processing_utils.spectrogram()
from the reference Gesture-Controlled-Robo-Soccer codebase.

The spectrogram() function there is what was used to generate ALL training
images.  We replicate it here step-for-step so that inference sees exactly
the same representation the model was trained on.

Reference algorithm (processing_utils.py):
  1. Select antenna 0 only          data[:, 0, :, :]
  2. Transpose + Fortran reshape    (n_sample, n_chirp*n_frame)
  3. Zero-padded range FFT          fft(data, 2*n_sample)[n_sample:] / n_sample
  4. Static clutter removal         subtract mean over slow-time
  5. MTI highpass filter            Butterworth butter(1, 0.01, 'high') + lfilter
  6. Range bin selection            np.arange(n_sample//2, n_sample-1)  [UPPER half]
  7. Range integration              np.sum(rngpro[rBin, :], axis=0)
  8. STFT                           Hanning window, nfft=1024, window=256, noverlap=200
  9. FFT-shift + magnitude          np.abs(np.fft.fftshift(spect, 0))
 10. dB conversion                  20*log10(|spect|/max), display clipped at -20 dB

Advisor-specified epoch/stride
──────────────────────────────
  Epoch  : 3 s  →  30 frames at 10 fps
  Stride : 0.5 s →  5 frames
"""

import numpy as np
from scipy.signal import butter, lfilter
from collections import deque


# ── Radar hardware constants  (cfg_simo_chirp.json + cfg_simo_seq.json) ───────
N_CHIRPS         = 128
N_SAMPLES        = 256
N_ANTENNAS       = 3
FRAME_TIME_S     = 0.10
CHIRP_REP_TIME_S = 0.0002
PRF              = 1.0 / CHIRP_REP_TIME_S          # 5 000 Hz
FC_HZ            = (58.0e9 + 63.5e9) / 2           # ≈ 60.75 GHz
WAVELENGTH       = 3e8 / FC_HZ                     # ≈ 4.94 mm
MAX_VELOCITY     = (PRF * WAVELENGTH) / 4           # ≈ ±6.17 m/s

# ── Epoch / stride (advisor-specified) ───────────────────────────────────────
EPOCH_FRAMES     = 30     # 3 s at 10 fps
STRIDE_FRAMES    = 5      # 0.5 s slide

# ── STFT parameters — EXACT match to processing_utils.py ─────────────────────
STFT_NFFT        = 1024   # nfft = 2**10
STFT_WINDOW      = 256    # window
STFT_NOVERLAP    = 200    # noverlap
STFT_SHIFT       = STFT_WINDOW - STFT_NOVERLAP   # shift = 56

# ── Derived STFT column counts ────────────────────────────────────────────────
# Reference formula: n = (len(data) - window - 1) // shift
EPOCH_CHIRPS     = EPOCH_FRAMES * N_CHIRPS         # 30 × 128 = 3840
EPOCH_COLS       = (EPOCH_CHIRPS - STFT_WINDOW - 1) // STFT_SHIFT   # 63

# New columns added per stride (5 frames × 128 chirps / 56 shift ≈ 11)
STRIDE_CHIRPS    = STRIDE_FRAMES * N_CHIRPS        # 640
STRIDE_COLS      = STRIDE_CHIRPS // STFT_SHIFT     # 11

# New columns added per single frame (for smooth streaming display)
COLS_PER_FRAME   = N_CHIRPS // STFT_SHIFT          # 128 // 56 = 2

# ── dB display range (matches plot_spectrogram: vmin=-20, vmax=None) ──────────
DB_MIN           = -20
DB_MAX           = 0

# ── Legacy aliases ────────────────────────────────────────────────────────────
NFFT             = STFT_NFFT
FREQ_BINS        = STFT_NFFT
BUFFER_FRAMES    = EPOCH_FRAMES

# ── Pre-compute MTI filter coefficients once ──────────────────────────────────
_MTI_B, _MTI_A  = butter(1, 0.01, 'high', output='ba')

# ── Pre-compute Hanning window once ───────────────────────────────────────────
_HANNING_WIN     = np.hanning(STFT_WINDOW).astype(np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# Core spectrogram function — exact port of processing_utils.spectrogram()
# ══════════════════════════════════════════════════════════════════════════════

def spectrogram_from_frames(frames: np.ndarray, mti: bool = True) -> np.ndarray:
    """
    Compute the micro-Doppler spectrogram from a batch of raw radar frames.

    This is a direct port of processing_utils.spectrogram() without matplotlib,
    producing the exact same numerical output used to generate training images.

    Args:
        frames: (n_frame, n_ant, n_chirp, n_sample)  raw IQ data.
                A 3-D input (n_ant, n_chirp, n_sample) is treated as n_frame=1.

    Returns:
        (STFT_NFFT, n_cols) float — FFT-shifted magnitude spectrogram.
        NOT yet converted to dB; call spectrogram_to_db() on the result.
    """
    frames = np.asarray(frames, dtype=complex)
    if frames.ndim == 3:
        frames = frames[np.newaxis]                  # → (1, n_ant, n_chirp, n_sample)

    # ── Step 1: Select antenna 0 only (exact match to reference) ─────────────
    data = frames[:, 0, :, :]                        # (n_frame, n_chirp, n_sample)

    # ── Step 2: Transpose + Fortran-order reshape ─────────────────────────────
    #   (n_frame, n_chirp, n_sample)
    #   → transpose(2,1,0) → (n_sample, n_chirp, n_frame)
    #   → reshape Fortran → (n_sample, n_chirp*n_frame)
    data     = np.transpose(data, (2, 1, 0))         # (n_sample, n_chirp, n_frame)
    n_sample = data.shape[0]
    n_chirps = data.shape[1] * data.shape[2]
    data     = data.reshape((n_sample, n_chirps), order='F')

    # ── Step 3: Zero-padded range FFT (2×N), keep positive half ──────────────
    range_fft = np.fft.fft(data, 2 * n_sample, axis=0)[n_sample:] / n_sample

    # ── Step 4: Static clutter removal (subtract slow-time mean per range bin)─
    range_fft -= np.mean(range_fft, axis=1, keepdims=True)

    # ── Step 5: MTI Butterworth highpass filter along slow-time ──────────────
    if mti:
        rngpro = lfilter(_MTI_B, _MTI_A, range_fft, axis=1)
    else:
        rngpro = range_fft

    # ── Step 6: Range bin selection — UPPER half (matches reference exactly) ──
    #   rBin = np.arange(num_samples // 2, num_samples - 1)
    r_start = n_sample // 2
    r_end   = n_sample - 1
    rBin    = slice(r_start, r_end)

    # ── Step 7: Sum over range bins → 1-D slow-time signal ───────────────────
    vec = np.sum(rngpro[rBin, :], axis=0)            # (n_chirps,) complex

    # ── Step 8: STFT with Hanning window ─────────────────────────────────────
    spect = _stft_hanning(vec)

    # ── Step 9: FFT-shift + magnitude ────────────────────────────────────────
    return np.abs(np.fft.fftshift(spect, axes=0))    # (STFT_NFFT, n_cols) float


def spectrogram_to_db(spect: np.ndarray) -> np.ndarray:
    """
    Convert a magnitude spectrogram to dB, normalised to its own peak.

    Matches plot_spectrogram():
        20 * log10(|spect| / max_val),  clipped at DB_MIN (−20 dB).

    Args:
        spect: (STFT_NFFT, n_cols) float magnitude (output of spectrogram_from_frames)

    Returns:
        (STFT_NFFT, n_cols) float dB, range [DB_MIN, 0]
    """
    max_val = np.max(spect)
    if max_val <= 0:
        max_val = 1.0
    db = 20.0 * np.log10(spect / max_val + 1e-10)
    return np.clip(db, DB_MIN, DB_MAX).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Vectorised STFT — Hanning window, exact column count from reference
# ══════════════════════════════════════════════════════════════════════════════

def _stft_hanning(signal: np.ndarray,
                  window: int = STFT_WINDOW,
                  nfft:   int = STFT_NFFT,
                  shift:  int = STFT_SHIFT) -> np.ndarray:
    """
    STFT using the exact same column-count formula as processing_utils.stft():
        n = (len(data) - window - 1) // shift

    Uses stride tricks + batch FFT (no Python loop) for performance.

    Returns:
        (nfft, n_cols) complex — NOT yet shifted or magnitude-taken.
    """
    n      = len(signal)
    n_cols = (n - window - 1) // shift     # exact reference formula
    if n_cols <= 0:
        return np.zeros((nfft, 1), dtype=complex)

    # Ensure contiguous for stride trick
    sig = np.ascontiguousarray(signal)
    win = _HANNING_WIN[:window]

    # Build (n_cols, window) view without copying
    shape   = (n_cols, window)
    strides = (sig.strides[0] * shift, sig.strides[0])
    frames  = np.lib.stride_tricks.as_strided(sig, shape=shape, strides=strides)

    # Batch FFT
    spectra = np.fft.fft(frames * win, n=nfft, axis=1)   # (n_cols, nfft)
    return spectra.T                                       # (nfft, n_cols)


# ══════════════════════════════════════════════════════════════════════════════
# SpectrogramProcessor — streaming interface used by RadarBridge
# ══════════════════════════════════════════════════════════════════════════════

class SpectrogramProcessor:
    """
    Accumulates raw radar frames and produces a micro-Doppler spectrogram
    using the exact same algorithm as processing_utils.spectrogram().

    streaming=True  (RadarBridge → SpectrogramWidget live display)
        Maintains a rolling frame deque.  The radar thread only accumulates
        frames via push_frame_raw(); the heavy STFT is triggered separately
        via get_streaming_result() from a main-thread QTimer.  This keeps the
        radar collection thread lightweight so it never falls behind.

    streaming=False  (_frames_to_pil → inference)
        Accumulates EPOCH_FRAMES frames then returns the full
        (STFT_NFFT × EPOCH_COLS) dB spectrogram in one shot.
    """

    def __init__(self, num_chirps=N_CHIRPS, num_samples=N_SAMPLES,
                 buffer_frames=EPOCH_FRAMES, streaming=False, **_):
        self._streaming = streaming
        n = max(buffer_frames, EPOCH_FRAMES)
        self._buf: deque = deque(maxlen=n)

    # ── lightweight frame accumulation (radar thread safe) ────────────────────

    def push_frame_raw(self, frame: np.ndarray):
        """
        Accumulate one raw frame WITHOUT running any computation.

        Safe to call from the radar background thread — only touches the
        deque, which is GIL-protected for single append operations.

        Args:
            frame: (n_ant, n_chirp, n_sample) or (n_chirp, n_sample)
        """
        if frame.ndim == 2:
            frame = frame[np.newaxis]
        self._buf.append(frame.copy())

    def push_frame(self, frame: np.ndarray):
        """
        Accumulate a frame and optionally compute the spectrogram.

        For streaming=True, computation is deferred — call get_streaming_result()
        from a timer instead.  Kept for backwards compatibility.

        Returns:
            dB spectrogram slice (streaming mode: never — always returns None)
            or full epoch spectrogram (batch mode) or None if not ready.
        """
        self.push_frame_raw(frame)
        if not self._streaming:
            return self._emit_batch()
        return None   # streaming: caller uses get_streaming_result() via timer

    # ── on-demand display spectrogram (call from main-thread timer) ───────────

    def get_streaming_result(self, n_cols: int = COLS_PER_FRAME,
                              n_frames: int = None,
                              mti: bool = True) -> "np.ndarray | None":
        """
        Compute and return the latest spectrogram columns for display.

        Designed to be called from a BACKGROUND thread — never the radar
        collection thread.  Takes a GIL-safe snapshot of the deque, then
        runs the STFT pipeline.

        Args:
            n_cols:   columns to return; controls scroll speed of the widget.
            n_frames: if given, use only the last n_frames from the buffer.
                      Fewer frames → much faster STFT (no history needed for
                      display).  8 frames gives ~6× speedup vs. 30.
            mti:      apply the Butterworth MTI highpass filter.  Set False
                      for live display — saves 10-50 ms and is imperceptible
                      visually.  Always True for inference.
        Returns:
            (STFT_NFFT, n_cols) float32 dB array, or None if too few frames.
        """
        n = len(self._buf)
        if n < 4:
            return None
        buf_list = list(self._buf)          # GIL-safe snapshot
        if n_frames is not None and n_frames < len(buf_list):
            buf_list = buf_list[-n_frames:]
        if len(buf_list) < 4:
            return None
        stack    = np.stack(buf_list, axis=0)
        spect    = spectrogram_from_frames(stack, mti=mti)
        spect_db = spectrogram_to_db(spect)
        n_emit   = min(n_cols, spect_db.shape[1])
        return spect_db[:, -n_emit:]

    # ── batch / inference ─────────────────────────────────────────────────────

    def _emit_batch(self):
        if len(self._buf) < self._buf.maxlen:
            return None
        stack    = np.stack(list(self._buf), axis=0)
        spect    = spectrogram_from_frames(stack)
        return spectrogram_to_db(spect)                   # (STFT_NFFT, EPOCH_COLS)


# ══════════════════════════════════════════════════════════════════════════════
# Exact reference functions from prediction_utils.py  (kept for completeness)
# ══════════════════════════════════════════════════════════════════════════════

def generate_range_doppler_profiles_per_antenna(data: np.ndarray) -> np.ndarray:
    """Exact copy of generate_range_doppler_profiles_per_antenna."""
    from scipy.signal import windows as sw
    n_frame, n_antenna, n_chirp, n_sample = data.shape
    data_c = data - np.mean(data, axis=-1, keepdims=True)
    rw     = sw.blackmanharris(n_sample).reshape(1, n_sample)
    data_w = data_c * rw
    rfft   = np.fft.fft(data_w, axis=-1) / np.sum(rw)
    rfft  -= np.mean(rfft, axis=2, keepdims=True)
    half   = rfft[..., :n_sample // 2 + 1]
    half[:, :, :, 1::-1] = 2 * half[:, :, :, 1::-1]
    dw     = sw.blackmanharris(n_chirp).reshape(1, 1, n_chirp, 1)
    half_w = half * dw
    return np.fft.fftshift(np.fft.fft(half_w, axis=2), axes=2) / np.sum(dw)


def generate_range_doppler_profiles(data: np.ndarray) -> np.ndarray:
    """Exact copy of generate_range_doppler_profiles."""
    if data.ndim != 4:
        raise ValueError("Input must be 4-D: (n_frame, n_antenna, n_chirp, n_sample)")
    n_frame, n_antenna, n_chirp, n_sample = data.shape
    rd = generate_range_doppler_profiles_per_antenna(data)
    return np.sum(np.abs(rd), axis=1) / n_antenna
