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
from scipy.ndimage import median_filter
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
        # Rolling peak for the Doppler method's live view.  The script picks
        # vmax from a whole recording; a scrolling view only ever sees a few
        # frames, so a raw per-block max would make the colours flicker as
        # blocks come and go.  An EMA that rises instantly but decays slowly
        # keeps the scale steady while still adapting to the scene.
        self._vmax_ema: float | None = None

    def reset(self):
        """
        Drop all accumulated frames and the colour-scale EMA.

        Called when starting a new streaming session or switching spectrogram
        method, so the next result is built only from fresh frames.
        """
        self._buf.clear()
        self._vmax_ema = None

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
        stack = np.stack(buf_list, axis=0)

        if _METHOD == METHOD_DOPPLER:
            raw   = doppler_spectrogram_from_frames(stack)
            peak  = float(np.nanmax(raw))
            # Rise instantly to a new peak, decay slowly back down.
            if self._vmax_ema is None or peak > self._vmax_ema:
                self._vmax_ema = peak
            else:
                self._vmax_ema = 0.9 * self._vmax_ema + 0.1 * peak
            spect_db = doppler_to_display_db(raw, vmax=self._vmax_ema)
        else:
            spect_db = spectrogram_to_db(spectrogram_from_frames(stack, mti=mti))

        n_emit = min(n_cols, spect_db.shape[1])
        return spect_db[:, -n_emit:]

    # ── batch / inference ─────────────────────────────────────────────────────

    def _emit_batch(self):
        if len(self._buf) < self._buf.maxlen:
            return None
        stack = np.stack(list(self._buf), axis=0)
        return epoch_spectrogram_db(stack)     # honours the selected method


# ══════════════════════════════════════════════════════════════════════════════
# Doppler-RDM spectrogram — exact port of doppler_spectrogram.py (Infineon style)
# ══════════════════════════════════════════════════════════════════════════════
#
# A completely different representation from the STFT method above:
#
#   STFT method    concatenates every chirp of every frame into one long
#                  slow-time signal, sums a FIXED upper-half range window,
#                  then slides a 256-sample Hanning STFT over it.
#                  → 1024 Doppler bins × ~63 time columns for a 3 s epoch
#                  → dB normalised to the image peak, clipped at -20
#
#   Doppler-RDM    builds a full Range-Doppler Map per frame (range FFT over
#                  fast time, then Doppler FFT over slow time), picks the
#                  single MOST ENERGETIC range bin per frame (median-smoothed
#                  across frames so it can't jitter), and emits that one
#                  Doppler slice as the frame's column.
#                  → 512 Doppler bins × 1 column PER FRAME (30 for a 3 s epoch)
#                  → absolute dB, colour-mapped with vmin=-20, vmax=data max
#
# The practical difference: the RDM method tracks the hand's range as it moves
# and reads Doppler only from there, so it rejects clutter at other ranges
# without needing an MTI filter.  The cost is 30× coarser time resolution
# (one column per frame instead of ~2 per frame).
#
# Implementation note — this is a VECTORISED port.  The original script loops
# frame → chirp for the range FFT and frame → range-bin for the Doppler FFT
# (19 200 sequential 1-D FFTs for a 3 s epoch).  Batching those into array FFTs
# measures ~2x faster here (≈320 ms → ≈165 ms per 30-frame epoch on an M-series
# Mac) and, more importantly, lets us bound peak memory via `chunk` instead of
# materialising ~190 MB of complex intermediates at once.
#
# Batched FFTs sum in a different order, so agreement with the original is to
# floating-point rounding (~3e-13 dB on ~120 dB values), not bit-for-bit.
# tests/test_doppler_port.py checks this against a literal transcription of the
# original loops and separately pins the range-bin selection — a different bin
# pick would show up as an O(1) dB error, far above the 1e-9 dB tolerance.

RDM_CLIPPING_VALUE   = 1e-6     # script: CLIPPING_VALUE
RDM_SPECT_THRESHOLD  = 1e-6     # script: SPECT_THRESHOLD
RDM_SMOOTH_WINDOW    = 7        # script: SMOOTH_WINDOW (median filter over frames)
RDM_JET_VMIN         = -20.0    # script: jet_vmin
RDM_RANGE_FFT_MULT   = 4        # script: range_fft_size   = n_sample * 4
RDM_DOPPLER_FFT_MULT = 4        # script: doppler_fft_size = n_chirp  * 4
RDM_MAX_SPEED_M_S    = 6.19405905   # script: max_speed_m_s

# Derived sizes for the app's radar configuration (256 samples, 128 chirps)
RDM_DOPPLER_BINS = N_CHIRPS  * RDM_DOPPLER_FFT_MULT            # 512
RDM_RANGE_BINS   = (N_SAMPLES * RDM_RANGE_FFT_MULT) // 2       # 512
RDM_COLS_PER_FRAME = 1          # the RDM method emits exactly one column/frame

# Cache window functions — chebwin in particular is expensive to build
_RDM_WINDOW_CACHE: dict = {}


def _rdm_windows(n_sample: int, n_chirp: int):
    """Return (range_window, normalised doppler_window), cached by size."""
    key = (n_sample, n_chirp)
    win = _RDM_WINDOW_CACHE.get(key)
    if win is None:
        from scipy import signal as _sig
        try:
            range_window   = _sig.blackmanharris(n_sample)
            doppler_window = _sig.chebwin(n_chirp, at=100.0)
        except AttributeError:
            range_window   = _sig.windows.blackmanharris(n_sample)
            doppler_window = _sig.windows.chebwin(n_chirp, at=100.0)
        doppler_window = doppler_window / np.sum(doppler_window)
        win = (range_window.astype(np.float64), doppler_window.astype(np.float64))
        _RDM_WINDOW_CACHE[key] = win
    return win


def doppler_spectrogram_from_frames(frames: np.ndarray,
                                    antenna: int = 0,
                                    chunk: int = 8,
                                    cube_dtype=np.float32) -> np.ndarray:
    """
    Build the Doppler spectrogram exactly as doppler_spectrogram.py:compute().

    Args:
        frames:     (n_frame, n_ant, n_chirp, n_sample) raw ADC data.
                    A 3-D input (n_ant, n_chirp, n_sample) is treated as one frame.
        antenna:    which antenna to use (script default 0).
        chunk:      frames processed per batch.  Bounds peak memory — the
                    complex intermediates are chunk x ~4 MB, so chunk=8 keeps
                    the transient allocation near 32 MB instead of the ~190 MB
                    a full 30-frame batch would need.
        cube_dtype: storage dtype for the (n_frame, n_range, n_doppler) dB cube.
                    float32 halves the 62 MB float64 cube to 31 MB; the values
                    are dB so the ~1e-5 dB rounding is far below display
                    resolution and does not change the argmax range-bin pick.

    Returns:
        (doppler_fft_size, n_frame) float64 ABSOLUTE dB — the script's
        `plot_data`.  Row 0 is the most-negative Doppler bin (post-fftshift),
        matching the app's existing convention where the display transform
        handles the velocity-axis orientation.
        Call doppler_to_display_db() to map it into the app's [-20, 0] range.
    """
    data = np.asarray(frames)
    if data.ndim == 3:
        data = data[np.newaxis]
    if data.ndim != 4:
        raise ValueError(
            f"Expected (n_frame, n_ant, n_chirp, n_sample), got {data.shape}"
        )

    n_frame, n_ant, n_chirp, n_sample = data.shape
    if antenna >= n_ant:
        raise ValueError(f"antenna {antenna} out of range (n_ant={n_ant})")

    range_fft_size   = n_sample * RDM_RANGE_FFT_MULT
    doppler_fft_size = n_chirp  * RDM_DOPPLER_FFT_MULT
    n_range_bins     = range_fft_size // 2

    range_window, doppler_window = _rdm_windows(n_sample, n_chirp)

    clip_db = 20.0 * np.log10(RDM_CLIPPING_VALUE)
    thr_sq  = RDM_SPECT_THRESHOLD ** 2

    rdm_cube = np.empty((n_frame, n_range_bins, doppler_fft_size), dtype=cube_dtype)

    for start in range(0, n_frame, chunk):
        stop  = min(start + chunk, n_frame)
        block = data[start:stop, antenna].astype(np.float64, copy=False)

        # ── Range FFT ── mean-subtract each chirp, window, zero-pad to 4x, keep
        #    the positive half.  Batched over (frames, chirps).
        x = block - np.mean(block, axis=-1, keepdims=True)
        x = x * range_window
        spec = np.fft.fft(x, n=range_fft_size, axis=-1)
        rdm  = spec[..., :n_range_bins]                 # (b, n_chirp, n_range)
        rdm  = np.swapaxes(rdm, 1, 2)                   # (b, n_range, n_chirp)

        # ── Doppler FFT ── mean-subtract slow time, window, zero-pad, fftshift.
        slow = rdm - np.mean(rdm, axis=-1, keepdims=True)
        slow = slow * doppler_window
        dspec = np.fft.fft(slow, n=doppler_fft_size, axis=-1)
        dspec = np.fft.fftshift(dspec, axes=-1)

        power = np.abs(dspec) ** 2
        db = np.full(power.shape, clip_db, dtype=np.float64)
        above = power >= thr_sq
        db[above] = 10.0 * np.log10(power[above])
        rdm_cube[start:stop] = db

    # ── Range-bin tracking ── strongest total-dB range bin per frame, then a
    #    median filter across frames so the pick cannot jitter frame to frame.
    per_frame_energy = np.argmax(rdm_cube.sum(axis=2), axis=1).astype(np.int32)
    window    = max(1, RDM_SMOOTH_WINDOW | 1)
    smoothed  = median_filter(per_frame_energy.astype(np.float64), size=window)
    range_bin = np.clip(np.round(smoothed), 0, n_range_bins - 1).astype(np.int32)

    spectrogram = rdm_cube[np.arange(n_frame), range_bin, :]   # (n_frame, n_doppler)
    return spectrogram.T.astype(np.float64)                     # script's plot_data


def doppler_to_display_db(spec_db: np.ndarray,
                          jet_vmin: float = RDM_JET_VMIN,
                          vmax: float | None = None) -> np.ndarray:
    """
    Map absolute-dB Doppler output into the app's [DB_MIN, DB_MAX] display range.

    The script colour-maps with
        vmax = np.nanmax(plot_data)
        vmin = jet_vmin if jet_vmin < vmax else vmax - 40.0
    and a jet colormap.  Everything downstream in SensDS (the ImageItem levels,
    the PNG normalisation) is built around a fixed [-20, 0] scale, so we apply
    the script's vmin/vmax and then affinely rescale into [-20, 0].

    Because both are linear maps into the same jet colormap, the rendered
    colours are IDENTICAL to the script's — only the numeric labels differ.

    Args:
        vmax: override the data-derived peak.  Used by the live scrolling view,
              which smooths vmax over time (see SpectrogramProcessor) so the
              colour scale doesn't flicker as blocks come and go.
    """
    if vmax is None:
        vmax = float(np.nanmax(spec_db))
    vmin = jet_vmin if jet_vmin < vmax else vmax - 40.0
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm = np.clip((spec_db - vmin) / (vmax - vmin), 0.0, 1.0)
    return (DB_MIN + norm * (DB_MAX - DB_MIN)).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Method selection — lets the whole app switch representation at runtime
# ══════════════════════════════════════════════════════════════════════════════

METHOD_STFT    = "stft"       # original processing_utils.spectrogram() port
METHOD_DOPPLER = "doppler"    # doppler_spectrogram.py port (Infineon RDM style)

_METHOD = METHOD_STFT


def set_method(method: str):
    """Select the spectrogram representation used app-wide."""
    global _METHOD
    if method not in (METHOD_STFT, METHOD_DOPPLER):
        raise ValueError(f"unknown spectrogram method: {method!r}")
    _METHOD = method


def get_method() -> str:
    return _METHOD


def method_freq_bins(method: str | None = None) -> int:
    """Number of Doppler/frequency bins produced by a method."""
    return RDM_DOPPLER_BINS if (method or _METHOD) == METHOD_DOPPLER else STFT_NFFT


def method_cols_per_frame(method: str | None = None) -> int:
    """New spectrogram columns produced per incoming radar frame."""
    return RDM_COLS_PER_FRAME if (method or _METHOD) == METHOD_DOPPLER else COLS_PER_FRAME


def method_max_velocity(method: str | None = None) -> float:
    """Velocity half-range (m/s) spanned by the frequency axis."""
    return RDM_MAX_SPEED_M_S if (method or _METHOD) == METHOD_DOPPLER else MAX_VELOCITY


def epoch_spectrogram_db(frames: np.ndarray,
                         method: str | None = None) -> np.ndarray:
    """
    Full-epoch dB spectrogram in the app's [DB_MIN, DB_MAX] display range.

    Single entry point used by collection, inference and preview so all three
    always agree on the representation.

    Args:
        frames: (n_frame, n_ant, n_chirp, n_sample)

    Returns:
        (freq_bins, n_cols) float32 dB in [DB_MIN, DB_MAX]
    """
    m = method or _METHOD
    if m == METHOD_DOPPLER:
        return doppler_to_display_db(doppler_spectrogram_from_frames(frames))
    return spectrogram_to_db(spectrogram_from_frames(frames))


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
