import numpy as np
from scipy.signal import butter, lfilter
from collections import deque


NFFT = 1024
WINDOW = 256
NOVERLAP = 248
SHIFT = WINDOW - NOVERLAP
FREQ_BINS = NFFT
DB_MIN = -60
DB_MAX = 0
BUFFER_FRAMES = 10


def _stft(signal, window, nfft, shift):
    n = (len(signal) - window - 1) // shift
    out = np.zeros((nfft, n), dtype=complex)
    for i in range(n):
        segment = signal[i * shift: i * shift + window]
        windowed = segment * np.hanning(window)
        out[:, i] = np.fft.fft(windowed, n=nfft)
    return out


def _mti_filter(range_profile):
    b, a = butter(1, 0.01, 'high', output='ba')
    filtered = np.zeros_like(range_profile)
    for r in range(filtered.shape[0]):
        filtered[r, :] = lfilter(b, a, range_profile[r, :])
    return filtered


class SpectrogramProcessor:
    def __init__(self, num_samples=64, num_chirps=64, buffer_frames=BUFFER_FRAMES,
                 streaming=False):
        self._num_samples = num_samples
        self._num_chirps = num_chirps
        self._streaming = streaming
        self._buffer = deque(maxlen=buffer_frames)

    def push_frame(self, frame):
        self._buffer.append(frame[0].copy())
        if len(self._buffer) < self._buffer.maxlen:
            return None
        return self._compute()

    def _compute(self):
        frames = np.array(self._buffer)
        data = np.transpose(frames, (2, 1, 0))
        num_samples = data.shape[0]
        num_chirps = data.shape[1] * data.shape[2]
        data = data.reshape((num_samples, num_chirps), order='F')

        range_fft = np.fft.fft(data, 2 * num_samples, axis=0)[num_samples:] / num_samples
        range_fft -= np.expand_dims(np.mean(range_fft, 1), 1)

        rng = _mti_filter(range_fft)

        rBin = np.arange(num_samples // 2, num_samples - 1)
        vec = np.sum(rng[rBin, :], axis=0)

        spect = _stft(vec, WINDOW, NFFT, SHIFT)
        spect = np.abs(np.fft.fftshift(spect, axes=0))

        maxval = np.max(spect) if np.max(spect) != 0 else 1.0
        spect_db = 20 * np.log10(spect / maxval + 1e-6)

        if self._streaming:
            # Only emit the newly computed columns (one frame's worth) so the
            # rolling display buffer advances correctly without duplicating data.
            cols_per_frame = max(1, self._num_chirps // SHIFT)
            return spect_db[:, -cols_per_frame:]
        return spect_db