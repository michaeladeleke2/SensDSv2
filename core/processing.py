import numpy as np
from scipy.signal import butter, lfilter
from scipy.signal import stft as scipy_stft


NUM_CHIRPS = 64
NUM_SAMPLES = 64
RANGE_BINS = slice(NUM_SAMPLES // 2, NUM_SAMPLES - 1)
NFFT = 512
WINDOW_SIZE = 32
NOVERLAP = 28


def range_fft(chirp_data):
    n = chirp_data.shape[0]
    fft_out = np.fft.fft(chirp_data, n=2 * n, axis=0)
    fft_out = fft_out[n:] / n
    fft_out -= np.mean(fft_out, axis=1, keepdims=True)
    return fft_out


def mti_filter(range_profile):
    b, a = butter(1, 0.01, btype='high')
    filtered = np.zeros_like(range_profile)
    for i in range(range_profile.shape[0]):
        filtered[i, :] = lfilter(b, a, range_profile[i, :])
    return filtered


def compute_spectrogram(frame):
    data = frame[0]
    data = data.T
    rng = range_fft(data)
    rng = mti_filter(rng)
    signal = np.sum(rng[RANGE_BINS, :], axis=0)
    _, _, Zxx = scipy_stft(signal, nperseg=WINDOW_SIZE, noverlap=NOVERLAP, nfft=NFFT)
    magnitude = np.abs(np.fft.fftshift(Zxx, axes=0))
    max_val = np.max(magnitude)
    if max_val == 0:
        max_val = 1.0
    spectrogram_db = 20 * np.log10(magnitude / max_val + 1e-6)
    return spectrogram_db