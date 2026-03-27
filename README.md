# SensDSv2

A radar-based gesture recognition and ML education tool built on the Infineon BGT60TR13C sensor.

---

## Environment Setup

### Requirements
- Python 3.11
- Miniforge or Anaconda
- Infineon BGT60TR13C radar sensor
- Platform-specific Infineon SDK `.whl` (not included in repo)

### Install
```bash
conda create -n sendsv2 python=3.11 -y
conda activate sendsv2
pip install -r requirements.txt
pip install <platform-specific-ifxradarsdk>.whl
```

**Mac:** `ifxradarsdk-3_6_4_4b4a6245-py3-none-macosx_10_14_universal2.whl`
**Windows:** `ifxradarsdk-3_6_4_4b4a6245-py3-none-win_amd64.whl`

### Verify
```bash
python -c "import PyQt6; import pyqtgraph; import ifxradarsdk; print('all good')"
```

---

## Project Structure
```
SensDSv2/
├── main.py                      # Entry point
├── requirements.txt             # Pip dependencies
├── core/
│   ├── radar.py                 # Radar connection and streaming
│   └── processing.py            # Signal processing pipeline
└── ui/
    ├── main_window.py           # Main application window
    └── spectrogram_widget.py    # Live spectrogram display
```

---

## Build Log

### Step 1 — Project structure and environment
- `ifxradarsdk` not on PyPI — install from local `.whl`, platform-specific, excluded from repo
- Conda env: Python 3.11, dependencies in `requirements.txt`

### Step 2 — core/radar.py
- Chirp parameters must be set on `cfg.chirp.*`, not on the config object directly (SDK 3.6.4)
- `rx_mask=7` required to enable all 3 RX antennas even though only antenna 0 is used
- Frame shape: `(3, 64, 64)` — antennas × chirps × samples (complex64)
- Verified on macOS (arm64) and Windows (x86_64)

### Step 3 — core/processing.py
- Pipeline: Range FFT → MTI filter → range bin sum → STFT → dB scale
- Accumulates 10 frames in a deque before first output — single frame has too few chirps for a clean STFT
- `WINDOW=256, NOVERLAP=248, NFFT=1024` — matched to v1 parameters
- Signal is complex so STFT must use both positive and negative Doppler bins

### Step 4 — ui/spectrogram_widget.py
- pyqtgraph ImageItem with rolling buffer for live scrolling display
- Jet colormap manually defined to match v1
- `DB_MIN=-20` clips noise floor so gestures stand out
- `gaussian_filter(sigma=[2.0, 1.5])` smooths across frequency and time axes
- RadarBridge uses `pyqtSignal` to safely pass data from radar thread to GUI thread

---

## Concepts Reference

### Doppler shift
Moving targets shift the reflected signal frequency. Toward the radar = higher frequency, away = lower. This is what makes gestures distinguishable.

### Frame structure (3, 64, 64)
3 RX antennas × 64 chirps × 64 samples. Chirps = time resolution, samples = range resolution. We use antenna 0 only.

### Range FFT
Converts each chirp from time domain to distance domain. Each output bin = a specific distance from the sensor.

### MTI filter
High-pass filter that removes static objects (walls, furniture). Only moving targets pass through.

### Range bin summation
Collapses the 2D range-time matrix to a 1D signal by summing across the distance bins where gestures occur.

### STFT
Sliding window FFT across the 1D signal. Output is frequency vs time — the spectrogram. X-axis = time, Y-axis = Doppler velocity.

### dB scale
Compresses signal dynamic range so the display is readable. Max signal = 0 dB, everything else is how far below that ceiling it falls.