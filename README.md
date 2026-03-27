# SensDSv2

A radar-based gesture recognition and ML education tool built on the Infineon BGT60TR13C sensor.

---

## Environment Setup

### Requirements
- Python 3.11
- Miniforge or Anaconda
- Infineon BGT60TR13C radar sensor
- Platform-specific Infineon SDK `.whl` (not included in repo — see below)

### Install
```bash
conda create -n sendsv2 python=3.11 -y
conda activate sendsv2
pip install -r requirements.txt
pip install <platform-specific-ifxradarsdk>.whl
```

**Mac:** `ifxradarsdk-3_6_4_4b4a6245-py3-none-macosx_10_14_universal2.whl`  
**Windows:** `ifxradarsdk-3_6_4_4b4a6245-py3-none-win_amd64.whl`

### Verify install
```bash
python -c "import PyQt6; import pyqtgraph; import ifxradarsdk; print('all good')"
```

---

## Project Structure
```
SensDSv2/
├── main.py                      # Entry point
├── requirements.txt             # Pip dependencies
├── README.md                    # This file
├── core/
│   ├── radar.py                 # Radar connection and streaming
│   └── processing.py            # Signal processing (MTI, STFT, spectrogram)
└── ui/
    ├── main_window.py           # Main application window
    └── spectrogram_widget.py    # Live spectrogram display
```

---

## Build Log

### Step 1 — Project structure and environment
- Created folder layout and `requirements.txt`
- Set up conda environment with Python 3.11
- Key packages: PyQt6, pyqtgraph, numpy, scipy
- Infineon SDK (`ifxradarsdk`) must be installed from a local `.whl` — it is not on PyPI
- `.whl` files are platform-specific and excluded from the repo

### Step 2 — core/radar.py
- Implemented `RadarStream` class with background threading
- Key discovery: in SDK 3.6.4, chirp parameters must be set on `cfg.chirp.*` not directly on the config object
- Valid config requires `rx_mask=7` (all 3 RX antennas enabled) even though we only use antenna 0 for processing
- Each frame returns shape `(3, 64, 64)`: 3 RX antennas × 64 chirps × 64 samples (complex64)
- Verified working on macOS (arm64) and Windows (x86_64)

---

## Concepts Reference

### How the spectrogram pipeline works

Raw radar data flows through these steps before it becomes a visible spectrogram:
```
raw frame (3, 64, 64) complex
        ↓
  take antenna 0 → (64, 64)
        ↓
  Range FFT — find which distances have signal
        ↓
  MTI filter — remove static objects (walls, furniture)
        ↓
  sum across range bins — collapse to 1D signal over time
        ↓
  STFT — sliding window FFT to get frequency vs time
        ↓
  dB scale — compress dynamic range so it's readable
        ↓
  2D spectrogram array ready for display
```

**The key insight:** Doppler shift = velocity. When your hand moves toward 
the radar, the reflected signal shifts to a higher frequency. When it moves 
away, it shifts lower. The spectrogram shows that frequency shift over time, 
which is why a swipe left looks visually different from a push forward — the 
velocity profile over time is different for each gesture.

**Frame shape breakdown:**
- `3` = 3 RX antennas (we use antenna 0 only)
- `64` = chirps per frame (time axis resolution)
- `64` = samples per chirp (range axis resolution)
---