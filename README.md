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
conda create -n sensds2 python=3.11 -y
conda activate sensds2
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

## Running the Application

From the project root:
```bash
python main.py
```

---

## Project Structure
```
SensDSv2/
├── main.py                      # Entry point
├── requirements.txt             # Pip dependencies
├── assets/
│   ├── SensDSLogo.png           # App logo
│   ├── SensDSLogo.ico           # Windows icon
│   └── SensDSLogo.icns          # macOS icon
├── core/
│   ├── radar.py                 # Radar connection and streaming
│   ├── processing.py            # Signal processing pipeline
│   ├── curve_fit.py             # Curve Fit models, Auto Fit tracing and peaks
│   ├── curve_explain.py         # The fit result in words for students
│   ├── range_map.py             # Range map settings; imports the two scripts
│   ├── range_map_v1.py          # range map script (exact copy)
│   └── range_map_live.py        # live range map script (exact copy)
└── ui/
    ├── main_window.py           # Main application window and tab container
    ├── spectrogram_widget.py    # Live spectrogram display (Visualize tab)
    ├── range_map_view.py        # Live range map, drawn by range_map_live.py
    ├── curve_fit_tab.py         # Curve Fit tab
    ├── collect_tab.py           # Gesture data collection
    ├── train_tab.py             # ViT model training
    └── test_tab.py              # Model testing and robot simulation
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
- Velocity axis: ±2.46 m/s derived from PRF=2000 Hz, wavelength=4.92mm (61 GHz)
- RadarBridge uses `pyqtSignal` to safely pass data from radar thread to GUI thread

### Step 5 — ui/main_window.py + ui/collect_tab.py
- Dark top bar with logo, connection status, session timer, connect/disconnect
- Full-width tab bar: Visualize, Collect, Train, Test, Results, RoboSoccer
- Tabs 4-6 are soft-locked placeholders
- RadarBridge emits two signals: `frame_ready` (processed spectrogram) and `raw_frame_ready` (raw frames)
- Collect tab: student name → personal data folder, gesture dropdown with custom label option
- Batch collection: countdown → capture → preview → save, repeats per sample count
- Saves both `.npy` (raw spectrogram) and `.png` (jet colormap, 400×300, for ViT training)
- Open Data Folder button works on Mac and Windows
- Verified on macOS (arm64) and Windows (x86_64)

### Step 6 — ui/train_tab.py
- Scans `~/SensDSv2_data/` for gesture classes and sample counts
- Unlocks when ≥3 gesture classes and ≥20 total PNG samples are found
- Student selection: train on all students or select specific ones
- Training mode: Fast (head only, ~1 min on CPU) or Full fine-tune (~5-10 min)
- Fine-tunes `google/vit-base-patch16-224` via Hugging Face `transformers`
- Runs training on QThread — live loss/accuracy chart + log text update per epoch
- Auto-detects device: CUDA → MPS → CPU
- Saves trained model to `~/SensDSv2_data/models/` as a Hugging Face model directory

### Step 7 — ui/test_tab.py
- Requires a trained model directory to be loaded before use
- Loads models using `AutoModelForImageClassification.from_pretrained` and `AutoImageProcessor.from_pretrained`
- Two modes: Single Prediction (capture one gesture → predict → robot animates) and RoboSoccer (continuous streaming → real-time robot steering)
- Top-down 2D soccer field drawn with QPainter — robot moves, steers, and trails
- Gesture → robot mapping: swipe_left = turn left, swipe_right = turn right, push = speed burst, idle = no change
- Confidence bars show per-class probabilities after each prediction
- All inference runs on QThread

### Range map (Visualize tab)
- **View** switches the Visualize tab between the Doppler spectrogram and the **Range map**: how far away things are (0 to 3.48 m, 512 bins) over time
- Drawn by `range_map_live.py` itself; `core/range_map_v1.py` and `core/range_map_live.py` are exact copies of the two scripts (tests check their hashes) and `core/range_map.py` only imports them
- **MTI filter (show only moving things)** turns MTI on and off; it starts off. `RANGE_MAP_MODE` in `core/range_map.py`, or the `SENSDS_RANGE_MODE` environment variable (`mti` or `raw`), sets the starting value
- **Distance shown (Y)**: From / To in metres. Only the bins inside are drawn and the colours are scaled to them
- MTI and the distance are shared with the Curve Fit tab

### Analysis: Physical Features
- Five measurements per sample: distance from radar, distance traveled, peak and mean speed (toward/away), and **velocity span** — the highest velocity in the gesture's micro-Doppler signature minus the lowest (bins 15 dB above the frame's noise and within 10 dB of its peak). It is not mean speed, and a still hand gives 0
- Clicking a point draws the sample from its raw recording with the Visualize tab's own plots: the micro-Doppler (`LiveSpectrogramPlot`) and the range map (`range_map_v1.py`, with the shared MTI and distance settings); click a picture to see it large (`core/sample_preview.py`)
- Doppler velocities use the chirp spacing recorded in each folder's `capture_info.json` (`chirp_repetition_time_s`, written by the Collect tab), or 0.3 ms when a folder does not say

### Curve Fit
- **Picture**: the Doppler spectrogram (speed) or the range map (distance, with the same MTI and distance controls)
- **Auto Fit** freezes, finds the strongest signal in each moment (on the spectrogram the louder of toward and away, ignoring the still band within 0.1 m/s), marks the high ▲ and low ▼ peaks and fits the chosen model starting from them
- Models: Sinusoid, Damped sinusoid, **Exponential** (y = A·e^(k·t) + c), Polynomial, Linear
- The result panel shows the fit quality (Excellent / Good / Fair / Poor with R²), the general equation and the one with the fitted numbers, a card per letter with its meaning, what the fit predicts (period and pendulum length, or time constant and half-life) and what the peaks show
- The Infineon picture's velocity axis is the reference script's ±4.11 m/s (the 0.3 ms chirp)

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

---

## Notes
- Platform-specific SDK wheel must be installed manually per environment.
- Build executables separately on macOS and Windows using PyInstaller.
- Do not commit build artifacts (`dist/`, `build/`, `.spec`, `.zip`).

