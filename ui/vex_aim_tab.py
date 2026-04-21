"""
ui/vex_aim_tab.py — VEX AIM robot control tab for SensDSv2

Left panel (300 px fixed):
  • Robot connection — IP input, Connect / Disconnect, status dot
  • Model          — loaded model name or warning, Load Model button
  • Mode           — Single Command | RoboSoccer radio buttons
  • Confidence threshold + capture duration spinboxes
  • Start / Stop buttons
  • Status log (dark bg, green mono, read-only QPlainTextEdit)

Right panel:
  • Top half  — live SpectrogramWidget
  • Bottom half — last-prediction card (gesture, confidence, command sent)

Threading notes
  ─────────────
  • Robot() MUST be created on the main thread — vex/aim.py calls signal.signal()
    which only works on the main thread.  We use QTimer.singleShot(100, _do_connect).
  • Inference always runs on a QThread (InferenceWorker).
  • The RoboSoccer drive loop runs on a QThread (DriveWorker).
    DriveWorker counts consecutive send-errors; after _MAX_ERRORS it emits
    robot_lost and exits — the tab resets without crashing the app.
"""

import os
import sys
import time
import datetime
import numpy as np
from collections import deque
from scipy.ndimage import gaussian_filter
from PyQt6 import QtWidgets, QtCore, QtGui
from core.processing import SpectrogramProcessor
from ui.spectrogram_widget import SpectrogramWidget, DB_MIN, DB_MAX
from ui import HintCard, _scrollable_left

# Ensure the project root is importable so "from vex.aim import Robot" resolves.
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

MODELS_ROOT = os.path.join(os.path.expanduser("~"), "SensDSv2_data", "models")

_PRED_CACHE_CONF   = 0.8   # reuse prediction when confidence exceeds this
_PRED_CACHE_FRAMES = 2     # ticks to reuse before re-running inference

# Minimum seconds between consecutive inference calls in RoboSoccer mode.
# Prevents back-to-back inference from saturating the CPU on Surface Pro 9.
from core.platform_utils import min_infer_gap_s as _min_infer_gap_s
_MIN_INFER_GAP_S = _min_infer_gap_s()

# ─── stylesheet ───────────────────────────────────────────────────────────────

VEX_STYLE = """
    QWidget#vex_root  { background: #f0f2f5; }
    QWidget#vex_left  {
        background: #ffffff;
        border-right: 1px solid #dddddd;
    }
    QLabel#vex_heading {
        font-size: 16px;
        font-weight: bold;
        color: #1a3a5c;
    }
    QLabel#vex_field_lbl {
        font-size: 11px;
        font-weight: bold;
        color: #555555;
    }
    QLabel#vex_model_lbl {
        font-size: 11px;
        color: #666666;
        font-family: monospace;
    }
    QRadioButton { font-size: 13px; color: #333333; }
    QDoubleSpinBox, QLineEdit {
        border: 1px solid #cccccc;
        border-radius: 5px;
        padding: 5px 8px;
        font-size: 13px;
        background: white;
        max-height: 30px;
    }
    QPushButton#vex_connect_btn {
        background-color: #27ae60;
        color: white; border: none; border-radius: 6px;
        padding: 8px 10px; font-size: 12px; font-weight: bold;
    }
    QPushButton#vex_connect_btn:hover    { background-color: #2ecc71; }
    QPushButton#vex_connect_btn:disabled { background-color: #aaaaaa; }
    QPushButton#vex_disconnect_btn {
        background-color: #c0392b;
        color: white; border: none; border-radius: 6px;
        padding: 8px 10px; font-size: 12px; font-weight: bold;
    }
    QPushButton#vex_disconnect_btn:hover    { background-color: #e74c3c; }
    QPushButton#vex_disconnect_btn:disabled { background-color: #aaaaaa; }
    QPushButton#vex_load_btn {
        background: white; border: 1px solid #cccccc; border-radius: 5px;
        padding: 6px 12px; font-size: 13px;
        color: #1a3a5c; font-weight: bold;
    }
    QPushButton#vex_load_btn:hover { background: #f0f0f0; }
    QPushButton#vex_start_btn {
        background-color: #1a3a5c;
        color: white; border: none; border-radius: 6px;
        padding: 10px; font-size: 13px; font-weight: bold;
    }
    QPushButton#vex_start_btn:hover    { background-color: #245080; }
    QPushButton#vex_start_btn:disabled { background-color: #aaaaaa; }
    QPushButton#vex_stop_btn {
        background-color: #c0392b;
        color: white; border: none; border-radius: 6px;
        padding: 10px; font-size: 13px; font-weight: bold;
    }
    QPushButton#vex_stop_btn:hover { background-color: #e74c3c; }
    QPlainTextEdit#vex_log {
        background: #0d1117;
        color: #39d353;
        font-family: monospace;
        font-size: 11px;
        border: 1px solid #333333;
        border-radius: 5px;
    }
"""


# ─── spectrogram → PIL helper ─────────────────────────────────────────────────

def _apply_jet(normalized: np.ndarray) -> np.ndarray:
    r = np.zeros_like(normalized)
    g = np.zeros_like(normalized)
    b = np.zeros_like(normalized)
    m = normalized < 0.125
    b[m] = 0.5 + normalized[m] * 4
    m = (normalized >= 0.125) & (normalized < 0.375)
    b[m] = 1.0
    g[m] = (normalized[m] - 0.125) * 4
    m = (normalized >= 0.375) & (normalized < 0.625)
    b[m] = 1.0 - (normalized[m] - 0.375) * 4
    g[m] = 1.0
    r[m] = (normalized[m] - 0.375) * 4
    m = (normalized >= 0.625) & (normalized < 0.875)
    r[m] = 1.0
    g[m] = 1.0 - (normalized[m] - 0.625) * 4
    m = normalized >= 0.875
    r[m] = 1.0 - (normalized[m] - 0.875) * 4 * 0.5
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _frames_to_pil(frames: list):
    """
    Convert raw radar frames → 224×224 PIL spectrogram image.
    Uses last EPOCH_FRAMES (30) frames matching the 3-second training window.
    """
    from PIL import Image
    from core.processing import (
        EPOCH_FRAMES, spectrogram_from_frames, spectrogram_to_db,
    )
    if len(frames) < 10:
        return None
    try:
        epoch  = list(frames)[-EPOCH_FRAMES:]
        stack  = np.stack([np.asarray(f) for f in epoch], axis=0)
        # Ensure (n_frame, n_ant, n_chirp, n_sample)
        if stack.ndim == 3:
            stack = stack[:, np.newaxis]
        spect    = spectrogram_from_frames(stack)   # (STFT_NFFT, n_cols) magnitude
        spect_db = spectrogram_to_db(spect)         # (STFT_NFFT, n_cols) float32 dB
        # Keep float32 — avoids the 2× memory + compute cost of a float64 round-trip.
        smoothed   = gaussian_filter(spect_db, sigma=[1.0, 0.5])
        clipped    = np.clip(smoothed, DB_MIN, DB_MAX)
        normalized = (clipped - DB_MIN) / (DB_MAX - DB_MIN)
        colored    = np.ascontiguousarray(_apply_jet(normalized)[::-1])
        return Image.fromarray(colored, "RGB").resize((224, 224), Image.BILINEAR)
    except Exception:
        return None


# ─── model-load worker ────────────────────────────────────────────────────────

class ModelLoadWorker(QtCore.QObject):
    success = QtCore.pyqtSignal(object, object, dict, str, list)
    error   = QtCore.pyqtSignal(str)
    done    = QtCore.pyqtSignal()

    def __init__(self, path: str):
        super().__init__()
        self._path = path

    @QtCore.pyqtSlot()
    def run(self):
        try:
            from transformers import AutoModelForImageClassification, AutoImageProcessor
            processor = AutoImageProcessor.from_pretrained(self._path)
            model     = AutoModelForImageClassification.from_pretrained(
                self._path, ignore_mismatched_sizes=True
            )
            model.eval()
            from core.platform_utils import get_device
            device = get_device()
            if device is not None:
                model.to(device)
            raw      = model.config.id2label
            id2label = {int(k): v for k, v in raw.items()}
            name     = os.path.basename(self._path.rstrip("/\\"))
            classes  = [id2label[i] for i in sorted(id2label)]
            self.success.emit(model, processor, id2label, name, classes)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.done.emit()


# ─── inference worker ─────────────────────────────────────────────────────────

class InferenceWorker(QtCore.QObject):
    result = QtCore.pyqtSignal(object)
    error  = QtCore.pyqtSignal(str)
    done   = QtCore.pyqtSignal()

    def __init__(self, frames, model, hf_processor, id2label):
        super().__init__()
        self._frames       = frames
        self._model        = model
        self._hf_processor = hf_processor
        self._id2label     = id2label

    @QtCore.pyqtSlot()
    def run(self):
        try:
            try:
                import torch
            except ImportError as ie:
                self.error.emit(
                    "PyTorch could not load — a required DLL is missing.\n\n"
                    "Fix: run  setup_windows.bat  to install the CPU-only build of "
                    "PyTorch, which works on all Windows devices.\n\n"
                    f"(Technical detail: {ie})"
                )
                return

            img = _frames_to_pil(self._frames)
            if img is None:
                self.error.emit("Not enough radar frames — connect the radar first.")
                return
            from core.platform_utils import get_device
            device = get_device()
            inputs = self._hf_processor(images=img, return_tensors="pt")
            if device is not None:
                inputs = {k: v.to(device) for k, v in inputs.items()}
            # inference_mode is faster than no_grad: also disables autograd
            # version tracking — safe for pure inference.
            with torch.inference_mode():
                logits = self._model(**inputs).logits
            probs = torch.softmax(logits[0], dim=0).cpu().numpy()
            self.result.emit(
                {self._id2label[i]: float(p) for i, p in enumerate(probs)}
            )
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")
        finally:
            self.done.emit()


# ─── RoboSoccer drive worker ──────────────────────────────────────────────────

class DriveWorker(QtCore.QObject):
    """
    Keeps the robot rolling forward and triggers inference periodically.

    Crash-safety: consecutive WebSocket send-errors are counted.  After
    _MAX_ERRORS failures in a row the worker assumes the robot has powered off,
    emits robot_lost, and exits its loop — the application keeps running.
    """
    infer_trigger = QtCore.pyqtSignal(list)
    log_msg       = QtCore.pyqtSignal(str)
    robot_lost    = QtCore.pyqtSignal()
    stopped       = QtCore.pyqtSignal()

    _INTERVAL_S  = 0.40
    _INFER_EVERY = 20
    _MAX_ERRORS  = 5

    def __init__(self, robot, frame_buf: deque):
        super().__init__()
        self._robot     = robot
        self._frame_buf = frame_buf
        self._running   = False
        self._tick      = 0

    @QtCore.pyqtSlot()
    def run(self):
        self._running = True
        self._tick    = 0
        err_count     = 0

        while self._running:
            try:
                self._robot.move_for(500, 0, wait=False)
                err_count = 0
            except Exception as e:
                err_count += 1
                self.log_msg.emit(
                    f"⚠ Drive error {err_count}/{self._MAX_ERRORS}: "
                    f"{str(e).split(chr(10))[0]}"
                )
                if err_count >= self._MAX_ERRORS:
                    self.log_msg.emit(
                        "Robot is not responding — it may have been powered off."
                    )
                    self.robot_lost.emit()
                    self._running = False
                    break

            self._tick += 1
            if self._tick % self._INFER_EVERY == 0:
                frames = list(self._frame_buf)
                if len(frames) >= 5:
                    self.infer_trigger.emit(frames)

            time.sleep(self._INTERVAL_S)

        self.stopped.emit()

    def stop(self):
        self._running = False


# ─── gesture → robot command ──────────────────────────────────────────────────

def _apply_gesture(robot, gesture: str) -> str:
    from vex import vex_types as vex
    if gesture == "swipe_left":
        robot.turn_for(vex.TurnType.LEFT, 30, wait=False)
        return "turn left"
    if gesture == "swipe_right":
        robot.turn_for(vex.TurnType.RIGHT, 30, wait=False)
        return "turn right"
    if gesture == "push":
        robot.kicker.kick(vex.KickType.HARD)
        return "kick / surge"
    return "idle (no command)"


# ─── tab widget ───────────────────────────────────────────────────────────────

class VexAimTab(QtWidgets.QWidget):

    # True  = start capturing frames (session beginning)
    # False = stop capturing frames  (session ended)
    stream_needed = QtCore.pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.setObjectName("vex_root")
        self.setStyleSheet(VEX_STYLE)

        self._robot          = None
        self._model          = None
        self._hf_processor   = None
        self._id2label: dict = {}

        self._frame_buf: deque     = deque(maxlen=50)   # 5 s at 10 fps — holds >1 full epoch
        self._capturing            = False
        self._capture_frames: list = []

        self._inference_running = False
        self._infer_thread      = None
        self._infer_worker      = None

        self._drive_thread              = None
        self._drive_worker: DriveWorker = None

        self._gesture_cooldown_until = 0.0  # time.time() threshold for next RS inference
        self._model_load_thread = None
        self._model_load_worker = None

        self._cache_probs: dict    = {}   # last high-confidence prediction probs
        self._cache_remaining: int = 0    # ticks left to reuse the cached result
        self._last_infer_done: float = 0.0  # monotonic time when last real inference completed

        self._setup_ui()

    # ── public interface ──────────────────────────────────────────────────────

    def on_raw_frame(self, frame: np.ndarray):
        self._frame_buf.append(frame)
        if self._capturing:
            self._capture_frames.append(frame)

    def on_spectrogram_frame(self, batch: np.ndarray):
        self._spectrogram.update_frame(batch)

    def stop_if_running(self):
        """
        Stop any active session and signal that the radar is no longer needed.
        Called by MainWindow when the user navigates away from the VEX AIM tab.
        """
        self._on_stop()

    # ── UI construction ───────────────────────────────────────────────────────

    def _setup_ui(self):
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_left())
        outer.addWidget(self._build_right(), 1)

    def _build_left(self):
        panel = QtWidgets.QWidget()
        panel.setObjectName("vex_left")
        lyt = QtWidgets.QVBoxLayout(panel)
        lyt.setContentsMargins(18, 16, 18, 16)
        lyt.setSpacing(7)

        heading = QtWidgets.QLabel("VEX AIM Control")
        heading.setObjectName("vex_heading")
        lyt.addWidget(heading)
        lyt.addWidget(self._divider())

        # ── Robot connection ──────────────────────────────────────────────────
        lyt.addWidget(self._flbl("Robot IP Address"))
        self._ip_input = QtWidgets.QLineEdit("192.168.4.1")
        self._ip_input.setPlaceholderText("192.168.4.1")
        lyt.addWidget(self._ip_input)

        conn_row = QtWidgets.QHBoxLayout()
        conn_row.setSpacing(6)
        self._connect_btn = QtWidgets.QPushButton("Connect")
        self._connect_btn.setObjectName("vex_connect_btn")
        self._connect_btn.clicked.connect(self._on_connect_clicked)
        conn_row.addWidget(self._connect_btn)
        self._disconnect_btn = QtWidgets.QPushButton("Disconnect")
        self._disconnect_btn.setObjectName("vex_disconnect_btn")
        self._disconnect_btn.setEnabled(False)
        self._disconnect_btn.clicked.connect(self._on_disconnect_clicked)
        conn_row.addWidget(self._disconnect_btn)
        lyt.addLayout(conn_row)

        self._robot_status = QtWidgets.QLabel("⬤  Not connected")
        self._robot_status.setStyleSheet("color: #aaaaaa; font-size: 12px;")
        lyt.addWidget(self._robot_status)
        lyt.addWidget(self._divider())

        # ── Model ─────────────────────────────────────────────────────────────
        self._load_btn = QtWidgets.QPushButton("📂  Load Model")
        self._load_btn.setObjectName("vex_load_btn")
        self._load_btn.clicked.connect(self._load_model)
        lyt.addWidget(self._load_btn)

        self._model_lbl = QtWidgets.QLabel("No model loaded.")
        self._model_lbl.setObjectName("vex_model_lbl")
        self._model_lbl.setWordWrap(True)
        lyt.addWidget(self._model_lbl)
        lyt.addWidget(self._divider())

        # ── Mode ──────────────────────────────────────────────────────────────
        lyt.addWidget(self._flbl("Mode"))
        self._radio_single = QtWidgets.QRadioButton("Single Command")
        self._radio_rs     = QtWidgets.QRadioButton("RoboSoccer")
        self._radio_single.setChecked(True)
        self._radio_single.toggled.connect(self._on_mode_changed)
        lyt.addWidget(self._radio_single)
        lyt.addWidget(self._radio_rs)

        # ── Confidence threshold ──────────────────────────────────────────────
        lyt.addWidget(self._flbl("Confidence Threshold"))
        self._conf_threshold = QtWidgets.QDoubleSpinBox()
        self._conf_threshold.setRange(0.1, 1.0)
        self._conf_threshold.setValue(0.6)
        self._conf_threshold.setSingleStep(0.05)
        lyt.addWidget(self._conf_threshold)

        # ── Capture duration (Single Command only) ────────────────────────────
        self._dur_box = QtWidgets.QWidget()
        db = QtWidgets.QVBoxLayout(self._dur_box)
        db.setContentsMargins(0, 0, 0, 0)
        db.setSpacing(4)
        db.addWidget(self._flbl("Capture Duration (s)"))
        self._duration = QtWidgets.QDoubleSpinBox()
        self._duration.setRange(1.0, 10.0)
        self._duration.setValue(3.0)
        self._duration.setSingleStep(0.5)
        db.addWidget(self._duration)
        lyt.addWidget(self._dur_box)

        # ── Start / Stop ──────────────────────────────────────────────────────
        self._start_btn = QtWidgets.QPushButton("▶  Start")
        self._start_btn.setObjectName("vex_start_btn")
        self._start_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._on_start)
        lyt.addWidget(self._start_btn)

        self._stop_btn = QtWidgets.QPushButton("■  Stop")
        self._stop_btn.setObjectName("vex_stop_btn")
        self._stop_btn.setVisible(False)
        self._stop_btn.clicked.connect(self._on_stop)
        lyt.addWidget(self._stop_btn)

        lyt.addWidget(self._divider())

        # ── Hint card ─────────────────────────────────────────────────────────
        lyt.addWidget(HintCard([
            "Connect to the VEX AIM robot's WiFi first (look for a network named "
            "'VEX-AIM-…' in your system WiFi settings), then hit Connect here.",
            "Single Command: perform a gesture during the capture window and "
            "the robot executes that action once.",
            "RoboSoccer: the robot drives forward continuously. "
            "Swipe left/right to steer, push to kick, idle to cruise.",
            "Confidence threshold: the robot only reacts when the model is at least "
            "this sure. Raise it to cut false moves; lower it if the robot ignores you.",
            "If the robot powers off mid-session the app detects it automatically "
            "and stops safely — just reconnect when it's back on.",
            "Getting wrong commands? Go to Collect, add more samples "
            "for that gesture, then retrain your model.",
        ]))

        # ── Status log ────────────────────────────────────────────────────────
        lyt.addWidget(self._flbl("Status Log"))
        self._log_widget = QtWidgets.QPlainTextEdit()
        self._log_widget.setObjectName("vex_log")
        self._log_widget.setReadOnly(True)
        self._log_widget.setMaximumBlockCount(300)
        lyt.addWidget(self._log_widget, 1)

        return _scrollable_left(panel, width=300)

    def _build_right(self):
        panel = QtWidgets.QWidget()
        panel.setStyleSheet("background: #f0f2f5;")
        lyt = QtWidgets.QVBoxLayout(panel)
        lyt.setContentsMargins(12, 12, 12, 12)
        lyt.setSpacing(8)

        self._spectrogram = SpectrogramWidget()
        self._spectrogram.setMinimumHeight(200)
        lyt.addWidget(self._spectrogram, 2)

        pred_frame = QtWidgets.QFrame()
        pred_frame.setStyleSheet(
            "QFrame { background: white; border-radius: 8px; border: 1px solid #ddd; }"
        )
        pf = QtWidgets.QVBoxLayout(pred_frame)
        pf.setContentsMargins(18, 14, 18, 14)
        pf.setSpacing(6)

        lbl = QtWidgets.QLabel("Last Prediction")
        lbl.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #1a3a5c; border: none;"
        )
        pf.addWidget(lbl)

        self._pred_gesture = QtWidgets.QLabel("—")
        self._pred_gesture.setStyleSheet(
            "font-size: 32px; font-weight: bold; color: #1a3a5c; border: none;"
        )
        pf.addWidget(self._pred_gesture)

        details = QtWidgets.QHBoxLayout()
        details.setSpacing(24)
        self._pred_conf = QtWidgets.QLabel("Confidence: —")
        self._pred_conf.setStyleSheet("font-size: 14px; color: #555; border: none;")
        details.addWidget(self._pred_conf)
        self._pred_cmd = QtWidgets.QLabel("Command sent: —")
        self._pred_cmd.setStyleSheet(
            "font-size: 14px; color: #27ae60; font-weight: bold; border: none;"
        )
        details.addWidget(self._pred_cmd)
        details.addStretch()
        pf.addLayout(details)

        lyt.addWidget(pred_frame, 1)
        return panel

    # ── helpers ───────────────────────────────────────────────────────────────

    def _flbl(self, text: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(text)
        lbl.setObjectName("vex_field_lbl")
        return lbl

    def _divider(self) -> QtWidgets.QFrame:
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setStyleSheet("color: #eeeeee; margin: 2px 0;")
        return line

    def _log(self, msg: str):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._log_widget.appendPlainText(f"[{ts}]  {msg}")

    def _refresh_start_btn(self):
        self._start_btn.setEnabled(
            self._robot is not None and self._model is not None
        )

    # ── robot connection ──────────────────────────────────────────────────────

    def _on_connect_clicked(self):
        ip = self._ip_input.text().strip() or "192.168.4.1"
        self._connect_btn.setEnabled(False)
        self._disconnect_btn.setEnabled(False)
        self._ip_input.setEnabled(False)
        self._robot_status.setText("⬤  Connecting…")
        self._robot_status.setStyleSheet("color: #e67e22; font-size: 12px;")
        self._log(f"Connecting to robot at {ip}…")
        # Robot() calls signal.signal() → must stay on the main thread.
        # QTimer.singleShot defers into the Qt event loop without a QThread.
        QtCore.QTimer.singleShot(100, self._do_connect)

    def _do_connect(self):
        ip = self._ip_input.text().strip() or "192.168.4.1"
        try:
            from vex.aim import Robot
            self._robot = Robot(host=ip)
            self._robot_status.setText("⬤  Connected")
            self._robot_status.setStyleSheet("color: #27ae60; font-size: 12px;")
            self._connect_btn.setEnabled(False)
            self._disconnect_btn.setEnabled(True)
            self._ip_input.setEnabled(False)
            self._log(f"✓ Connected to robot at {ip}")
            self._refresh_start_btn()
        except SystemExit:
            # vex/aim.py calls sys.exit(1) on WebSocket timeout.
            # Catching SystemExit here keeps the rest of the app running.
            self._log(
                "❌ Could not reach the robot. "
                "Make sure it is powered on and your device is on its WiFi network."
            )
            self._set_robot_disconnected()
        except Exception as e:
            self._log(f"❌ Connection error: {e}")
            self._set_robot_disconnected()

    def _on_disconnect_clicked(self):
        self._on_stop()
        if self._robot is not None:
            try:
                self._robot.stop_all_movement()
            except Exception:
                pass
        self._robot = None
        self._set_robot_disconnected()
        self._log("Disconnected from robot.")

    def _set_robot_disconnected(self):
        self._robot = None
        self._robot_status.setText("⬤  Not connected")
        self._robot_status.setStyleSheet("color: #aaaaaa; font-size: 12px;")
        self._connect_btn.setEnabled(True)
        self._disconnect_btn.setEnabled(False)
        self._ip_input.setEnabled(True)
        self._start_btn.setEnabled(False)

    # ── robot lost mid-session ────────────────────────────────────────────────

    @QtCore.pyqtSlot()
    def _on_robot_lost(self):
        """Called via signal from DriveWorker when the robot stops responding."""
        self._log("⚠ Robot connection lost — it may have been powered off.")
        self._robot_status.setText("⬤  Connection lost")
        self._robot_status.setStyleSheet("color: #c0392b; font-size: 12px;")
        # Clear drive state without calling stop_all_movement (robot is gone)
        self._drive_worker = None
        self._drive_thread = None
        self._capturing    = False
        self._start_btn.setVisible(True)
        self._stop_btn.setVisible(False)
        self._set_robot_disconnected()
        self.stream_needed.emit(False)  # robot gone — stop radar too

    # ── model loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        os.makedirs(MODELS_ROOT, exist_ok=True)
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Model Folder", MODELS_ROOT
        )
        if not path:
            return

        self._model_lbl.setText("Loading…")
        self._model_lbl.setStyleSheet("font-size: 11px; color: #e67e22; font-family: monospace;")
        self._load_btn.setEnabled(False)
        self._start_btn.setEnabled(False)

        worker = ModelLoadWorker(path)
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.success.connect(self._on_model_loaded)
        worker.error.connect(self._on_model_load_error)
        worker.done.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(lambda: self._load_btn.setEnabled(True))
        thread.start()
        self._model_load_thread = thread
        self._model_load_worker = worker

    def _on_model_loaded(self, model, processor, id2label, name, classes):
        self._model        = model
        self._hf_processor = processor
        self._id2label     = id2label
        self._model_lbl.setText(f"✓  {name}")
        self._model_lbl.setStyleSheet(
            "font-size: 11px; color: #27ae60; font-family: monospace;"
        )
        self._log(
            f"✓ Model loaded: {name}  "
            f"({len(id2label)} classes: {', '.join(classes)})"
        )
        self._refresh_start_btn()

    def _on_model_load_error(self, msg):
        self._model        = None
        self._hf_processor = None
        self._id2label     = {}
        self._model_lbl.setText(f"✗  {msg}")
        self._model_lbl.setStyleSheet(
            "font-size: 11px; color: #c0392b; font-family: monospace;"
        )
        self._log(f"❌ Model load failed: {msg}")
        self._start_btn.setEnabled(False)

    # ── mode toggle ───────────────────────────────────────────────────────────

    def _on_mode_changed(self):
        self._dur_box.setVisible(self._radio_single.isChecked())

    # ── start / stop ──────────────────────────────────────────────────────────

    def _on_start(self):
        if self._radio_single.isChecked():
            self._start_single()
        else:
            self._start_robosoccer()

    def _on_stop(self):
        if self._drive_worker is not None:
            self._drive_worker.stop()
        if self._drive_thread is not None:
            self._drive_thread.quit()
            self._drive_thread.wait(1500)
        self._drive_thread = None
        self._drive_worker = None
        self._capturing    = False

        if self._robot is not None:
            try:
                self._robot.stop_all_movement()
            except Exception:
                pass

        self._start_btn.setVisible(True)
        self._stop_btn.setVisible(False)
        self._refresh_start_btn()
        self._log("Stopped.")
        self.stream_needed.emit(False)  # radar no longer needed

    # ── single command ────────────────────────────────────────────────────────

    def _start_single(self):
        self._capture_frames = []
        self._capturing      = True
        self._start_btn.setEnabled(False)
        dur_ms = int(self._duration.value() * 1000)
        self._log(f"Capturing for {self._duration.value():.1f} s…")
        self.stream_needed.emit(True)   # start radar for this capture window
        QtCore.QTimer.singleShot(dur_ms, self._single_capture_done)

    def _single_capture_done(self):
        self._capturing = False
        frames = list(self._capture_frames)
        if len(frames) < 5:
            self._log("Too few frames — connect the radar and try again.")
            self._refresh_start_btn()
            self.stream_needed.emit(False)   # abort — stop radar
            return
        self._log(f"Running inference on {len(frames)} frames…")
        self._run_inference(frames, mode="single")

    # ── robosoccer ────────────────────────────────────────────────────────────

    def _start_robosoccer(self):
        self._log("Starting RoboSoccer — robot moving forward…")
        self._start_btn.setVisible(False)
        self._stop_btn.setVisible(True)
        self.stream_needed.emit(True)   # start radar streaming for this session

        worker = DriveWorker(self._robot, self._frame_buf)
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.log_msg.connect(self._log)
        worker.infer_trigger.connect(self._on_rs_infer_trigger)
        worker.robot_lost.connect(self._on_robot_lost)
        worker.stopped.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(worker.deleteLater)
        thread.start()

        self._drive_thread = thread
        self._drive_worker = worker

    @QtCore.pyqtSlot(list)
    def _on_rs_infer_trigger(self, frames: list):
        if (not self._inference_running
                and time.time() >= self._gesture_cooldown_until
                and (time.monotonic() - self._last_infer_done) >= _MIN_INFER_GAP_S):
            self._run_inference(frames, mode="robosoccer")

    # ── inference ─────────────────────────────────────────────────────────────

    def _run_inference(self, frames: list, mode: str):
        if self._inference_running or self._model is None:
            return

        # ── prediction cache ──────────────────────────────────────────────────
        # For RoboSoccer mode, reuse a recent high-confidence result for up to
        # _PRED_CACHE_FRAMES ticks to reduce CPU load on low-power devices.
        if mode == "robosoccer" and self._cache_remaining > 0 and self._cache_probs:
            self._cache_remaining -= 1
            self._on_inference_result(self._cache_probs, mode, _from_cache=True)
            return

        self._inference_running = True

        worker = InferenceWorker(
            frames, self._model, self._hf_processor, self._id2label
        )
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.result.connect(lambda p, m=mode: self._on_inference_result(p, m))
        worker.error.connect(self._on_inference_error)
        worker.done.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(worker.deleteLater)
        thread.start()

        self._infer_thread = thread
        self._infer_worker = worker

    def _on_inference_result(self, probs: dict, mode: str, _from_cache: bool = False):
        self._inference_running = False
        best      = max(probs, key=probs.get)
        conf      = probs[best]
        threshold = self._conf_threshold.value()

        # ── update prediction cache + inference gap timer ─────────────────────
        # Only update from real inference results, NOT from cached re-plays.
        if not _from_cache:
            self._last_infer_done = time.monotonic()   # start the gap timer
            if mode == "robosoccer" and conf >= _PRED_CACHE_CONF:
                self._cache_probs     = dict(probs)
                self._cache_remaining = _PRED_CACHE_FRAMES
            elif mode == "robosoccer":
                self._cache_remaining = 0

        self._pred_gesture.setText(best.replace("_", " "))
        self._pred_conf.setText(f"Confidence: {conf:.0%}")

        cmd_text = "—"
        if self._robot is not None and conf >= threshold:
            try:
                cmd_text = _apply_gesture(self._robot, best)
                # Hold off the next RoboSoccer inference for 3 s so the robot
                # has time to execute the command before we re-classify.
                if best != "idle":
                    self._gesture_cooldown_until = time.time() + 3.0
                    # Clear cache so fresh inference runs after cooldown instead
                    # of the same gesture re-firing indefinitely.
                    self._cache_probs     = {}
                    self._cache_remaining = 0
            except Exception as e:
                cmd_text = "error"
                self._log(f"Command error: {e}")
        elif conf < threshold:
            cmd_text = f"skipped ({conf:.0%} < {threshold:.0%} threshold)"

        self._pred_cmd.setText(f"Command: {cmd_text}")
        self._log(f"{best}  {conf:.0%}  →  {cmd_text}")

        if mode == "single":
            self._refresh_start_btn()
            self.stream_needed.emit(False)  # single capture done — stop radar

    def _on_inference_error(self, msg: str):
        self._inference_running = False
        self._log(f"Inference error: {msg.split(chr(10))[0]}")
        self._refresh_start_btn()
        self.stream_needed.emit(False)  # stop radar on error
