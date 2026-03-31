import os
import math
import numpy as np
from collections import deque
from scipy.ndimage import gaussian_filter
from PyQt6 import QtWidgets, QtCore, QtGui
from core.processing import SpectrogramProcessor
from ui.spectrogram_widget import DB_MIN, DB_MAX, FREQ_BINS, MAX_VELOCITY, FRAME_TIME_S


MODELS_ROOT = os.path.join(os.path.expanduser("~"), "SensDSv2_data", "models")

_ROBOT_R = 14
_BALL_R = 8
_TRAIL_MAX = 40
_TICK_MS = 33           # ~30 fps
_INFER_EVERY = 15       # ticks between inferences in RoboSoccer
_BASE_SPEED = 2.0       # px per tick
_PUSH_BURST = 20        # extra px spread across burst ticks
_PUSH_BURST_TICKS = 10
_SINGLE_PUSH_PX = 40
_SINGLE_PUSH_STEPS = 20


TEST_STYLE = """
    QWidget#test_root { background: #f0f2f5; }
    QWidget#test_left {
        background: #ffffff;
        border-right: 1px solid #ddd;
    }
    QLabel#test_heading {
        font-size: 16px;
        font-weight: bold;
        color: #1a3a5c;
    }
    QLabel#test_field_label {
        font-size: 12px;
        font-weight: bold;
        color: #555;
    }
    QLabel#test_model_lbl {
        font-size: 11px;
        color: #666;
        font-family: monospace;
    }
    QRadioButton { font-size: 13px; color: #333; }
    QDoubleSpinBox, QSpinBox {
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 5px 8px;
        font-size: 13px;
        background: white;
        max-height: 30px;
    }
    QPushButton#capture_btn, QPushButton#rs_start_btn {
        background-color: #1a3a5c;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px;
        font-size: 13px;
        font-weight: bold;
    }
    QPushButton#capture_btn:hover, QPushButton#rs_start_btn:hover {
        background-color: #245080;
    }
    QPushButton#capture_btn:disabled, QPushButton#rs_start_btn:disabled {
        background-color: #aaa;
    }
    QPushButton#rs_stop_btn {
        background-color: #c0392b;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px;
        font-size: 13px;
        font-weight: bold;
    }
    QPushButton#rs_stop_btn:hover { background-color: #e74c3c; }
    QLabel#test_status {
        font-size: 12px;
        font-weight: bold;
        color: #27ae60;
    }
"""


# ─── jet colormap (mirrors collect_tab._apply_jet_colormap) ──────────────────

def _apply_jet(normalized):
    r = np.zeros_like(normalized)
    g = np.zeros_like(normalized)
    b = np.zeros_like(normalized)
    mask = normalized < 0.125
    b[mask] = 0.5 + normalized[mask] * 4
    mask = (normalized >= 0.125) & (normalized < 0.375)
    b[mask] = 1.0
    g[mask] = (normalized[mask] - 0.125) * 4
    mask = (normalized >= 0.375) & (normalized < 0.625)
    b[mask] = 1.0 - (normalized[mask] - 0.375) * 4
    g[mask] = 1.0
    r[mask] = (normalized[mask] - 0.375) * 4
    mask = (normalized >= 0.625) & (normalized < 0.875)
    r[mask] = 1.0
    g[mask] = 1.0 - (normalized[mask] - 0.625) * 4
    mask = normalized >= 0.875
    r[mask] = 1.0 - (normalized[mask] - 0.875) * 4 * 0.5
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _frames_to_pil(frames):
    from PIL import Image
    n = len(frames)
    proc = SpectrogramProcessor(buffer_frames=n)
    result = None
    for f in frames:
        result = proc.push_frame(f)
    if result is None:
        return None
    smoothed = gaussian_filter(result.astype(np.float64), sigma=[2.0, 1.5])
    clipped = np.clip(smoothed, DB_MIN, DB_MAX).astype(np.float32)
    normalized = (clipped - DB_MIN) / (DB_MAX - DB_MIN)
    colored = np.ascontiguousarray(_apply_jet(normalized)[::-1])
    return Image.fromarray(colored, "RGB").resize((224, 224), Image.BILINEAR)


# ─── inference worker ────────────────────────────────────────────────────────

class InferenceWorker(QtCore.QObject):
    result = QtCore.pyqtSignal(object)   # dict {label: float}
    error  = QtCore.pyqtSignal(str)
    done   = QtCore.pyqtSignal()

    def __init__(self, frames, model, hf_processor, id2label):
        super().__init__()
        self._frames = frames
        self._model = model
        self._hf_processor = hf_processor
        self._id2label = id2label

    @QtCore.pyqtSlot()
    def run(self):
        try:
            import torch
            img = _frames_to_pil(self._frames)
            if img is None:
                self.error.emit("Not enough frames for inference.")
                return
            inputs = self._hf_processor(images=img, return_tensors="pt")
            with torch.no_grad():
                logits = self._model(**inputs).logits
            probs = torch.softmax(logits[0], dim=0).numpy()
            result = {self._id2label[i]: float(p) for i, p in enumerate(probs)}
            self.result.emit(result)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")
        finally:
            self.done.emit()


# ─── soccer field widget ─────────────────────────────────────────────────────

class SoccerFieldWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._rx = 0.0
        self._ry = 0.0
        self._heading = -90.0   # degrees; -90 = facing up
        self._bx = 0.0
        self._by = 0.0
        self._trail: deque = deque(maxlen=_TRAIL_MAX)
        self._ready = False

    def showEvent(self, event):
        super().showEvent(event)
        if not self._ready and self.width() > 0:
            self._place_at_center()
            self._ready = True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._ready and self.width() > 0:
            self._place_at_center()
            self._ready = True

    def _place_at_center(self):
        cx, cy = self.width() / 2.0, self.height() / 2.0
        self._rx, self._ry = cx, cy
        self._bx, self._by = cx, cy

    def reset(self):
        self._place_at_center()
        self._heading = -90.0
        self._trail.clear()
        self.update()

    # ── read-only accessors used by TestTab ──
    @property
    def robot_pos(self):
        return self._rx, self._ry

    @property
    def heading(self):
        return self._heading

    @property
    def ball_pos(self):
        return self._bx, self._by

    def set_robot(self, x, y, heading):
        self._trail.append((self._rx, self._ry))
        w = max(self.width(), 1)
        h = max(self.height(), 1)
        self._rx = x % w
        self._ry = y % h
        self._heading = heading
        self._push_ball_if_colliding()
        self.update()

    def set_ball(self, x, y):
        self._bx = x
        self._by = y
        self.update()

    def _push_ball_if_colliding(self):
        dx = self._bx - self._rx
        dy = self._by - self._ry
        dist = math.hypot(dx, dy)
        combined = _ROBOT_R + _BALL_R
        if 0 < dist < combined:
            nx, ny = dx / dist, dy / dist
            overlap = combined - dist + 2.0
            w, h = max(self.width(), 1), max(self.height(), 1)
            self._bx = (self._bx + nx * overlap) % w
            self._by = (self._by + ny * overlap) % h

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0

        # Green field
        p.fillRect(0, 0, w, h, QtGui.QColor("#2d8a3e"))

        # White markings
        p.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.white, 2))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)

        # Center line
        p.drawLine(0, int(cy), w, int(cy))

        # Center circle
        cr = int(min(w, h) / 6)
        p.drawEllipse(QtCore.QPointF(cx, cy), cr, cr)

        # Goal areas (top and bottom)
        gw = min(w, h) // 3
        gh = max(int(min(w, h) / 10), 12)
        p.drawRect(int(cx - gw / 2), 0, gw, gh)
        p.drawRect(int(cx - gw / 2), h - gh, gw, gh)

        # Trail
        n_trail = len(self._trail)
        for i, (tx, ty) in enumerate(self._trail):
            alpha = int(20 + 80 * (i + 1) / max(n_trail, 1))
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(QtGui.QColor(180, 180, 180, alpha))
            p.drawEllipse(QtCore.QPointF(tx, ty), 3, 3)

        # Ball
        p.setPen(QtGui.QPen(QtGui.QColor(100, 100, 100), 1))
        p.setBrush(QtGui.QBrush(QtCore.Qt.GlobalColor.white))
        p.drawEllipse(QtCore.QPointF(self._bx, self._by), _BALL_R, _BALL_R)

        # Robot body
        p.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.white, 2))
        p.setBrush(QtGui.QBrush(QtGui.QColor("#1a3a5c")))
        p.drawEllipse(QtCore.QPointF(self._rx, self._ry), _ROBOT_R, _ROBOT_R)

        # Direction indicator
        a = math.radians(self._heading)
        ex = self._rx + math.cos(a) * _ROBOT_R
        ey = self._ry + math.sin(a) * _ROBOT_R
        p.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.white, 2))
        p.drawLine(QtCore.QPointF(self._rx, self._ry), QtCore.QPointF(ex, ey))


# ─── confidence bars widget ──────────────────────────────────────────────────

class ConfidenceBarsWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._probs: dict = {}

    def set_probs(self, probs: dict):
        self._probs = dict(sorted(probs.items()))
        self.update()

    def paintEvent(self, event):
        if not self._probs:
            p = QtGui.QPainter(self)
            p.setPen(QtGui.QColor("#aaa"))
            p.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter,
                       "No prediction yet.")
            return

        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        n = len(self._probs)
        best = max(self._probs, key=self._probs.get)

        margin = 6
        label_w = 110
        pct_w = 46
        total_spacing = margin * (n + 1)
        bar_h = max(8, (h - total_spacing) // n)
        bar_area_w = w - margin * 2 - label_w - pct_w

        font = QtGui.QFont()
        font.setPixelSize(12)
        p.setFont(font)

        for i, (label, prob) in enumerate(self._probs.items()):
            y = margin + i * (bar_h + margin)
            bar_px = int(bar_area_w * prob)
            bar_color = QtGui.QColor("#1a3a5c") if label == best else QtGui.QColor("#aaaaaa")

            # Background track
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(QtGui.QColor("#e0e0e0"))
            p.drawRoundedRect(margin + label_w, y, bar_area_w, bar_h, 3, 3)

            # Filled bar
            if bar_px > 0:
                p.setBrush(bar_color)
                p.drawRoundedRect(margin + label_w, y, bar_px, bar_h, 3, 3)

            # Label
            p.setPen(QtGui.QColor("#333333"))
            p.drawText(
                QtCore.QRectF(margin, y, label_w - 4, bar_h),
                QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter,
                label,
            )

            # Percentage
            p.drawText(
                QtCore.QRectF(margin + label_w + bar_area_w + 4, y, pct_w - 4, bar_h),
                QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
                f"{prob:.0%}",
            )


# ─── main tab ────────────────────────────────────────────────────────────────

class TestTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("test_root")
        self.setStyleSheet(TEST_STYLE)

        self._model = None
        self._hf_processor = None
        self._id2label: dict = {}

        self._frame_buf: deque = deque(maxlen=40)
        self._capturing = False
        self._capture_frames: list = []

        self._inference_running = False
        self._infer_thread = None
        self._infer_worker = None

        self._rs_timer = None
        self._rs_tick_count = 0
        self._rs_speed = _BASE_SPEED
        self._rs_burst_remaining = 0

        self._anim_timer = None
        self._anim_steps = 0
        self._anim_step_px = 0.0

        self._setup_ui()

    # ── public interface ──────────────────────────────────────────────────────

    def on_raw_frame(self, frame: np.ndarray):
        self._frame_buf.append(frame)
        if self._capturing:
            self._capture_frames.append(frame)

    def refresh(self):
        pass

    # ── UI construction ───────────────────────────────────────────────────────

    def _setup_ui(self):
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_left())
        outer.addWidget(self._build_right())

    def _build_left(self):
        panel = QtWidgets.QWidget()
        panel.setObjectName("test_left")
        panel.setFixedWidth(300)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        heading = QtWidgets.QLabel("Test Gestures")
        heading.setObjectName("test_heading")
        layout.addWidget(heading)
        layout.addWidget(self._divider())

        # Model loader
        load_btn = QtWidgets.QPushButton("📂  Load Model")
        load_btn.setStyleSheet("""
            QPushButton {
                background: white; border: 1px solid #ccc; border-radius: 5px;
                padding: 6px 12px; font-size: 13px;
                color: #1a3a5c; font-weight: bold;
            }
            QPushButton:hover { background: #f0f0f0; }
        """)
        load_btn.clicked.connect(self._load_model)
        layout.addWidget(load_btn)

        self._model_lbl = QtWidgets.QLabel("No model loaded.")
        self._model_lbl.setObjectName("test_model_lbl")
        self._model_lbl.setWordWrap(True)
        layout.addWidget(self._model_lbl)

        layout.addWidget(self._divider())

        # Mode selection
        layout.addWidget(self._flbl("Mode"))
        self._radio_single = QtWidgets.QRadioButton("Single Prediction")
        self._radio_single.setChecked(True)
        self._radio_rs = QtWidgets.QRadioButton("RoboSoccer")
        self._radio_single.toggled.connect(self._on_mode_changed)
        layout.addWidget(self._radio_single)
        layout.addWidget(self._radio_rs)

        # ── single prediction controls ──
        self._single_box = QtWidgets.QWidget()
        sb = QtWidgets.QVBoxLayout(self._single_box)
        sb.setContentsMargins(0, 6, 0, 0)
        sb.setSpacing(6)
        sb.addWidget(self._flbl("Capture Duration (s)"))
        self._duration = QtWidgets.QDoubleSpinBox()
        self._duration.setRange(1.0, 10.0)
        self._duration.setValue(3.0)
        self._duration.setSingleStep(0.5)
        sb.addWidget(self._duration)
        self._capture_btn = QtWidgets.QPushButton("⬤  Capture & Predict")
        self._capture_btn.setObjectName("capture_btn")
        self._capture_btn.setEnabled(False)
        self._capture_btn.clicked.connect(self._start_capture)
        sb.addWidget(self._capture_btn)
        layout.addWidget(self._single_box)

        # ── robosoccer controls ──
        self._rs_box = QtWidgets.QWidget()
        self._rs_box.setVisible(False)
        rb = QtWidgets.QVBoxLayout(self._rs_box)
        rb.setContentsMargins(0, 6, 0, 0)
        rb.setSpacing(6)
        rb.addWidget(self._flbl("Confidence Threshold"))
        self._conf_threshold = QtWidgets.QDoubleSpinBox()
        self._conf_threshold.setRange(0.1, 1.0)
        self._conf_threshold.setValue(0.6)
        self._conf_threshold.setSingleStep(0.05)
        rb.addWidget(self._conf_threshold)
        self._rs_start_btn = QtWidgets.QPushButton("▶  Start RoboSoccer")
        self._rs_start_btn.setObjectName("rs_start_btn")
        self._rs_start_btn.setEnabled(False)
        self._rs_start_btn.clicked.connect(self._start_robosoccer)
        rb.addWidget(self._rs_start_btn)
        self._rs_stop_btn = QtWidgets.QPushButton("■  Stop")
        self._rs_stop_btn.setObjectName("rs_stop_btn")
        self._rs_stop_btn.setVisible(False)
        self._rs_stop_btn.clicked.connect(self._stop_robosoccer)
        rb.addWidget(self._rs_stop_btn)
        layout.addWidget(self._rs_box)

        layout.addStretch()

        self._status = QtWidgets.QLabel("")
        self._status.setObjectName("test_status")
        self._status.setWordWrap(True)
        self._status.setMinimumHeight(44)
        layout.addWidget(self._status)

        return panel

    def _build_right(self):
        panel = QtWidgets.QWidget()
        panel.setStyleSheet("background: #f0f2f5;")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._field = SoccerFieldWidget()
        self._field.setMinimumHeight(200)
        layout.addWidget(self._field, 2)

        bars_frame = QtWidgets.QFrame()
        bars_frame.setStyleSheet(
            "QFrame { background: white; border-radius: 8px; border: 1px solid #ddd; }"
        )
        bf = QtWidgets.QVBoxLayout(bars_frame)
        bf.setContentsMargins(10, 8, 10, 8)
        bf.setSpacing(4)

        pred_lbl = QtWidgets.QLabel("Last Prediction")
        pred_lbl.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #1a3a5c; border: none;"
        )
        bf.addWidget(pred_lbl)

        self._bars = ConfidenceBarsWidget()
        bf.addWidget(self._bars, 1)

        layout.addWidget(bars_frame, 1)

        return panel

    def _flbl(self, text):
        l = QtWidgets.QLabel(text)
        l.setObjectName("test_field_label")
        return l

    def _divider(self):
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setStyleSheet("color: #eee; margin: 2px 0;")
        return line

    # ── model loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        os.makedirs(MODELS_ROOT, exist_ok=True)
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Model Folder", MODELS_ROOT
        )
        if not path:
            return
        try:
            from transformers import AutoModelForImageClassification, AutoImageProcessor
            self._status.setText("Loading model…")
            QtWidgets.QApplication.processEvents()
            self._hf_processor = AutoImageProcessor.from_pretrained(path)
            self._model = AutoModelForImageClassification.from_pretrained(
                path, ignore_mismatched_sizes=True
            )
            self._model.eval()
            raw = self._model.config.id2label
            self._id2label = {int(k): v for k, v in raw.items()}
            name = os.path.basename(path.rstrip("/\\"))
            self._model_lbl.setText(f"✓  {name}")
            self._model_lbl.setStyleSheet(
                "font-size: 11px; color: #27ae60; font-family: monospace;"
            )
            self._capture_btn.setEnabled(True)
            self._rs_start_btn.setEnabled(True)
            self._status.setText(
                f"Loaded — {len(self._id2label)} classes"
            )
            self._field.reset()
        except Exception as e:
            self._model = None
            self._hf_processor = None
            self._model_lbl.setText(f"✗  {e}")
            self._model_lbl.setStyleSheet(
                "font-size: 11px; color: #c0392b; font-family: monospace;"
            )
            self._capture_btn.setEnabled(False)
            self._rs_start_btn.setEnabled(False)
            self._status.setText("Failed to load model.")

    # ── mode toggle ───────────────────────────────────────────────────────────

    def _on_mode_changed(self):
        single = self._radio_single.isChecked()
        self._single_box.setVisible(single)
        self._rs_box.setVisible(not single)
        if not single and self._rs_timer and self._rs_timer.isActive():
            self._stop_robosoccer()

    # ── single prediction ─────────────────────────────────────────────────────

    def _start_capture(self):
        self._capture_frames = []
        self._capturing = True
        self._capture_btn.setEnabled(False)
        self._status.setText("⬤  Capturing…")
        self._status.setStyleSheet(
            "color: #c0392b; font-size: 16px; font-weight: bold;"
        )
        QtCore.QTimer.singleShot(
            int(self._duration.value() * 1000), self._capture_done
        )

    def _capture_done(self):
        self._capturing = False
        self._capture_btn.setEnabled(True)
        frames = list(self._capture_frames)
        if len(frames) < 5:
            self._status.setText("Too few frames — connect the radar and try again.")
            self._status.setStyleSheet(
                "color: #c0392b; font-size: 12px; font-weight: bold;"
            )
            return
        self._status.setText("Running inference…")
        self._status.setStyleSheet(
            "color: #e67e22; font-size: 12px; font-weight: bold;"
        )
        self._run_inference(frames, mode="single")

    # ── robosoccer ────────────────────────────────────────────────────────────

    def _start_robosoccer(self):
        self._field.reset()
        self._rs_tick_count = 0
        self._rs_speed = _BASE_SPEED
        self._rs_burst_remaining = 0
        self._inference_running = False
        self._rs_start_btn.setVisible(False)
        self._rs_stop_btn.setVisible(True)
        self._status.setText("RoboSoccer running…")
        self._status.setStyleSheet(
            "color: #27ae60; font-size: 12px; font-weight: bold;"
        )
        self._rs_timer = QtCore.QTimer(self)
        self._rs_timer.timeout.connect(self._on_rs_tick)
        self._rs_timer.start(_TICK_MS)

    def _stop_robosoccer(self):
        if self._rs_timer:
            self._rs_timer.stop()
            self._rs_timer = None
        self._rs_start_btn.setVisible(True)
        self._rs_stop_btn.setVisible(False)
        self._status.setText("RoboSoccer stopped.")
        self._status.setStyleSheet("color: #888; font-size: 12px; font-weight: bold;")

    def _on_rs_tick(self):
        self._rs_tick_count += 1

        rx, ry = self._field.robot_pos
        hdg = self._field.heading
        dx = math.cos(math.radians(hdg)) * self._rs_speed
        dy = math.sin(math.radians(hdg)) * self._rs_speed
        self._field.set_robot(rx + dx, ry + dy, hdg)

        if self._rs_burst_remaining > 0:
            self._rs_burst_remaining -= 1
            if self._rs_burst_remaining == 0:
                self._rs_speed = _BASE_SPEED

        if self._rs_tick_count % _INFER_EVERY == 0 and not self._inference_running:
            frames = list(self._frame_buf)
            if len(frames) >= 5:
                self._run_inference(frames, mode="robosoccer")

    # ── inference ─────────────────────────────────────────────────────────────

    def _run_inference(self, frames, mode):
        if self._inference_running:
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

    def _on_inference_result(self, probs, mode):
        self._inference_running = False
        self._bars.set_probs(probs)
        best = max(probs, key=probs.get)
        conf = probs[best]

        if mode == "single":
            self._status.setText(f"✓  {best}  ({conf:.0%})")
            self._status.setStyleSheet(
                "color: #27ae60; font-size: 16px; font-weight: bold;"
            )
            self._animate_single(best)

        elif mode == "robosoccer":
            if conf >= self._conf_threshold.value():
                self._apply_rs_gesture(best)

    def _on_inference_error(self, msg):
        self._inference_running = False
        first_line = msg.split("\n")[0]
        self._status.setText(f"Inference error: {first_line}")
        self._status.setStyleSheet(
            "color: #c0392b; font-size: 12px; font-weight: bold;"
        )

    # ── robot animation ───────────────────────────────────────────────────────

    def _animate_single(self, gesture):
        rx, ry = self._field.robot_pos
        hdg = self._field.heading
        if gesture == "swipe_left":
            self._field.set_robot(rx, ry, hdg - 45.0)
        elif gesture == "swipe_right":
            self._field.set_robot(rx, ry, hdg + 45.0)
        elif gesture == "push":
            self._anim_steps = _SINGLE_PUSH_STEPS
            self._anim_step_px = _SINGLE_PUSH_PX / _SINGLE_PUSH_STEPS
            if self._anim_timer is None:
                self._anim_timer = QtCore.QTimer(self)
                self._anim_timer.timeout.connect(self._anim_tick)
            self._anim_timer.start(25)
        # idle: no action

    def _anim_tick(self):
        if self._anim_steps <= 0:
            self._anim_timer.stop()
            return
        rx, ry = self._field.robot_pos
        hdg = self._field.heading
        dx = math.cos(math.radians(hdg)) * self._anim_step_px
        dy = math.sin(math.radians(hdg)) * self._anim_step_px
        self._field.set_robot(rx + dx, ry + dy, hdg)
        self._anim_steps -= 1

    def _apply_rs_gesture(self, gesture):
        rx, ry = self._field.robot_pos
        hdg = self._field.heading
        if gesture == "swipe_left":
            self._field.set_robot(rx, ry, hdg - 15.0)
        elif gesture == "swipe_right":
            self._field.set_robot(rx, ry, hdg + 15.0)
        elif gesture == "push":
            self._rs_speed = _BASE_SPEED + _PUSH_BURST / _PUSH_BURST_TICKS
            self._rs_burst_remaining = _PUSH_BURST_TICKS
        # idle: no action
