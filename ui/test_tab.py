import os
import math
import random
import numpy as np
from collections import deque
from scipy.ndimage import gaussian_filter
from PyQt6 import QtWidgets, QtCore, QtGui
from core.processing import SpectrogramProcessor
from ui import HintCard, _scrollable_left
from ui.spectrogram_widget import DB_MIN, DB_MAX, FREQ_BINS, MAX_VELOCITY, FRAME_TIME_S


MODELS_ROOT = os.path.join(os.path.expanduser("~"), "SensDSv2_data", "models")

# ── soccer/animation constants ────────────────────────────────────────────────
_ROBOT_R = 14
_BALL_R = 8
_TRAIL_MAX = 40
_TICK_MS = 33           # ~30 fps
_INFER_EVERY = 20       # ticks between inferences in RoboSoccer (~667 ms at 30 fps)
_PRED_CACHE_CONF   = 0.8   # reuse prediction when confidence exceeds this
_PRED_CACHE_FRAMES = 2     # frames to reuse before re-running inference

# Maze has its own (slower) inference rhythm so the student has time to
# complete a full gesture before the model looks at the frames.
_MAZE_INFER_EVERY    = 45   # ticks between maze inferences (~1.5 s at 30 fps)
_MAZE_MIN_FRAMES     = 30   # need a full 3-second epoch (30 frames at 10 fps)
_MAZE_CAPTURE_FRAMES = 50   # send last 50 frames; _frames_to_pil picks the last 30
_BASE_SPEED = 2.0       # px per tick
_PUSH_BURST = 20        # extra px spread across burst ticks
_PUSH_BURST_TICKS = 10
_SINGLE_PUSH_PX = 40
_SINGLE_PUSH_STEPS = 20

# ── maze constants ────────────────────────────────────────────────────────────
_MAZE_ROWS = 5
_MAZE_COLS = 7
_MAZE_SEED = 42

# Direction bits
_N = 1
_E = 2
_S = 4
_W = 8

_OPP   = {_N: _S, _S: _N, _E: _W, _W: _E}
_DR    = {_N: -1, _S: 1,  _E: 0,  _W: 0}
_DC    = {_N: 0,  _S: 0,  _E: 1,  _W: -1}
_LABEL = {_N: "North ↑", _E: "East →", _S: "South ↓", _W: "West ←"}
_TURN_L = {_N: _W, _W: _S, _S: _E, _E: _N}
_TURN_R = {_N: _E, _E: _S, _S: _W, _W: _N}
_ARROW  = {_N: "↑", _E: "→", _S: "↓", _W: "←"}


TEST_STYLE = """
    QWidget#test_root { background: #f0f2f5; }
    QWidget#test_left {
        background: #ffffff;
        border-right: 1px solid #ddd;
    }
    QLabel#test_heading {
        font-size: 18px;
        font-weight: bold;
        color: #1a3a5c;
    }
    QLabel#test_subtitle {
        font-size: 13px;
        color: #666;
    }
    QLabel#test_field_label {
        font-size: 13px;
        font-weight: bold;
        color: #555;
    }
    QLabel#test_model_lbl {
        font-size: 12px;
        color: #666;
        font-family: monospace;
    }
    QDoubleSpinBox, QSpinBox {
        border: 1px solid #ccc;
        border-radius: 6px;
        padding: 5px 8px;
        font-size: 13px;
        background: white;
        max-height: 32px;
    }
    QPushButton#capture_btn {
        background-color: #c0392b;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 11px;
        font-size: 14px;
        font-weight: bold;
    }
    QPushButton#capture_btn:hover { background-color: #e74c3c; }
    QPushButton#capture_btn:disabled { background-color: #aaa; }
    QPushButton#rs_start_btn {
        background-color: #27ae60;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 11px;
        font-size: 14px;
        font-weight: bold;
    }
    QPushButton#rs_start_btn:hover { background-color: #2ecc71; }
    QPushButton#rs_start_btn:disabled { background-color: #aaa; }
    QPushButton#rs_stop_btn {
        background-color: #c0392b;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 11px;
        font-size: 14px;
        font-weight: bold;
    }
    QPushButton#rs_stop_btn:hover { background-color: #e74c3c; }
    QPushButton#maze_capture_btn {
        background-color: #8e44ad;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 11px;
        font-size: 14px;
        font-weight: bold;
    }
    QPushButton#maze_capture_btn:hover { background-color: #9b59b6; }
    QPushButton#maze_capture_btn:disabled { background-color: #aaa; }
    QPushButton#maze_reset_btn {
        background-color: #f39c12;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 11px;
        font-size: 14px;
        font-weight: bold;
    }
    QPushButton#maze_reset_btn:hover { background-color: #f1c40f; }
    QLabel#test_status {
        font-size: 13px;
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
    """
    Convert raw radar frames → 224×224 PIL spectrogram image.

    Directly uses the vectorised pipeline (no intermediate SpectrogramProcessor
    state) so inference always gets a clean 3-second epoch regardless of how
    many frames are in the live buffer.

    Each frame must be shape (n_antenna, n_chirp, n_sample).
    Requires at least 10 frames; uses the last EPOCH_FRAMES (30) if more given.
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
        spect_db = spectrogram_to_db(spect)         # (STFT_NFFT, n_cols) dB, clipped at -20 dB

        smoothed   = gaussian_filter(spect_db.astype(np.float64), sigma=[1.0, 0.5])
        clipped    = np.clip(smoothed, DB_MIN, DB_MAX).astype(np.float32)
        normalized = (clipped - DB_MIN) / (DB_MAX - DB_MIN)
        colored    = np.ascontiguousarray(_apply_jet(normalized)[::-1])
        return Image.fromarray(colored, "RGB").resize((224, 224), Image.BILINEAR)
    except Exception:
        return None


def _generate_maze(rows, cols, seed=42):
    """Recursive-backtracker DFS maze generation."""
    walls = [[_N | _E | _S | _W for _ in range(cols)] for _ in range(rows)]
    visited = [[False] * cols for _ in range(rows)]
    rng = random.Random(seed)

    def carve(r, c):
        visited[r][c] = True
        dirs = [_N, _E, _S, _W]
        rng.shuffle(dirs)
        for d in dirs:
            nr, nc = r + _DR[d], c + _DC[d]
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                walls[r][c] &= ~d
                walls[nr][nc] &= ~_OPP[d]
                carve(nr, nc)

    carve(0, 0)
    return walls


# ─── model-load worker ───────────────────────────────────────────────────────

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
            try:
                import torch
            except ImportError as ie:
                # Common on Windows when the CUDA PyTorch build is installed on a
                # device without an NVIDIA GPU (c10.dll / torch_cuda.dll missing).
                self.error.emit(
                    "PyTorch could not load — a required DLL is missing.\n\n"
                    "Fix: run  setup_windows.bat  to install the CPU-only build of "
                    "PyTorch, which works on all Windows devices.\n\n"
                    f"(Technical detail: {ie})"
                )
                return

            img = _frames_to_pil(self._frames)
            if img is None:
                self.error.emit("Not enough frames for inference.")
                return
            from core.platform_utils import get_device
            device = get_device()
            inputs = self._hf_processor(images=img, return_tensors="pt")
            if device is not None:
                inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self._model(**inputs).logits
            probs = torch.softmax(logits[0], dim=0).cpu().numpy()
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
        try:
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

            # Ball — ⚽ emoji
            ball_font = QtGui.QFont()
            ball_font.setPixelSize(_BALL_R * 2)
            p.setFont(ball_font)
            p.setPen(QtCore.Qt.GlobalColor.white)
            p.drawText(
                QtCore.QRectF(self._bx - _BALL_R, self._by - _BALL_R,
                              _BALL_R * 2, _BALL_R * 2),
                QtCore.Qt.AlignmentFlag.AlignCenter, "⚽",
            )

            # Robot — 🤖 emoji
            robot_font = QtGui.QFont()
            robot_font.setPixelSize(_ROBOT_R * 2)
            p.setFont(robot_font)
            p.drawText(
                QtCore.QRectF(self._rx - _ROBOT_R, self._ry - _ROBOT_R,
                              _ROBOT_R * 2, _ROBOT_R * 2),
                QtCore.Qt.AlignmentFlag.AlignCenter, "🤖",
            )

            # Direction indicator — white arrow tip extending from robot center
            a = math.radians(self._heading)
            ex = self._rx + math.cos(a) * (_ROBOT_R + 8)
            ey = self._ry + math.sin(a) * (_ROBOT_R + 8)
            p.setPen(QtGui.QPen(QtCore.Qt.GlobalColor.white, 2))
            p.drawLine(QtCore.QPointF(self._rx, self._ry), QtCore.QPointF(ex, ey))
        except Exception:
            pass


# ─── confidence bars widget ──────────────────────────────────────────────────

class ConfidenceBarsWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._probs: dict = {}

    def set_probs(self, probs: dict):
        self._probs = dict(sorted(probs.items()))
        self.update()

    def paintEvent(self, event):
        try:
            if not self._probs:
                p = QtGui.QPainter(self)
                p.setPen(QtGui.QColor("#aaa"))
                font = QtGui.QFont()
                font.setPixelSize(13)
                p.setFont(font)
                p.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter,
                           "No prediction yet — do a gesture!")
                return

            p = QtGui.QPainter(self)
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            w, h = self.width(), self.height()
            n = len(self._probs)
            best = max(self._probs, key=self._probs.get)

            margin = 6
            label_w = 120
            pct_w = 50
            total_spacing = margin * (n + 1)
            bar_h = max(10, (h - total_spacing) // n)
            bar_area_w = w - margin * 2 - label_w - pct_w

            font = QtGui.QFont()
            font.setPixelSize(13)
            p.setFont(font)

            for i, (label, prob) in enumerate(self._probs.items()):
                y = margin + i * (bar_h + margin)
                bar_px = int(bar_area_w * prob)
                bar_color = QtGui.QColor("#27ae60") if label == best else QtGui.QColor("#aaaaaa")

                # Background track
                p.setPen(QtCore.Qt.PenStyle.NoPen)
                p.setBrush(QtGui.QColor("#e0e0e0"))
                p.drawRoundedRect(margin + label_w, y, bar_area_w, bar_h, 4, 4)

                # Filled bar
                if bar_px > 0:
                    p.setBrush(bar_color)
                    p.drawRoundedRect(margin + label_w, y, bar_px, bar_h, 4, 4)

                # Label — replace underscores with spaces, title case
                display_label = label.replace("_", " ").title()
                p.setPen(QtGui.QColor("#333333"))
                p.drawText(
                    QtCore.QRectF(margin, y, label_w - 4, bar_h),
                    QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter,
                    display_label,
                )

                # Percentage
                p.drawText(
                    QtCore.QRectF(margin + label_w + bar_area_w + 4, y, pct_w - 4, bar_h),
                    QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
                    f"{prob:.0%}",
                )
        except Exception:
            pass


# ─── maze widget ─────────────────────────────────────────────────────────────

class MazeWidget(QtWidgets.QWidget):
    won = QtCore.pyqtSignal()

    def __init__(self, rows=_MAZE_ROWS, cols=_MAZE_COLS, parent=None):
        super().__init__(parent)
        self._rows = rows
        self._cols = cols
        self._maze_num = 1
        self._walls = _generate_maze(rows, cols, random.randint(0, 99999))

        # Player state
        self._pr = 0
        self._pc = 0
        self._facing = _E
        self._path = [(0, 0)]
        self._won = False
        self._bump = False
        self._moves = 0

        # Timer for clearing bump flash
        self._bump_timer = QtCore.QTimer(self)
        self._bump_timer.setSingleShot(True)
        self._bump_timer.timeout.connect(self._clear_bump)

        self.setMinimumSize(200, 150)

    def _clear_bump(self):
        self._bump = False
        self.update()

    def new_maze(self, rows: int, cols: int):
        """Change grid size and generate a brand-new random maze."""
        self._rows = rows
        self._cols = cols
        self._maze_num = 1
        self._walls = _generate_maze(rows, cols, random.randint(0, 99999))
        self._reset_player()
        self.update()

    def reset(self):
        """Keep the same grid size but generate a fresh random maze."""
        self._walls = _generate_maze(self._rows, self._cols, random.randint(0, 99999))
        self._maze_num += 1
        self._reset_player()
        self.update()

    def _reset_player(self):
        self._pr = 0
        self._pc = 0
        self._facing = _E
        self._path = [(0, 0)]
        self._won = False
        self._bump = False
        self._moves = 0
        if self._bump_timer.isActive():
            self._bump_timer.stop()

    @property
    def star_rating(self) -> int:
        """1–3 stars based on move efficiency vs. minimum possible path."""
        min_path = self._rows + self._cols - 2   # theoretical minimum cells
        if self._moves <= int(min_path * 2.5):
            return 3
        if self._moves <= int(min_path * 4.5):
            return 2
        return 1

    @property
    def facing_label(self):
        return _LABEL.get(self._facing, "East →")

    @property
    def facing_arrow(self):
        return _ARROW.get(self._facing, "→")

    def apply_gesture(self, gesture: str) -> str:
        """Apply a gesture to the maze player. Returns a feedback string."""
        if self._won:
            return "You already won! Press Reset to play again. 🎉"

        if gesture == "swipe_left":
            self._facing = _TURN_L[self._facing]
            self._moves += 1
            self.update()
            return f"Turned left! Now facing {_LABEL[self._facing]}"

        elif gesture == "swipe_right":
            self._facing = _TURN_R[self._facing]
            self._moves += 1
            self.update()
            return f"Turned right! Now facing {_LABEL[self._facing]}"

        elif gesture == "push":
            # Check if there's a wall in the facing direction
            if self._walls[self._pr][self._pc] & self._facing:
                # Wall in the way — flash bump
                self._bump = True
                self._bump_timer.start(300)
                self.update()
                return f"Oops! There's a wall to the {_LABEL[self._facing].split()[0]}. Try turning!"
            else:
                # Move forward
                nr = self._pr + _DR[self._facing]
                nc = self._pc + _DC[self._facing]
                self._pr = nr
                self._pc = nc
                self._moves += 1
                if (nr, nc) not in self._path:
                    self._path.append((nr, nc))
                self.update()
                if nr == self._rows - 1 and nc == self._cols - 1:
                    self._won = True
                    self.won.emit()
                    return f"🎉 You reached the goal in {self._moves} moves!"
                return f"Moved {_LABEL[self._facing].split()[0]}! Keep going!"

        elif gesture == "idle":
            return "Idle — no move made. Do a swipe or push!"

        else:
            return f"Unknown gesture: {gesture}"

    def paintEvent(self, event):
        try:
            p = QtGui.QPainter(self)
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            w, h = self.width(), self.height()

            # Soft gradient background
            grad = QtGui.QLinearGradient(0, 0, 0, h)
            grad.setColorAt(0, QtGui.QColor("#f0f4ff"))
            grad.setColorAt(1, QtGui.QColor("#e8edf8"))
            p.fillRect(0, 0, w, h, grad)

            top_margin = 32

            cell_w = (w - 4) // self._cols
            cell_h = (h - top_margin - 4) // self._rows
            cell_sz = min(cell_w, cell_h)
            wall_thick = max(2, cell_sz // 10)

            grid_w = cell_sz * self._cols
            grid_h = cell_sz * self._rows
            ox = (w - grid_w) // 2
            oy = top_margin + (h - top_margin - grid_h) // 2

            # Header: "Maze #N  •  Moves: M"
            hdr_font = QtGui.QFont()
            hdr_font.setPixelSize(13)
            hdr_font.setBold(True)
            p.setFont(hdr_font)
            p.setPen(QtGui.QColor("#34495e"))
            p.drawText(
                QtCore.QRectF(0, 4, w, top_margin - 4),
                QtCore.Qt.AlignmentFlag.AlignCenter,
                f"Maze #{self._maze_num}   •   Moves: {self._moves}",
            )

            path_set = set(self._path)

            # Cell fills
            for r in range(self._rows):
                for c in range(self._cols):
                    cx = ox + c * cell_sz
                    cy = oy + r * cell_sz
                    if r == self._rows - 1 and c == self._cols - 1:
                        color = QtGui.QColor("#ffeaa7")   # goal — warm gold
                    elif (r, c) == (self._pr, self._pc):
                        color = QtGui.QColor("#dfe6fd")   # player cell — soft blue
                    elif (r, c) in path_set:
                        color = QtGui.QColor("#c8e6ff")   # visited — light teal-blue
                    else:
                        color = QtGui.QColor("#ffffff")
                    p.fillRect(cx + 1, cy + 1, cell_sz - 1, cell_sz - 1, color)

            # Walls
            wall_pen = QtGui.QPen(QtGui.QColor("#2c3e50"), wall_thick)
            wall_pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            p.setPen(wall_pen)
            for r in range(self._rows):
                for c in range(self._cols):
                    cx = ox + c * cell_sz
                    cy = oy + r * cell_sz
                    wf = self._walls[r][c]
                    if wf & _N: p.drawLine(cx, cy, cx + cell_sz, cy)
                    if wf & _S: p.drawLine(cx, cy + cell_sz, cx + cell_sz, cy + cell_sz)
                    if wf & _W: p.drawLine(cx, cy, cx, cy + cell_sz)
                    if wf & _E: p.drawLine(cx + cell_sz, cy, cx + cell_sz, cy + cell_sz)

            # Outer border (slightly thicker)
            border_pen = QtGui.QPen(QtGui.QColor("#1a2a3a"), wall_thick + 1)
            p.setPen(border_pen)
            p.drawRect(ox, oy, grid_w, grid_h)

            # Goal star
            gcx = ox + (self._cols - 1) * cell_sz
            gcy = oy + (self._rows - 1) * cell_sz
            gfnt = QtGui.QFont()
            gfnt.setPixelSize(max(10, int(cell_sz * 0.55)))
            p.setFont(gfnt)
            p.drawText(
                QtCore.QRectF(gcx + 1, gcy + 1, cell_sz - 2, cell_sz - 2),
                QtCore.Qt.AlignmentFlag.AlignCenter, "⭐",
            )

            # Player — bump flash
            pcx = ox + self._pc * cell_sz
            pcy = oy + self._pr * cell_sz
            if self._bump:
                p.fillRect(pcx + 2, pcy + 2, cell_sz - 3, cell_sz - 3,
                           QtGui.QColor(231, 76, 60, 140))

            # Robot emoji
            rfnt = QtGui.QFont()
            rfnt.setPixelSize(max(8, int(cell_sz * 0.50)))
            p.setFont(rfnt)
            p.setPen(QtGui.QColor("#2c3e50"))
            emoji_h = int(cell_sz * 0.60)
            p.drawText(
                QtCore.QRectF(pcx + 1, pcy + 2, cell_sz - 2, emoji_h),
                QtCore.Qt.AlignmentFlag.AlignCenter, "🤖",
            )

            # Direction arrow
            afnt = QtGui.QFont()
            afnt.setPixelSize(max(6, int(cell_sz * 0.20)))
            afnt.setBold(True)
            p.setFont(afnt)
            p.setPen(QtGui.QColor("#2980b9"))
            p.drawText(
                QtCore.QRectF(pcx + 1, pcy + emoji_h, cell_sz - 2, cell_sz - emoji_h - 2),
                QtCore.Qt.AlignmentFlag.AlignCenter, _ARROW.get(self._facing, "→"),
            )

            # Win overlay
            if self._won:
                p.fillRect(ox, oy, grid_w, grid_h, QtGui.QColor(39, 174, 96, 150))
                stars = "⭐" * self.star_rating
                wfnt = QtGui.QFont()
                wfnt.setPixelSize(max(14, min(26, grid_w // 9)))
                wfnt.setBold(True)
                p.setFont(wfnt)
                p.setPen(QtGui.QColor("#ffffff"))
                p.drawText(QtCore.QRectF(ox, oy, grid_w, grid_h * 0.40),
                           QtCore.Qt.AlignmentFlag.AlignCenter, "🎉 You Did It!")
                sfnt = QtGui.QFont()
                sfnt.setPixelSize(max(12, min(22, grid_w // 10)))
                p.setFont(sfnt)
                p.drawText(QtCore.QRectF(ox, oy + grid_h * 0.40, grid_w, grid_h * 0.30),
                           QtCore.Qt.AlignmentFlag.AlignCenter, stars)
                mfnt = QtGui.QFont()
                mfnt.setPixelSize(max(10, min(16, grid_w // 13)))
                p.setFont(mfnt)
                p.drawText(QtCore.QRectF(ox, oy + grid_h * 0.70, grid_w, grid_h * 0.30),
                           QtCore.Qt.AlignmentFlag.AlignCenter,
                           f"{self._moves} moves — press Reset for a new maze!")
        except Exception:
            pass


# ─── main tab ────────────────────────────────────────────────────────────────

class TestTab(QtWidgets.QWidget):
    # emits (gesture: str, confidence: float, threshold: float, actual: str | None)
    prediction_made = QtCore.pyqtSignal(str, float, float, object)
    # emits (model_name: str, classes: list) when a model is loaded
    model_loaded = QtCore.pyqtSignal(str, list)
    # ── gamification signals ──────────────────────────────────────────────────
    # single-mode prediction (gesture, confidence)
    gesture_tested = QtCore.pyqtSignal(str, float)
    # robosoccer non-idle gesture applied
    soccer_gesture_applied = QtCore.pyqtSignal(str)
    # maze won (stars, moves)
    maze_solved = QtCore.pyqtSignal(int, int)

    _MODE_SINGLE = 0
    _MODE_RS = 1
    _MODE_MAZE = 2

    def __init__(self):
        super().__init__()
        self.setObjectName("test_root")
        self.setStyleSheet(TEST_STYLE)

        self._model = None
        self._hf_processor = None
        self._id2label: dict = {}

        self._frame_buf: deque = deque(maxlen=50)   # 5 s at 10 fps — holds >1 full epoch
        self._capturing = False
        self._capture_frames: list = []

        self._inference_running = False
        self._infer_thread = None
        self._infer_worker = None

        self._rs_timer = None
        self._rs_tick_count = 0
        self._rs_speed = _BASE_SPEED
        self._rs_burst_remaining = 0
        self._rs_cooldown_ticks = 0

        self._maze_timer = None
        self._maze_tick_count = 0
        self._maze_cooldown_ticks = 0
        self._mazes_solved = 0
        # 0=Easy(4×5), 1=Medium(5×7), 2=Hard(7×9)
        self._maze_difficulty = 1

        self._model_load_thread = None
        self._model_load_worker = None
        self._last_frames: list = []   # frames used for the most recent inference

        self._cache_probs: dict    = {}   # last high-confidence prediction probs
        self._cache_remaining: int = 0    # frames left to reuse the cached result

        self._anim_timer = None
        self._anim_steps = 0
        self._anim_step_px = 0.0

        # pending single-prediction state (awaiting confirmation)
        self._pending_gesture = ""
        self._pending_conf = 0.0
        self._pending_threshold = 0.0

        self._mode = self._MODE_SINGLE

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
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(7)

        heading = QtWidgets.QLabel("Test Gestures")
        heading.setObjectName("test_heading")
        layout.addWidget(heading)

        subtitle = QtWidgets.QLabel("Load your model, then pick a game mode!")
        subtitle.setObjectName("test_subtitle")
        layout.addWidget(subtitle)

        layout.addWidget(self._divider())

        # ── model section ──
        load_btn = QtWidgets.QPushButton("📂  Load Model")
        load_btn.setStyleSheet("""
            QPushButton {
                background: #eaf2fb; border: 2px solid #1a3a5c; border-radius: 8px;
                padding: 7px 12px; font-size: 14px;
                color: #1a3a5c; font-weight: bold;
            }
            QPushButton:hover { background: #d6eaf8; }
        """)
        load_btn.clicked.connect(self._load_model)
        layout.addWidget(load_btn)

        self._model_lbl = QtWidgets.QLabel("No model loaded.")
        self._model_lbl.setObjectName("test_model_lbl")
        self._model_lbl.setWordWrap(True)
        layout.addWidget(self._model_lbl)

        # Classes chip strip — shown after model loads
        self._classes_frame = QtWidgets.QFrame()
        self._classes_frame.setStyleSheet(
            "QFrame { background: #eef2f8; border-radius: 6px; border: 1px solid #d0d8e8; }"
        )
        self._classes_frame.setVisible(False)
        cf = QtWidgets.QVBoxLayout(self._classes_frame)
        cf.setContentsMargins(8, 6, 8, 6)
        cf.setSpacing(2)
        self._classes_title = QtWidgets.QLabel("Classes")
        self._classes_title.setStyleSheet(
            "font-size: 11px; font-weight: bold; color: #1a3a5c; border: none;"
        )
        cf.addWidget(self._classes_title)
        self._classes_lbl = QtWidgets.QLabel("")
        self._classes_lbl.setWordWrap(True)
        self._classes_lbl.setStyleSheet(
            "font-size: 12px; color: #333; border: none; line-height: 1.4;"
        )
        cf.addWidget(self._classes_lbl)
        layout.addWidget(self._classes_frame)

        layout.addWidget(self._divider())

        # ── mode picker ──
        mode_lbl = self._flbl("Choose a Mode")
        layout.addWidget(mode_lbl)
        layout.addWidget(self._build_mode_picker())

        layout.addWidget(self._divider())

        # ── stacked mode controls ──
        self._controls_stack = QtWidgets.QStackedWidget()
        self._controls_stack.addWidget(self._build_single_controls())   # index 0
        self._controls_stack.addWidget(self._build_rs_controls())       # index 1
        self._controls_stack.addWidget(self._build_maze_controls())     # index 2
        self._controls_stack.setCurrentIndex(0)
        layout.addWidget(self._controls_stack)

        layout.addStretch()

        layout.addWidget(HintCard([
            "🎯 Single Prediction: do a gesture, hit Capture — the model tells you what it thinks!",
            "⚽ RoboSoccer mode: the model watches you continuously. Swipe to steer, push to speed up!",
            "🌀 Maze Game: navigate through the maze using gestures. Swipe to turn, push to move forward!",
            "📊 Confidence: a percentage showing how sure the model is. 90%+ means very confident!",
            "🔧 Confidence threshold: the robot only reacts if the model is at least this confident.",
            "📈 The bar chart shows every gesture's score — one tall bar means confident!",
            "🔁 Getting wrong predictions? Go back to Collect, add more samples, then retrain!",
        ]))

        self._status = QtWidgets.QLabel("")
        self._status.setObjectName("test_status")
        self._status.setWordWrap(True)
        self._status.setMinimumHeight(40)
        layout.addWidget(self._status)

        layout.addWidget(self._build_confirm_widget())

        return _scrollable_left(panel, width=310)

    def _build_mode_picker(self):
        """3 stacked buttons for mode selection."""
        container = QtWidgets.QWidget()
        lyt = QtWidgets.QVBoxLayout(container)
        lyt.setContentsMargins(0, 0, 0, 0)
        lyt.setSpacing(4)

        specs = [
            (self._MODE_SINGLE, "🎯  Try a Gesture"),
            (self._MODE_RS,     "⚽  RoboSoccer"),
            (self._MODE_MAZE,   "🌀  Maze Game"),
        ]
        self._mode_btns = []
        for mode, label in specs:
            btn = QtWidgets.QPushButton(label)
            btn.setFixedHeight(36)
            btn.clicked.connect(lambda checked, m=mode: self._set_mode(m))
            self._mode_btns.append(btn)
            lyt.addWidget(btn)

        self._update_mode_btn_styles()
        return container

    def _update_mode_btn_styles(self):
        active_style = """
            QPushButton {
                background: #1a3a5c; color: white;
                border: none; border-radius: 7px;
                font-size: 13px; font-weight: bold;
                text-align: left; padding: 0 12px;
            }
        """
        inactive_style = """
            QPushButton {
                background: #edf0f5; color: #444;
                border: 1px solid #d0d6e0; border-radius: 7px;
                font-size: 13px;
                text-align: left; padding: 0 12px;
            }
            QPushButton:hover { background: #dde3ec; }
        """
        for i, btn in enumerate(self._mode_btns):
            btn.setStyleSheet(active_style if i == self._mode else inactive_style)

    def _build_single_controls(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(6)

        layout.addWidget(self._flbl("Capture Duration (s)"))
        self._duration = QtWidgets.QDoubleSpinBox()
        self._duration.setRange(1.0, 10.0)
        self._duration.setValue(3.0)
        self._duration.setSingleStep(0.5)
        layout.addWidget(self._duration)

        self._capture_btn = QtWidgets.QPushButton("🎤  Capture & Predict")
        self._capture_btn.setObjectName("capture_btn")
        self._capture_btn.setEnabled(False)
        self._capture_btn.clicked.connect(self._start_capture)
        layout.addWidget(self._capture_btn)

        return widget

    def _build_rs_controls(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(6)

        layout.addWidget(self._flbl("Confidence Threshold"))
        self._conf_threshold = QtWidgets.QDoubleSpinBox()
        self._conf_threshold.setRange(0.1, 1.0)
        self._conf_threshold.setValue(0.6)
        self._conf_threshold.setSingleStep(0.05)
        layout.addWidget(self._conf_threshold)

        self._rs_start_btn = QtWidgets.QPushButton("▶  Start RoboSoccer")
        self._rs_start_btn.setObjectName("rs_start_btn")
        self._rs_start_btn.setEnabled(False)
        self._rs_start_btn.clicked.connect(self._start_robosoccer)
        layout.addWidget(self._rs_start_btn)

        self._rs_stop_btn = QtWidgets.QPushButton("■  Stop")
        self._rs_stop_btn.setObjectName("rs_stop_btn")
        self._rs_stop_btn.setVisible(False)
        self._rs_stop_btn.clicked.connect(self._stop_robosoccer)
        layout.addWidget(self._rs_stop_btn)

        return widget

    def _build_maze_controls(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(6)

        # ── Gesture guide ─────────────────────────────────────────────────────
        guide = QtWidgets.QFrame()
        guide.setStyleSheet("""
            QFrame { background:#f5eef8; border:1px solid #c39bd3; border-radius:8px; }
            QLabel { border:none; font-size:12px; color:#5b2c6f; }
        """)
        gl = QtWidgets.QVBoxLayout(guide)
        gl.setContentsMargins(10, 7, 10, 7)
        gl.setSpacing(2)
        title_lbl = QtWidgets.QLabel("Gesture Controls")
        title_lbl.setStyleSheet("font-weight:bold; font-size:12px; color:#5b2c6f; border:none;")
        gl.addWidget(title_lbl)
        for gesture, action in [("👈 Swipe Left", "Turn left"),
                                 ("👉 Swipe Right", "Turn right"),
                                 ("👋 Push", "Move forward"),
                                 ("🤚 Idle", "Wait / skip")]:
            row_w = QtWidgets.QWidget()
            row_w.setStyleSheet("background:transparent;")
            rl = QtWidgets.QHBoxLayout(row_w)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(4)
            rl.addWidget(QtWidgets.QLabel(gesture))
            arrow = QtWidgets.QLabel(f"→ {action}")
            arrow.setStyleSheet("color:#8e44ad; border:none; font-size:12px;")
            rl.addWidget(arrow)
            rl.addStretch()
            gl.addWidget(row_w)
        layout.addWidget(guide)

        # ── Difficulty picker ──────────────────────────────────────────────────
        layout.addWidget(self._flbl("Difficulty"))
        diff_row = QtWidgets.QWidget()
        diff_row.setStyleSheet("background:transparent;")
        dr = QtWidgets.QHBoxLayout(diff_row)
        dr.setContentsMargins(0, 0, 0, 0)
        dr.setSpacing(4)
        self._diff_btns = []
        for i, (label, size) in enumerate([("Easy\n4×5", (4, 5)),
                                            ("Medium\n5×7", (5, 7)),
                                            ("Hard\n7×9",  (7, 9))]):
            btn = QtWidgets.QPushButton(label)
            btn.setFixedHeight(44)
            btn.clicked.connect(lambda checked, idx=i: self._set_difficulty(idx))
            self._diff_btns.append(btn)
            dr.addWidget(btn)
        layout.addWidget(diff_row)
        self._update_diff_btn_styles()

        # ── Facing + solved stats ──────────────────────────────────────────────
        info_row = QtWidgets.QHBoxLayout()
        info_row.setSpacing(6)
        self._facing_lbl = QtWidgets.QLabel("Facing: East →")
        self._facing_lbl.setStyleSheet("font-size:12px; font-weight:bold; color:#8e44ad;")
        info_row.addWidget(self._facing_lbl, 1)
        self._solved_lbl = QtWidgets.QLabel("Solved: 0")
        self._solved_lbl.setStyleSheet(
            "font-size:12px; font-weight:bold; color:#27ae60;"
            "background:#eafaf1; border:1px solid #a9dfbf; border-radius:5px; padding:2px 6px;"
        )
        info_row.addWidget(self._solved_lbl)
        layout.addLayout(info_row)

        # ── Confidence threshold ───────────────────────────────────────────────
        layout.addWidget(self._flbl("Min. confidence to act"))
        self._maze_conf_threshold = QtWidgets.QDoubleSpinBox()
        self._maze_conf_threshold.setRange(0.1, 1.0)
        self._maze_conf_threshold.setValue(0.55)
        self._maze_conf_threshold.setSingleStep(0.05)
        layout.addWidget(self._maze_conf_threshold)

        # ── Start / Stop / Reset ───────────────────────────────────────────────
        self._maze_start_btn = QtWidgets.QPushButton("▶  Start Maze")
        self._maze_start_btn.setObjectName("rs_start_btn")
        self._maze_start_btn.setEnabled(False)
        self._maze_start_btn.clicked.connect(self._start_maze)
        layout.addWidget(self._maze_start_btn)

        self._maze_stop_btn = QtWidgets.QPushButton("■  Stop")
        self._maze_stop_btn.setObjectName("rs_stop_btn")
        self._maze_stop_btn.setVisible(False)
        self._maze_stop_btn.clicked.connect(self._stop_maze)
        layout.addWidget(self._maze_stop_btn)

        self._maze_reset_btn = QtWidgets.QPushButton("↺  New Maze")
        self._maze_reset_btn.setObjectName("maze_reset_btn")
        self._maze_reset_btn.clicked.connect(self._maze_reset)
        layout.addWidget(self._maze_reset_btn)

        return widget

    def _build_confirm_widget(self):
        """Inline confirmation row shown after each Single Prediction."""
        self._confirm_widget = QtWidgets.QFrame()
        self._confirm_widget.setStyleSheet("""
            QFrame {
                background: #fffbee;
                border: 1px solid #f0c84a;
                border-radius: 6px;
            }
            QLabel { border: none; }
        """)
        cl = QtWidgets.QVBoxLayout(self._confirm_widget)
        cl.setContentsMargins(10, 8, 10, 8)
        cl.setSpacing(6)

        q_lbl = QtWidgets.QLabel("What gesture did you actually perform?")
        q_lbl.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #5a4000;"
        )
        q_lbl.setWordWrap(True)
        cl.addWidget(q_lbl)

        # Button container — rebuilt when a model loads
        self._confirm_btns_widget = QtWidgets.QWidget()
        self._confirm_btns_widget.setStyleSheet("background: transparent;")
        self._confirm_btns_layout = QtWidgets.QGridLayout(self._confirm_btns_widget)
        self._confirm_btns_layout.setContentsMargins(0, 0, 0, 0)
        self._confirm_btns_layout.setSpacing(4)
        cl.addWidget(self._confirm_btns_widget)

        self._confirm_widget.setVisible(False)
        return self._confirm_widget

    def _rebuild_confirm_buttons(self, class_names: list):
        """Recreate the per-class + Skip buttons (called after model load)."""
        while self._confirm_btns_layout.count():
            item = self._confirm_btns_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        all_btns = list(class_names) + ["Skip"]
        cols = min(3, len(all_btns))
        for idx, name in enumerate(all_btns):
            is_skip = (name == "Skip")
            btn = QtWidgets.QPushButton(name.replace("_", " ") if not is_skip else "Skip")
            if is_skip:
                btn.setStyleSheet("""
                    QPushButton {
                        background: #f5f5f5;
                        border: 1px solid #ccc;
                        border-radius: 5px;
                        padding: 5px 8px;
                        font-size: 12px;
                        color: #888;
                    }
                    QPushButton:hover { background: #ececec; }
                """)
                btn.clicked.connect(lambda: self._on_confirm(None))
            else:
                btn.setStyleSheet("""
                    QPushButton {
                        background: white;
                        border: 1px solid #1a3a5c;
                        border-radius: 5px;
                        padding: 5px 8px;
                        font-size: 12px;
                        color: #1a3a5c;
                    }
                    QPushButton:hover { background: #e8f0fb; }
                """)
                btn.clicked.connect(
                    lambda checked, g=name: self._on_confirm(g)
                )
            row, col = divmod(idx, cols)
            self._confirm_btns_layout.addWidget(btn, row, col)

    def _on_confirm(self, actual):
        """Called when student picks their actual gesture or clicks Skip."""
        self._confirm_widget.setVisible(False)
        self.prediction_made.emit(
            self._pending_gesture,
            self._pending_conf,
            self._pending_threshold,
            actual,
        )

    def _build_right(self):
        panel = QtWidgets.QWidget()
        panel.setStyleSheet("background: #f0f2f5;")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Stacked game area
        self._game_stack = QtWidgets.QStackedWidget()
        self._field = SoccerFieldWidget()
        self._field.setMinimumHeight(200)
        self._game_stack.addWidget(self._field)       # index 0

        self._maze_widget = MazeWidget()
        self._maze_widget.setMinimumHeight(200)
        self._maze_widget.won.connect(self._on_maze_won)
        self._game_stack.addWidget(self._maze_widget)  # index 1

        self._game_stack.setCurrentIndex(0)
        layout.addWidget(self._game_stack, 3)

        # ── Bottom row: spectrogram preview + confidence bars ──────────────────
        bottom_row = QtWidgets.QHBoxLayout()
        bottom_row.setSpacing(8)

        # Spectrogram preview panel
        spec_frame = QtWidgets.QFrame()
        spec_frame.setStyleSheet(
            "QFrame { background: #1a1a2e; border-radius: 8px; border: 1px solid #2e4a6a; }"
        )
        spec_frame.setFixedWidth(180)
        sf = QtWidgets.QVBoxLayout(spec_frame)
        sf.setContentsMargins(8, 6, 8, 6)
        sf.setSpacing(4)

        spec_title = QtWidgets.QLabel("Last Gesture")
        spec_title.setStyleSheet(
            "font-size: 11px; font-weight: bold; color: #aac4e0; border: none;"
        )
        spec_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        sf.addWidget(spec_title)

        self._spec_img_lbl = QtWidgets.QLabel()
        self._spec_img_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._spec_img_lbl.setMinimumHeight(100)
        self._spec_img_lbl.setStyleSheet("border: none; background: transparent;")
        sf.addWidget(self._spec_img_lbl, 1)

        self._spec_hint_lbl = QtWidgets.QLabel("Do a gesture\nto see it here")
        self._spec_hint_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._spec_hint_lbl.setStyleSheet(
            "font-size: 11px; color: #4a6a8a; border: none; background: transparent;"
        )
        self._spec_hint_lbl.setWordWrap(True)
        sf.addWidget(self._spec_hint_lbl)

        self._spec_gesture_lbl = QtWidgets.QLabel("")
        self._spec_gesture_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._spec_gesture_lbl.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: #3498db; border: none; background: transparent;"
        )
        sf.addWidget(self._spec_gesture_lbl)

        bottom_row.addWidget(spec_frame)

        # Confidence bars panel
        bars_frame = QtWidgets.QFrame()
        bars_frame.setStyleSheet(
            "QFrame { background: white; border-radius: 8px; border: 1px solid #ddd; }"
        )
        bf = QtWidgets.QVBoxLayout(bars_frame)
        bf.setContentsMargins(10, 8, 10, 8)
        bf.setSpacing(4)

        pred_lbl = QtWidgets.QLabel("Model Confidence")
        pred_lbl.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #1a3a5c; border: none;"
        )
        bf.addWidget(pred_lbl)

        self._bars = ConfidenceBarsWidget()
        bf.addWidget(self._bars, 1)

        bottom_row.addWidget(bars_frame, 1)
        layout.addLayout(bottom_row, 1)

        return panel

    def _update_spectrogram_preview(self, frames: list):
        """Render the frames as a spectrogram image and display it."""
        if not frames:
            return
        try:
            pil_img = _frames_to_pil(frames)
            if pil_img is None:
                return
            # PIL → numpy → QImage → QPixmap
            import numpy as np
            arr = np.array(pil_img)
            h, w, ch = arr.shape
            bytes_per_line = ch * w
            qimg = QtGui.QImage(
                arr.tobytes(), w, h, bytes_per_line,
                QtGui.QImage.Format.Format_RGB888,
            )
            # Scale to fit the label while keeping square aspect ratio
            available = self._spec_img_lbl.height()
            target_sz = max(60, available)
            pixmap = QtGui.QPixmap.fromImage(qimg).scaled(
                target_sz, target_sz,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            self._spec_img_lbl.setPixmap(pixmap)
            self._spec_hint_lbl.setVisible(False)
        except Exception:
            pass

    def _flbl(self, text):
        l = QtWidgets.QLabel(text)
        l.setObjectName("test_field_label")
        return l

    def _divider(self):
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setStyleSheet("color: #eee; margin: 2px 0;")
        return line

    # ── mode switching ────────────────────────────────────────────────────────

    def _set_mode(self, mode: int):
        # Stop any running game when switching modes
        if self._mode == self._MODE_RS and mode != self._MODE_RS:
            if self._rs_timer and self._rs_timer.isActive():
                self._stop_robosoccer()
        if self._mode == self._MODE_MAZE and mode != self._MODE_MAZE:
            if self._maze_timer and self._maze_timer.isActive():
                self._stop_maze()

        self._mode = mode
        self._update_mode_btn_styles()
        self._controls_stack.setCurrentIndex(mode)

        # Switch game display: maze gets index 1, everything else index 0
        if mode == self._MODE_MAZE:
            self._game_stack.setCurrentIndex(1)
        else:
            self._game_stack.setCurrentIndex(0)

        # Hide confirm widget when switching modes
        self._confirm_widget.setVisible(False)

        # Update facing label when switching to maze
        if mode == self._MODE_MAZE:
            self._update_facing_label()

        self._set_status("")

    # ── model loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        os.makedirs(MODELS_ROOT, exist_ok=True)
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Model Folder", MODELS_ROOT
        )
        if not path:
            return

        self._set_status("Loading model…", "#e67e22")
        self._model_lbl.setText("Loading…")
        self._capture_btn.setEnabled(False)
        self._rs_start_btn.setEnabled(False)
        self._maze_start_btn.setEnabled(False)

        worker = ModelLoadWorker(path)
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.success.connect(self._on_model_loaded)
        worker.error.connect(self._on_model_load_error)
        worker.done.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(worker.deleteLater)
        thread.start()
        self._model_load_thread = thread
        self._model_load_worker = worker

    def _on_model_loaded(self, model, processor, id2label, name, classes):
        self._model        = model
        self._hf_processor = processor
        self._id2label     = id2label
        self._model_lbl.setText(f"✓  {name}")
        self._model_lbl.setStyleSheet(
            "font-size: 12px; color: #27ae60; font-family: monospace;"
        )
        self._classes_lbl.setText("  ·  ".join(classes))
        self._classes_title.setText(f"Classes  ({len(classes)})")
        self._classes_frame.setVisible(True)
        self._capture_btn.setEnabled(True)
        self._rs_start_btn.setEnabled(True)
        self._maze_start_btn.setEnabled(True)
        self._set_status(f"✅ Loaded — {len(id2label)} classes ready!", "#27ae60")
        self._field.reset()
        self._rebuild_confirm_buttons(classes)
        self._confirm_widget.setVisible(False)
        self.model_loaded.emit(name, classes)

    def _on_model_load_error(self, msg):
        self._model        = None
        self._hf_processor = None
        self._id2label     = {}
        self._model_lbl.setText(f"✗  {msg}")
        self._model_lbl.setStyleSheet(
            "font-size: 12px; color: #c0392b; font-family: monospace;"
        )
        self._classes_frame.setVisible(False)
        self._capture_btn.setEnabled(False)
        self._rs_start_btn.setEnabled(False)
        self._maze_start_btn.setEnabled(False)
        self._set_status("⚠️ Failed to load model. Try a different folder.", "#c0392b")

    # ── single prediction ─────────────────────────────────────────────────────

    def _start_capture(self):
        self._capture_frames = []
        self._capturing = True
        self._capture_btn.setEnabled(False)
        self._set_status("🔴 Capturing… do your gesture now!", "#c0392b")
        QtCore.QTimer.singleShot(
            int(self._duration.value() * 1000), self._capture_done
        )

    def _capture_done(self):
        self._capturing = False
        frames = list(self._capture_frames)
        if len(frames) < 5:
            self._capture_btn.setEnabled(True)
            self._set_status("⚠️ No radar frames — is the radar connected?", "#c0392b")
            return
        self._set_status("🔍 Running inference…", "#e67e22")
        self._run_inference(frames, mode="single")

    # ── robosoccer ────────────────────────────────────────────────────────────

    def _start_robosoccer(self):
        self._field.reset()
        self._rs_tick_count = 0
        self._rs_speed = _BASE_SPEED
        self._rs_burst_remaining = 0
        self._inference_running = False
        self._rs_cooldown_ticks = 0
        self._rs_start_btn.setVisible(False)
        self._rs_stop_btn.setVisible(True)
        self._set_status("⚽ RoboSoccer running — do gestures to steer!", "#27ae60")
        self._rs_timer = QtCore.QTimer(self)
        self._rs_timer.timeout.connect(self._on_rs_tick)
        self._rs_timer.start(_TICK_MS)

    def _stop_robosoccer(self):
        if self._rs_timer:
            self._rs_timer.stop()
            self._rs_timer = None
        self._rs_start_btn.setVisible(True)
        self._rs_stop_btn.setVisible(False)
        self._set_status("RoboSoccer stopped.", "#888")

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

        if self._rs_cooldown_ticks > 0:
            self._rs_cooldown_ticks -= 1

        if (self._rs_tick_count % _INFER_EVERY == 0
                and not self._inference_running
                and self._rs_cooldown_ticks == 0):
            frames = list(self._frame_buf)
            if len(frames) >= 5:
                self._run_inference(frames, mode="robosoccer")

    def _apply_rs_gesture(self, gesture):
        rx, ry = self._field.robot_pos
        hdg = self._field.heading
        _COOLDOWN = 120   # ~4 s at 30 fps
        if gesture == "swipe_left":
            self._field.set_robot(rx, ry, hdg - 30.0)
            self._rs_cooldown_ticks = _COOLDOWN
            self._frame_buf.clear()   # discard stale frames so they can't re-fire
        elif gesture == "swipe_right":
            self._field.set_robot(rx, ry, hdg + 30.0)
            self._rs_cooldown_ticks = _COOLDOWN
            self._frame_buf.clear()
        elif gesture == "push":
            self._rs_speed = _BASE_SPEED + _PUSH_BURST / _PUSH_BURST_TICKS
            self._rs_burst_remaining = _PUSH_BURST_TICKS
            self._rs_cooldown_ticks = _COOLDOWN
            self._frame_buf.clear()
        # idle: no action, no cooldown

    # ── animation (single mode) ───────────────────────────────────────────────

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

    # ── maze (continuous) ─────────────────────────────────────────────────────

    def _start_maze(self):
        rows, cols = self._DIFFICULTY_SIZES[self._maze_difficulty]
        self._maze_widget.new_maze(rows, cols)
        self._update_facing_label()
        self._maze_tick_count = 0
        self._maze_cooldown_ticks = 0
        self._inference_running = False
        self._maze_start_btn.setVisible(False)
        self._maze_stop_btn.setVisible(True)
        self._set_status("Maze running — do a gesture to move!", "#8e44ad")
        self._maze_timer = QtCore.QTimer(self)
        self._maze_timer.timeout.connect(self._on_maze_tick)
        self._maze_timer.start(_TICK_MS)

    def _stop_maze(self):
        if self._maze_timer:
            self._maze_timer.stop()
            self._maze_timer = None
        self._maze_start_btn.setVisible(True)
        self._maze_stop_btn.setVisible(False)
        self._set_status("Maze stopped. Press Start to try again!", "#555")

    def _on_maze_tick(self):
        self._maze_tick_count += 1
        if self._maze_cooldown_ticks > 0:
            self._maze_cooldown_ticks -= 1
        if (self._maze_tick_count % _MAZE_INFER_EVERY == 0
                and not self._inference_running
                and self._maze_cooldown_ticks == 0):
            # Use only the most recent frames so the window closely matches
            # the gesture duration used when the model was trained (~3 s).
            # Older frames in the buffer are pre-gesture idle and dilute the signal.
            frames = list(self._frame_buf)[-_MAZE_CAPTURE_FRAMES:]
            if len(frames) >= _MAZE_MIN_FRAMES:
                self._run_inference(frames, mode="maze")

    def _maze_reset(self):
        if self._maze_timer and self._maze_timer.isActive():
            self._stop_maze()
        self._maze_widget.reset()
        self._update_facing_label()
        self._maze_start_btn.setVisible(True)
        self._maze_stop_btn.setVisible(False)
        if self._model is not None:
            self._maze_start_btn.setEnabled(True)
        self._set_status("Maze reset! Press Start to play.", "#8e44ad")

    def _update_facing_label(self):
        try:
            self._facing_lbl.setText(f"Facing: {self._maze_widget.facing_label}")
        except Exception:
            pass

    def _on_maze_won(self):
        self._stop_maze()
        self._mazes_solved += 1
        self._solved_lbl.setText(f"Solved: {self._mazes_solved}")
        moves = self._maze_widget._moves
        star_count = self._maze_widget.star_rating
        stars = "⭐" * star_count
        self._set_status(
            f"🎉 Maze #{self._maze_widget._maze_num} done in {moves} moves! {stars}  Press ↺ New Maze to keep going.",
            "#27ae60",
        )
        self.maze_solved.emit(star_count, moves)

    # ── inference ─────────────────────────────────────────────────────────────

    def _run_inference(self, frames, mode):
        if self._inference_running:
            return
        if self._model is None:
            self._set_status("⚠️ No model loaded! Please load a model first.", "#c0392b")
            if mode == "single":
                self._capture_btn.setEnabled(True)
            # maze and robosoccer are continuous — keep running on their timer
            return

        # ── prediction cache ──────────────────────────────────────────────────
        # For continuous modes (robosoccer, maze), reuse a recent high-confidence
        # result for up to _PRED_CACHE_FRAMES ticks to reduce CPU load.
        if mode != "single" and self._cache_remaining > 0 and self._cache_probs:
            self._cache_remaining -= 1
            self._on_inference_result(self._cache_probs, mode)
            return

        self._last_frames = list(frames)   # remember for spectrogram preview
        self._inference_running = True

        worker = InferenceWorker(
            frames, self._model, self._hf_processor, self._id2label
        )
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.result.connect(lambda p, m=mode: self._on_inference_result(p, m))
        worker.error.connect(lambda msg, m=mode: self._on_inference_error(msg, m))
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
        threshold = self._conf_threshold.value()
        nice = best.replace("_", " ").title()

        # ── update prediction cache ───────────────────────────────────────────
        # High-confidence continuous predictions are cached so the next
        # _PRED_CACHE_FRAMES ticks can skip inference entirely, saving CPU.
        if mode != "single" and conf >= _PRED_CACHE_CONF:
            self._cache_probs     = dict(probs)
            self._cache_remaining = _PRED_CACHE_FRAMES
        elif mode != "single":
            self._cache_remaining = 0

        # Update spectrogram preview for every inference
        self._update_spectrogram_preview(self._last_frames)
        self._spec_gesture_lbl.setText(f"{nice} ({conf:.0%})")

        if mode == "single":
            self._pending_gesture = best
            self._pending_conf = conf
            self._pending_threshold = threshold
            # Color status by confidence
            if conf >= 0.70:
                color = "#27ae60"
            elif conf >= 0.40:
                color = "#e67e22"
            else:
                color = "#c0392b"
            self._set_status(f"✓  {nice}  ({conf:.0%})", color)
            self._capture_btn.setEnabled(True)
            self._animate_single(best)
            self._confirm_widget.setVisible(True)
            self.gesture_tested.emit(best, conf)

        elif mode == "robosoccer":
            self.prediction_made.emit(best, conf, threshold, None)
            if conf >= threshold and best != "idle":
                self._apply_rs_gesture(best)
                self.soccer_gesture_applied.emit(best)

        elif mode == "maze":
            maze_threshold = self._maze_conf_threshold.value()
            if conf < maze_threshold or best == "idle":
                # Not confident enough or idle — show quietly, keep listening
                self._set_status(f"Listening… ({nice} {conf:.0%})", "#888")
            else:
                feedback = self._maze_widget.apply_gesture(best)
                self._update_facing_label()
                color = "#27ae60" if "wall" not in feedback.lower() else "#e74c3c"
                self._set_status(f"{nice} ({conf:.0%}) — {feedback}", color)
                # Cooldown so we don't re-classify immediately after a gesture
                self._maze_cooldown_ticks = 90    # ~3 s at 30 fps
                self._frame_buf.clear()           # discard stale frames so they can't re-fire

    def _on_inference_error(self, msg, mode=None):
        self._inference_running = False
        first_line = msg.split("\n")[0]
        self._set_status(f"⚠️ {first_line}", "#c0392b")
        # Re-enable capture button only for one-shot modes
        current = mode if mode is not None else self._mode
        if current == "single" or current == self._MODE_SINGLE:
            self._capture_btn.setEnabled(True)
        # maze and robosoccer are continuous — they keep running on their timer

    # ── difficulty picker ─────────────────────────────────────────────────────

    _DIFFICULTY_SIZES = [(4, 5), (5, 7), (7, 9)]   # Easy, Medium, Hard

    def _set_difficulty(self, idx: int):
        self._maze_difficulty = idx
        self._update_diff_btn_styles()
        rows, cols = self._DIFFICULTY_SIZES[idx]
        self._maze_widget.new_maze(rows, cols)
        self._update_facing_label()

    def _update_diff_btn_styles(self):
        active = (
            "QPushButton {"
            "background:#8e44ad; color:white; border:none; border-radius:6px;"
            "font-size:11px; font-weight:bold;"
            "}"
        )
        inactive = (
            "QPushButton {"
            "background:#f5eef8; color:#5b2c6f;"
            "border:1px solid #c39bd3; border-radius:6px; font-size:11px;"
            "}"
            "QPushButton:hover { background:#ead9f5; }"
        )
        for i, btn in enumerate(self._diff_btns):
            btn.setStyleSheet(active if i == self._maze_difficulty else inactive)

    # ── status helper ─────────────────────────────────────────────────────────

    def _set_status(self, text, color="#27ae60"):
        try:
            self._status.setText(text)
            self._status.setStyleSheet(
                f"color: {color}; font-size: 13px; font-weight: bold;"
            )
        except Exception:
            pass
