import os
import time
import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui
from scipy.ndimage import gaussian_filter
from core.processing import SpectrogramProcessor
from ui.spectrogram_widget import (
    DB_MIN, DB_MAX, FREQ_BINS, MAX_VELOCITY, FRAME_TIME_S, make_jet_colormap,
)
import pyqtgraph as pg


def _apply_jet_colormap(normalized):
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


COLLECT_STYLE = """
    QWidget#collect_root { background: #f0f2f5; }
    QWidget#left_panel {
        background: #ffffff;
        border-right: 1px solid #ddd;
    }
    QLabel#heading {
        font-size: 16px;
        font-weight: bold;
        color: #1a3a5c;
    }
    QLabel#field_label {
        font-size: 12px;
        font-weight: bold;
        color: #555;
    }
    QLineEdit, QSpinBox, QDoubleSpinBox {
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 5px 8px;
        font-size: 13px;
        background: white;
        max-height: 30px;
    }
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
        border: 1px solid #1a3a5c;
    }
    QComboBox {
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 5px 8px;
        font-size: 13px;
        background: white;
        max-height: 30px;
    }
    QComboBox:focus { border: 1px solid #1a3a5c; }
    QComboBox::drop-down { border: none; }
    QPushButton#collect_btn {
        background-color: #1a3a5c;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px;
        font-size: 13px;
        font-weight: bold;
    }
    QPushButton#collect_btn:hover { background-color: #245080; }
    QPushButton#collect_btn:disabled { background-color: #aaa; }
    QPushButton#stop_btn {
        background-color: #c0392b;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px;
        font-size: 13px;
        font-weight: bold;
    }
    QPushButton#stop_btn:hover { background-color: #e74c3c; }
    QProgressBar {
        border: none;
        border-radius: 4px;
        background: #e0e0e0;
        max-height: 8px;
        font-size: 0px;
    }
    QProgressBar::chunk {
        background-color: #1a3a5c;
        border-radius: 4px;
    }
    QLabel#status_msg {
        font-size: 12px;
        color: #27ae60;
        font-weight: bold;
    }
    QLabel#preview_heading {
        font-size: 14px;
        font-weight: bold;
        color: #1a3a5c;
    }
    QLabel#sample_count {
        font-size: 12px;
        color: #777;
    }
"""


class CaptureWorker(QtCore.QObject):
    countdown = QtCore.pyqtSignal(int)
    capturing = QtCore.pyqtSignal()
    # emits (spectrogram_array, n_frames_used) so the preview can compute
    # the correct time axis from the actual number of captured frames.
    sample_done = QtCore.pyqtSignal(np.ndarray, int)
    batch_done = QtCore.pyqtSignal()
    stopped = QtCore.pyqtSignal()

    def __init__(self, num_samples, duration_s, delay_s):
        super().__init__()
        self._num_samples = num_samples
        self._duration_s = duration_s
        self._delay_s = delay_s
        self._running = False
        self._collecting = False
        self._frames = []

    def feed_frame(self, frame):
        if self._collecting:
            self._frames.append(frame)

    @QtCore.pyqtSlot()
    def run(self):
        self._running = True
        for i in range(self._num_samples):
            if not self._running:
                self.stopped.emit()
                return

            for c in range(int(self._delay_s), 0, -1):
                if not self._running:
                    self.stopped.emit()
                    return
                self.countdown.emit(c)
                time.sleep(1)

            self._frames = []
            self._collecting = True
            self.capturing.emit()
            time.sleep(self._duration_s)
            self._collecting = False

            if self._frames:
                # Use ALL captured frames (not a fixed 10) so the spectrogram
                # covers the full gesture window and has a reasonable aspect ratio.
                n = len(self._frames)
                proc = SpectrogramProcessor(buffer_frames=n)
                result = None
                for f in self._frames:
                    result = proc.push_frame(f)
                if result is not None:
                    self.sample_done.emit(result, n)

        self.batch_done.emit()

    def stop(self):
        self._running = False
        self._collecting = False


class CollectTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("collect_root")
        self.setStyleSheet(COLLECT_STYLE)
        self._worker = None
        self._thread = None
        self._samples_collected = 0
        self._total_samples = 0
        self._save_dir = ""
        self._setup_ui()

    def _setup_ui(self):
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_left_panel())
        outer.addWidget(self._build_right_panel())

    def _build_left_panel(self):
        panel = QtWidgets.QWidget()
        panel.setObjectName("left_panel")
        panel.setFixedWidth(300)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(8)

        heading = QtWidgets.QLabel("Collect Gesture Samples")
        heading.setObjectName("heading")
        layout.addWidget(heading)

        layout.addWidget(self._divider())

        layout.addWidget(self._lbl("Student Name"))
        self._name_input = QtWidgets.QLineEdit()
        self._name_input.setPlaceholderText("e.g. Alex")
        layout.addWidget(self._name_input)

        layout.addWidget(self._lbl("Gesture Label"))
        self._gesture_combo = QtWidgets.QComboBox()
        self._gesture_combo.addItems([
            "push", "swipe_left", "swipe_right",
            "swipe_up", "swipe_down", "idle"
        ])
        self._gesture_combo.setEditable(True)
        self._gesture_combo.setInsertPolicy(
            QtWidgets.QComboBox.InsertPolicy.NoInsert
        )
        self._gesture_combo.lineEdit().setPlaceholderText(
            "Select or type a label..."
        )
        layout.addWidget(self._gesture_combo)

        layout.addWidget(self._divider())

        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(10)

        left_col = QtWidgets.QVBoxLayout()
        left_col.setSpacing(4)
        left_col.addWidget(self._lbl("# Samples"))
        self._num_samples = QtWidgets.QSpinBox()
        self._num_samples.setRange(1, 100)
        self._num_samples.setValue(10)
        left_col.addWidget(self._num_samples)
        row1.addLayout(left_col)

        right_col = QtWidgets.QVBoxLayout()
        right_col.setSpacing(4)
        right_col.addWidget(self._lbl("Duration (s)"))
        self._duration = QtWidgets.QDoubleSpinBox()
        self._duration.setRange(1.0, 10.0)
        self._duration.setValue(3.0)
        self._duration.setSingleStep(0.5)
        right_col.addWidget(self._duration)
        row1.addLayout(right_col)
        layout.addLayout(row1)

        layout.addWidget(self._lbl("Delay Between Samples (s)"))
        self._delay = QtWidgets.QSpinBox()
        self._delay.setRange(1, 10)
        self._delay.setValue(3)
        layout.addWidget(self._delay)

        layout.addWidget(self._divider())

        instructions = QtWidgets.QLabel(
            "💡  Enter your name and gesture label above, "
            "then click Start to begin collecting samples. "
            "Each sample will be previewed on the right after capture."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet(
            "font-size: 12px; color: #888; "
            "background: #f0f4ff; border-radius: 6px; padding: 10px;"
        )
        layout.addWidget(instructions)

        self._progress_bar = QtWidgets.QProgressBar()
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        self._status_msg = QtWidgets.QLabel("")
        self._status_msg.setObjectName("status_msg")
        self._status_msg.setWordWrap(True)
        self._status_msg.setMinimumHeight(36)
        layout.addWidget(self._status_msg)

        layout.addStretch()

        self._collect_btn = QtWidgets.QPushButton("▶  Start Batch Collection")
        self._collect_btn.setObjectName("collect_btn")
        self._collect_btn.clicked.connect(self._start_collection)
        layout.addWidget(self._collect_btn)

        self._stop_btn = QtWidgets.QPushButton("■  Stop")
        self._stop_btn.setObjectName("stop_btn")
        self._stop_btn.clicked.connect(self._stop_collection)
        self._stop_btn.setVisible(False)
        layout.addWidget(self._stop_btn)

        return panel

    def _build_right_panel(self):
        panel = QtWidgets.QWidget()
        panel.setStyleSheet("background: #f0f2f5;")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(8)

        preview_heading = QtWidgets.QLabel("Last Captured Gesture")
        preview_heading.setObjectName("preview_heading")
        layout.addWidget(preview_heading)

        # Use a proper plot widget (same approach as SpectrogramWidget) so the
        # preview has calibrated velocity/time axes and matches the live display.
        self._preview_widget = pg.GraphicsLayoutWidget()
        self._preview_widget.setBackground('#00008F')

        self._preview_plot = self._preview_widget.addPlot()
        self._preview_plot.setLabel('left', 'Velocity', units='m/s')
        self._preview_plot.setLabel('bottom', 'Time', units='s')
        self._preview_plot.hideButtons()
        self._preview_plot.setMouseEnabled(x=False, y=False)

        for axis_name in ('left', 'bottom'):
            ax = self._preview_plot.getAxis(axis_name)
            ax.setTextPen(pg.mkPen('w'))
            ax.setPen(pg.mkPen('w'))

        self._preview_img = pg.ImageItem()
        self._preview_plot.addItem(self._preview_img)

        self._preview_img.setColorMap(make_jet_colormap())
        self._preview_img.setLevels([DB_MIN, DB_MAX])

        zero_line = pg.InfiniteLine(
            pos=0, angle=0,
            pen=pg.mkPen(color=(255, 255, 255, 60), width=1)
        )
        self._preview_plot.addItem(zero_line)

        # Start with empty range so axes look clean before first capture.
        self._preview_plot.setYRange(-MAX_VELOCITY, MAX_VELOCITY, padding=0)
        self._preview_plot.setXRange(0, 3.0, padding=0)

        layout.addWidget(self._preview_widget)

        bottom_row = QtWidgets.QHBoxLayout()

        self._sample_count = QtWidgets.QLabel("No samples collected yet.")
        self._sample_count.setObjectName("sample_count")
        bottom_row.addWidget(self._sample_count)

        bottom_row.addStretch()

        self._open_folder_btn = QtWidgets.QPushButton("📂  Open Data Folder")
        self._open_folder_btn.setStyleSheet("""
            QPushButton {
                background: white;
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 5px 12px;
                font-size: 12px;
                color: #333;
            }
            QPushButton:hover { background: #f0f0f0; }
            QPushButton:disabled { color: #aaa; }
        """)
        self._open_folder_btn.setEnabled(False)
        self._open_folder_btn.clicked.connect(self._open_data_folder)
        bottom_row.addWidget(self._open_folder_btn)

        layout.addLayout(bottom_row)
        return panel

    def _lbl(self, text):
        l = QtWidgets.QLabel(text)
        l.setObjectName("field_label")
        return l

    def _divider(self):
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setStyleSheet("color: #eee; margin: 2px 0;")
        return line

    def on_raw_frame(self, frame):
        if self._worker:
            self._worker.feed_frame(frame)

    def _start_collection(self):
        name = self._name_input.text().strip()
        label = self._gesture_combo.currentText().strip()

        if not name:
            QtWidgets.QMessageBox.warning(self, "Missing Name",
                                          "Please enter a student name.")
            return
        if not label:
            QtWidgets.QMessageBox.warning(self, "Missing Label",
                                          "Please enter a gesture label.")
            return

        self._save_dir = os.path.join(
            os.path.expanduser("~"), "SensDSv2_data", name, label
        )
        os.makedirs(self._save_dir, exist_ok=True)

        # Continue numbering from where we left off instead of restarting at 0.
        import re as _re
        existing_nums = []
        for fname in os.listdir(self._save_dir):
            m = _re.match(r'sample_(\d+)\.npy', fname)
            if m:
                existing_nums.append(int(m.group(1)))
        self._samples_collected = max(existing_nums) if existing_nums else 0

        self._total_samples = self._num_samples.value()
        self._progress_bar.setMaximum(self._total_samples)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._collect_btn.setVisible(False)
        self._stop_btn.setVisible(True)
        offset_note = f"  (continuing from {self._samples_collected})" if self._samples_collected > 0 else ""
        self._status_msg.setText(f"Saving to ~/SensDSv2_data/{name}/{label}/{offset_note}")
        self._status_msg.setStyleSheet("font-size: 11px; color: #888;")

        self._worker = CaptureWorker(
            num_samples=self._total_samples,
            duration_s=self._duration.value(),
            delay_s=self._delay.value()
        )
        self._thread = QtCore.QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.countdown.connect(self._on_countdown)
        self._worker.capturing.connect(self._on_capturing)
        self._worker.sample_done.connect(self._on_sample_done)
        self._worker.batch_done.connect(self._on_batch_done)
        self._worker.stopped.connect(self._on_stopped)
        self._thread.start()

    def _stop_collection(self):
        if self._worker:
            self._worker.stop()

    def _on_countdown(self, count):
        self._status_msg.setText(f"Get ready... {count}")
        self._status_msg.setStyleSheet(
            "color: #e67e22; font-size: 22px; font-weight: bold;"
        )

    def _on_capturing(self):
        self._status_msg.setText("⬤  Perform your gesture NOW!")
        self._status_msg.setStyleSheet(
            "color: #c0392b; font-size: 22px; font-weight: bold;"
        )

    def _on_sample_done(self, spectrogram, n_frames):
        self._samples_collected += 1
        self._progress_bar.setValue(self._samples_collected)

        # --- Save raw numpy array ---
        npy_path = os.path.join(
            self._save_dir, f"sample_{self._samples_collected:03d}.npy"
        )
        np.save(npy_path, spectrogram)

        # --- Smooth and clip for display / PNG save ---
        smoothed = gaussian_filter(
            spectrogram.astype(np.float64), sigma=[2.0, 1.5]
        )
        display = np.clip(smoothed, DB_MIN, DB_MAX).astype(np.float32)

        # --- Save PNG ---
        # spectrogram shape: (FREQ_BINS, n_cols)
        # We want: width = n_cols (time), height = FREQ_BINS (velocity)
        # Flip vertically so positive velocity is at the top of the image.
        normalized = (display - DB_MIN) / (DB_MAX - DB_MIN)
        colored = _apply_jet_colormap(normalized)           # (FREQ_BINS, n_cols, 3)
        colored_flipped = np.ascontiguousarray(colored[::-1])  # positive vel → top
        n_cols = colored_flipped.shape[1]
        img_raw = QtGui.QImage(
            colored_flipped.tobytes(),
            n_cols,
            FREQ_BINS,
            n_cols * 3,
            QtGui.QImage.Format.Format_RGB888,
        )
        # Scale to a consistent landscape size (400×300) for the training pipeline.
        img_save = img_raw.scaled(
            400, 300,
            QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        png_path = os.path.join(
            self._save_dir, f"sample_{self._samples_collected:03d}.png"
        )
        img_save.save(png_path)

        # --- Update preview plot ---
        # Compute the actual gesture duration from the number of radar frames.
        duration = n_frames * FRAME_TIME_S
        n_cols_display = display.shape[1]
        time_scale = duration / n_cols_display
        vel_scale = (2 * MAX_VELOCITY) / FREQ_BINS

        self._preview_img.setTransform(
            QtGui.QTransform().scale(time_scale, vel_scale).translate(0, -FREQ_BINS / 2)
        )
        self._preview_img.setImage(display.T, autoLevels=False)
        self._preview_plot.setXRange(0, duration, padding=0)
        self._preview_plot.setYRange(-MAX_VELOCITY, MAX_VELOCITY, padding=0)

        self._open_folder_btn.setEnabled(True)
        self._sample_count.setText(
            f"Sample {self._samples_collected} of {self._total_samples} saved  •  {self._save_dir}"
        )
        self._status_msg.setText(f"✓ Sample {self._samples_collected} saved.")
        self._status_msg.setStyleSheet(
            "color: #27ae60; font-size: 18px; font-weight: bold;"
        )

    def _on_batch_done(self):
        self._cleanup_thread()
        self._status_msg.setText(
            f"✓ All {self._total_samples} samples collected and saved!"
        )
        self._status_msg.setStyleSheet(
            "color: #27ae60; font-size: 13px; font-weight: bold;"
        )
        self._collect_btn.setVisible(True)
        self._stop_btn.setVisible(False)

    def _on_stopped(self):
        self._cleanup_thread()
        self._status_msg.setText(
            f"Stopped. {self._samples_collected} samples saved."
        )
        self._status_msg.setStyleSheet("color: #888; font-size: 13px;")
        self._collect_btn.setVisible(True)
        self._stop_btn.setVisible(False)

    def _cleanup_thread(self):
        if self._thread:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None
        self._progress_bar.setVisible(False)

    def _open_data_folder(self):
        base_dir = os.path.join(os.path.expanduser("~"), "SensDSv2_data")
        folder = self._save_dir if self._save_dir and os.path.exists(
            self._save_dir) else base_dir
        os.makedirs(folder, exist_ok=True)
        import subprocess, sys
        if sys.platform == "darwin":
            subprocess.Popen(["open", folder])
        elif sys.platform == "win32":
            subprocess.Popen(["explorer", folder])
        else:
            subprocess.Popen(["xdg-open", folder])
