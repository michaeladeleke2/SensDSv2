import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets
from scipy.ndimage import gaussian_filter

# Pull ground-truth constants directly from the processing module so
# there is only one place to change them.
from core.processing import (
    STFT_NFFT, COLS_PER_FRAME, DB_MIN, DB_MAX,
    MAX_VELOCITY, FRAME_TIME_S,
    get_method, set_method, method_freq_bins, method_cols_per_frame,
    method_max_velocity, METHOD_STFT, METHOD_INFINEON,
    get_dynamic_range_db, set_dynamic_range_db,
)

DISPLAY_SECONDS = 5   # seconds of rolling history to show

# ── Derived display constants ─────────────────────────────────────────────────
# Defaults describe the STFT method (the historical behaviour):
# COLS_PER_FRAME (= 2) new STFT columns arrive every frame (0.1 s = 10 fps)
# STFT_NFFT = 1024, STFT_SHIFT = 56, so COLS_PER_FRAME = 128 // 56 = 2
#
# The Infineon SDK method has different dimensions (512 bins, 1 column/frame),
# so the widget rebuilds its buffer and axes via set_method() when it changes.
FREQ_BINS        = STFT_NFFT                              # 1024 Doppler bins
COLS_PER_SECOND  = int(round(COLS_PER_FRAME / FRAME_TIME_S))  # 2/0.1 = 20 cols/s
BUFFER_WIDTH     = DISPLAY_SECONDS * COLS_PER_SECOND          # 100 cols = 5 s


def make_jet_colormap():
    positions = [0.0, 0.125, 0.375, 0.625, 0.875, 1.0]
    colors = [
        (0,   0,   143, 255),
        (0,   0,   255, 255),
        (0,   255, 255, 255),
        (255, 255, 0,   255),
        (255, 0,   0,   255),
        (128, 0,   0,   255),
    ]
    return pg.ColorMap(positions, colors)


class VisualizeTab(QtWidgets.QWidget):
    """
    The Visualize tab: a live SpectrogramWidget plus a selector for which
    spectrogram representation the whole app should use.

    Switching here changes the representation everywhere — live view, collected
    training PNGs, and inference input — because all three route through
    core.processing.epoch_spectrogram_db() / get_streaming_result().
    """

    _METHODS = [
        (METHOD_STFT,
         "STFT (original)",
         "Concatenates every chirp into one slow-time signal, sums a fixed "
         "upper-half range window, then slides a 256-sample Hanning STFT.\n"
         "1024 Doppler bins x ~2 columns per frame. MTI clutter filter.\n"
         "This is what all existing trained models were built on."),
        (METHOD_INFINEON,
         "Infineon SDK",
         "Builds a full Range-Doppler Map per frame, tracks the most energetic "
         "range bin (median-smoothed), and emits that Doppler slice.\n"
         "512 Doppler bins x 1 column per frame. No MTI — range tracking "
         "rejects clutter instead.\n"
         "Retrain before using a model with this representation."),
    ]

    method_changed = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._build_controls())

        self.spectrogram = SpectrogramWidget()
        layout.addWidget(self.spectrogram, 1)
        self._sync_noise_visibility()

    def _build_controls(self):
        bar = QtWidgets.QFrame()
        bar.setFixedHeight(44)
        bar.setStyleSheet(
            "QFrame { background: #12121c; border-bottom: 1px solid #2a2a3a; }"
        )
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(14, 0, 14, 0)
        row.setSpacing(10)

        lbl = QtWidgets.QLabel("Spectrogram method")
        lbl.setStyleSheet(
            "color: #9aa4b2; font-size: 12px; font-weight: bold; border: none;"
        )
        row.addWidget(lbl)

        self._combo = QtWidgets.QComboBox()
        self._combo.setFixedWidth(170)
        self._combo.setStyleSheet("""
            QComboBox {
                background: #1e1e2e; color: #e6e6e6;
                border: 1px solid #3a3a4a; border-radius: 5px;
                padding: 4px 8px; font-size: 12px;
            }
            QComboBox::drop-down { border: none; width: 18px; }
            QComboBox QAbstractItemView {
                background: #1e1e2e; color: #e6e6e6;
                selection-background-color: #2980b9;
            }
        """)
        current = get_method()
        for i, (key, label, tip) in enumerate(self._METHODS):
            self._combo.addItem(label, key)
            self._combo.setItemData(i, tip, QtCore.Qt.ItemDataRole.ToolTipRole)
            if key == current:
                self._combo.setCurrentIndex(i)
        self._combo.currentIndexChanged.connect(self._on_changed)
        row.addWidget(self._combo)

        # ── Noise-floor control (Infineon SDK method only) ────────────────────
        # Drives vmin = vmax - N in doppler_to_display_db().  Lower N clips more
        # of the noise floor to black; the right value depends on the room and
        # the radar gain, so it has to be tunable against live data.
        self._noise_lbl = QtWidgets.QLabel("Noise floor")
        self._noise_lbl.setStyleSheet(
            "color: #9aa4b2; font-size: 12px; font-weight: bold; border: none;"
        )
        row.addWidget(self._noise_lbl)

        self._noise_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._noise_slider.setRange(10, 60)
        self._noise_slider.setValue(int(round(get_dynamic_range_db())))
        self._noise_slider.setFixedWidth(140)
        self._noise_slider.setToolTip(
            "Dynamic range shown, in dB below the strongest return.\n"
            "This is the vmin term from the reference script: vmin = vmax - N.\n\n"
            "Lower  -> darker background, only the strongest motion survives\n"
            "Higher -> more of the noise floor becomes visible\n\n"
            "Tune this live with the radar connected and no gesture happening:\n"
            "reduce it until the background goes uniformly dark blue."
        )
        self._noise_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 4px; background: #3a3a4a; border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 12px; margin: -5px 0; border-radius: 6px;
                background: #5dade2;
            }
            QSlider::sub-page:horizontal {
                background: #2980b9; border-radius: 2px;
            }
        """)
        self._noise_slider.valueChanged.connect(self._on_noise_changed)
        row.addWidget(self._noise_slider)

        self._noise_val = QtWidgets.QLabel()
        self._noise_val.setFixedWidth(52)
        self._noise_val.setStyleSheet(
            "color: #5dade2; font-size: 12px; font-weight: bold;"
            " font-family: monospace; border: none;"
        )
        row.addWidget(self._noise_val)

        self._desc = QtWidgets.QLabel()
        self._desc.setStyleSheet("color: #6b7280; font-size: 11px; border: none;")
        row.addWidget(self._desc, 1)

        row.addStretch()
        self._update_noise_label()
        self._update_desc()
        return bar

    def _update_noise_label(self):
        self._noise_val.setText(f"-{self._noise_slider.value()} dB")

    def _on_noise_changed(self, value: int):
        set_dynamic_range_db(float(value))
        self._update_noise_label()

    def _sync_noise_visibility(self):
        """The noise-floor control only affects the Infineon SDK method."""
        show = self._combo.currentData() == METHOD_INFINEON
        for w in (self._noise_lbl, self._noise_slider, self._noise_val):
            w.setVisible(show)

    def _update_desc(self):
        key = self._combo.currentData()
        if key == METHOD_INFINEON:
            self._desc.setText(
                "512 bins x 1 col/frame  ·  range-tracked  ·  retrain models"
            )
        else:
            self._desc.setText(
                "1024 bins x 2 cols/frame  ·  MTI filtered  ·  current models"
            )

    def _on_changed(self):
        key = self._combo.currentData()
        set_method(key)
        self.spectrogram.set_method(key)
        self._update_desc()
        self._sync_noise_visibility()
        self.method_changed.emit(key)

    # Convenience passthrough so main_window can keep calling update_frame
    def update_frame(self, batch):
        self.spectrogram.update_frame(batch)


class SpectrogramWidget(pg.GraphicsLayoutWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._method = get_method()
        self._apply_method_dims()
        self._col = 0
        self._setup_plot()

    # ── method-dependent geometry ─────────────────────────────────────────────

    def _apply_method_dims(self):
        """Recompute buffer size / axis scaling for the active method."""
        self._freq_bins = method_freq_bins(self._method)
        cols_per_frame  = method_cols_per_frame(self._method)
        self._max_vel   = method_max_velocity(self._method)
        cols_per_second = max(1, int(round(cols_per_frame / FRAME_TIME_S)))
        self._width     = DISPLAY_SECONDS * cols_per_second
        self._buffer    = np.full((self._freq_bins, self._width),
                                  DB_MIN, dtype=np.float32)

    def set_method(self, method: str):
        """
        Switch representation live.  Rebuilds the rolling buffer and rescales
        the axes, since the two methods differ in both bin count and columns
        per frame (1024 x 2/frame for STFT vs 512 x 1/frame for Infineon SDK).
        """
        if method == self._method:
            return
        self._method = method
        self._apply_method_dims()
        self._col = 0
        self._rescale_axes()
        self._img.setImage(self._buffer.T, autoLevels=False)

    def _rescale_axes(self):
        time_scale = DISPLAY_SECONDS / self._width
        vel_scale  = (2 * self._max_vel) / self._freq_bins
        self._img.setTransform(
            QtGui.QTransform().scale(time_scale, vel_scale)
                              .translate(0, -self._freq_bins / 2)
        )
        self._plot.setXRange(0, DISPLAY_SECONDS, padding=0)
        self._plot.setYRange(-self._max_vel, self._max_vel, padding=0)

    def _setup_plot(self):
        self.setBackground('#00008F')

        plot = self.addPlot()
        self._plot = plot
        plot.setLabel('left', 'Velocity', units='m/s')
        plot.setLabel('bottom', 'Time', units='s')
        plot.hideButtons()
        plot.setMouseEnabled(x=False, y=False)

        ax_left = plot.getAxis('left')
        ax_left.setTextPen(pg.mkPen('w'))
        ax_left.setPen(pg.mkPen('w'))
        ax_bottom = plot.getAxis('bottom')
        ax_bottom.setTextPen(pg.mkPen('w'))
        ax_bottom.setPen(pg.mkPen('w'))

        self._img = pg.ImageItem()
        plot.addItem(self._img)

        colormap = make_jet_colormap()
        self._img.setColorMap(colormap)
        self._img.setLevels([DB_MIN, DB_MAX])

        self._rescale_axes()

        zero_line = pg.InfiniteLine(
            pos=0,
            angle=0,
            pen=pg.mkPen(color=(255, 255, 255, 60), width=1)
        )
        plot.addItem(zero_line)

    def update_frame(self, spectrogram_batch):
        # A stale batch from the previous method can still be in flight when
        # the user switches — drop it rather than crashing on the shape change.
        if spectrogram_batch.shape[0] != self._freq_bins:
            return
        n_cols = spectrogram_batch.shape[1]
        for i in range(n_cols):
            col = np.clip(spectrogram_batch[:, i], DB_MIN, DB_MAX).astype(np.float32)
            self._buffer[:, self._col] = col
            self._col = (self._col + 1) % self._width
        display = np.roll(self._buffer, -self._col, axis=1)
        # Keep float32 throughout — scipy gaussian_filter handles it natively
        # and avoids the 2× memory + compute cost of a float64 round-trip.
        display = gaussian_filter(display, sigma=[1.5, 0.8])
        self._img.setImage(display.T, autoLevels=False)