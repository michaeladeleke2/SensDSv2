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
from ui import app_colors, _scrollable_left, zoom_button_row

DISPLAY_SECONDS = 5   # seconds of rolling history to show

# ── Derived display constants ─────────────────────────────────────────────────
# Defaults describe the STFT method (the historical behavior):
# COLS_PER_FRAME (= 2) new STFT columns arrive every frame (0.1 s = 10 fps)
# STFT_NFFT = 1024, STFT_SHIFT = 56, so COLS_PER_FRAME = 128 // 56 = 2
#
# The Infineon SDK method has different dimensions (512 bins, 1 column/frame),
# so the widget rebuilds its buffer and axes via set_method() when it changes.
FREQ_BINS        = STFT_NFFT                              # 1024 Doppler bins
COLS_PER_SECOND  = int(round(COLS_PER_FRAME / FRAME_TIME_S))  # 2/0.1 = 20 cols/s
BUFFER_WIDTH     = DISPLAY_SECONDS * COLS_PER_SECOND          # 100 cols = 5 s


def _viz_style(c: dict) -> str:
    return f"""
    QWidget#viz_root {{ background: {c['bg']}; }}
    QWidget#left_panel {{
        background: {c['panel']};
        border-right: 1px solid {c['border']};
    }}
    QLabel#heading {{
        font-size: 16px; font-weight: bold; color: {c['accent']};
    }}
    QLabel#field_label {{
        font-size: 12px; font-weight: bold; color: {c['subtext']};
    }}
    QLabel#param_name {{ font-size: 12px; color: {c['text']}; }}
    QLabel#param_value {{
        font-size: 12px; font-weight: bold; color: {c['accent']};
        font-family: monospace;
    }}
    QLabel#desc {{ font-size: 11px; color: {c['faint']}; }}
    QLabel#readout {{
        font-size: 11px; color: {c['subtext']};
        font-family: monospace;
        background: {c['bg']};
        border: 1px solid {c['border']};
        border-radius: 5px;
        padding: 8px;
    }}
    QComboBox {{
        border: 1px solid {c['input_border']};
        border-radius: 5px; padding: 5px 8px; font-size: 13px;
        background: {c['input_bg']}; color: {c['text']}; max-height: 30px;
    }}
    QComboBox::drop-down {{ border: none; }}
    QDoubleSpinBox {{
        border: 1px solid {c['input_border']};
        border-radius: 5px; padding: 4px 6px; font-size: 12px;
        background: {c['input_bg']}; color: {c['text']}; max-height: 28px;
    }}
    QDoubleSpinBox:focus {{ border: 1px solid {c['accent']}; }}
    QComboBox QAbstractItemView {{
        background: {c['panel']}; color: {c['text']};
        border: 1px solid {c['border']};
        selection-background-color: {c['accent']}; selection-color: white;
    }}
    QPushButton#preset_btn {{
        background: {c['panel']};
        border: 1px solid {c['input_border']};
        border-radius: 5px; padding: 4px 6px;
        font-size: 11px; color: {c['accent']}; font-weight: bold;
    }}
    QPushButton#preset_btn:hover {{ background: {c['tab_hover']}; }}
    QSlider::groove:horizontal {{
        height: 4px; background: {c['progress_bg']}; border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        width: 13px; margin: -5px 0; border-radius: 6px;
        background: {c['accent']};
    }}
    QSlider::sub-page:horizontal {{
        background: {c['accent']}; border-radius: 2px;
    }}
    """


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
    The Visualize tab: a live SpectrogramWidget plus a parameters panel.

    The panel is the tuning surface for students working out what a spectrogram
    should look like. Axis controls are view-only and instant; the image
    controls affect how the data is mapped to color.

    Switching method changes the representation everywhere — live view,
    collected training PNGs, and inference input — because all three route
    through core.processing.
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
         "512 Doppler bins x 1 column per frame. No MTI; range tracking "
         "rejects clutter instead.\n"
         "Retrain before using a model with this representation."),
    ]

    # Velocity presets: hand gestures sit well inside the radar's Nyquist limit.
    _VEL_PRESETS = [("Gestures", 1.5), ("Wide", 3.0), ("Full", None)]

    method_changed = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._c = app_colors()
        self.setObjectName("viz_root")
        self.setStyleSheet(_viz_style(self._c))

        # Built first: the panel reads its slider limits off the widget.
        self.spectrogram = SpectrogramWidget()

        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_panel())

        # Spectrogram plus its zoom controls.
        right = QtWidgets.QWidget()
        rlay = QtWidgets.QVBoxLayout(right)
        rlay.setContentsMargins(0, 0, 8, 6)
        rlay.setSpacing(4)
        rlay.addWidget(self.spectrogram, 1)
        rlay.addWidget(zoom_button_row(
            self.spectrogram._plot, self._c,
            on_reset=self.spectrogram.reset_view,
            reset_tip="Back to the slider settings",
        ))
        outer.addWidget(right, 1)

        self._sync_method_widgets()
        self._update_readout()

    # ── panel ────────────────────────────────────────────────────────────────

    def _build_panel(self):
        panel = QtWidgets.QWidget()
        panel.setObjectName("left_panel")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(8)

        heading = QtWidgets.QLabel("Visualize")
        heading.setObjectName("heading")
        layout.addWidget(heading)
        layout.addWidget(self._divider())

        # ── method ───────────────────────────────────────────────────────────
        layout.addWidget(self._lbl("Spectrogram Method"))
        self._combo = QtWidgets.QComboBox()
        current = get_method()
        for i, (key, label, tip) in enumerate(self._METHODS):
            self._combo.addItem(label, key)
            self._combo.setItemData(i, tip, QtCore.Qt.ItemDataRole.ToolTipRole)
            if key == current:
                self._combo.setCurrentIndex(i)
        self._combo.currentIndexChanged.connect(self._on_method_changed)
        layout.addWidget(self._combo)

        self._desc = QtWidgets.QLabel()
        self._desc.setObjectName("desc")
        self._desc.setWordWrap(True)
        layout.addWidget(self._desc)

        layout.addWidget(self._divider())

        # ── axes ─────────────────────────────────────────────────────────────
        axes_lbl = self._lbl("Display Axes")
        layout.addWidget(axes_lbl)

        max_v = self.spectrogram.max_velocity()
        vmin0, vmax0 = self.spectrogram.velocity_limits()

        head = QtWidgets.QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        name = QtWidgets.QLabel("Velocity range (Y)")
        name.setObjectName("param_name")
        head.addWidget(name)
        head.addStretch()
        self._vel_value = QtWidgets.QLabel()
        self._vel_value.setObjectName("param_value")
        head.addWidget(self._vel_value)
        layout.addLayout(head)

        tip = ("Type the exact velocity limits to display, in m/s.\n\n"
               "The radar measures up to +/-%.2f m/s, but hand gestures only\n"
               "reach about +/-1.5 m/s, so most of the axis is empty at full\n"
               "range. The two ends are independent, so an asymmetric window\n"
               "such as -0.5 to 3.0 can be used when a gesture sits mostly on\n"
               "one side of zero.\n\n"
               "View only: no data is discarded and nothing is recomputed."
               % max_v)

        vel_row = QtWidgets.QHBoxLayout()
        vel_row.setSpacing(6)
        self._vel_min_box = self._vel_spin(-max_v, max_v, vmin0, tip)
        self._vel_max_box = self._vel_spin(-max_v, max_v, vmax0, tip)
        for label, box in (("Min", self._vel_min_box), ("Max", self._vel_max_box)):
            cap = QtWidgets.QLabel(label)
            cap.setObjectName("desc")
            vel_row.addWidget(cap)
            vel_row.addWidget(box, 1)
        layout.addLayout(vel_row)

        self._vel_min_box.valueChanged.connect(self._on_vel_limits_changed)
        self._vel_max_box.valueChanged.connect(self._on_vel_limits_changed)

        preset_row = QtWidgets.QHBoxLayout()
        preset_row.setSpacing(4)
        for name, value in self._VEL_PRESETS:
            btn = QtWidgets.QPushButton(name)
            btn.setObjectName("preset_btn")
            v = max_v if value is None else value
            btn.setToolTip(f"Set the velocity axis to +/-{v:.2f} m/s")
            btn.clicked.connect(lambda _, val=v: self._set_vel(val))
            preset_row.addWidget(btn)
        layout.addLayout(preset_row)

        self._time_slider, self._time_value = self._slider_row(
            layout, "Time window (X)",
            lo=1, hi=60, init=int(round(self.spectrogram.time_window())),
            tip=("Seconds of history shown across the X axis.\n\n"
                 "This also controls how fine the image looks: the radar\n"
                 "produces a fixed number of columns per second, so a short\n"                 "window spreads few columns over the full width and looks\n"
                 "blocky. Increase it for a finer picture.\n\n"
                 "Changing this resizes the scroll buffer, so the current\n"
                 "history clears and refills at the incoming frame rate."),
            on_change=self._on_time_changed,
        )

        layout.addWidget(self._divider())

        # ── image ────────────────────────────────────────────────────────────
        layout.addWidget(self._lbl("Image"))

        self._noise_lbl_row = QtWidgets.QWidget()
        nrow = QtWidgets.QVBoxLayout(self._noise_lbl_row)
        nrow.setContentsMargins(0, 0, 0, 0)
        nrow.setSpacing(8)
        self._noise_slider, self._noise_value = self._slider_row(
            nrow, "Noise floor",
            lo=10, hi=60, init=int(round(get_dynamic_range_db())),
            tip=("Dynamic range shown, in dB below the strongest return.\n"
                 "This is the vmin term from the reference script: "
                 "vmin = vmax - N.\n\n"
                 "Lower  -> darker background, only the strongest motion "
                 "survives\n"
                 "Higher -> more of the noise floor becomes visible\n\n"
                 "Tune with the radar connected and no gesture happening:\n"
                 "reduce it until the background goes uniformly dark blue."),
            on_change=self._on_noise_changed, suffix=" dB",
        )
        layout.addWidget(self._noise_lbl_row)

        self._smooth_slider, self._smooth_value = self._slider_row(
            layout, "Smoothing",
            lo=0, hi=30, init=15,
            tip=("Gaussian blur applied to the displayed image.\n"
                 "0 shows the raw bins; higher values look smoother but\n"
                 "blur fine Doppler detail."),
            on_change=self._on_smooth_changed, scale=10.0,
        )

        layout.addStretch()

        # ── derived readout ──────────────────────────────────────────────────
        layout.addWidget(self._divider())
        self._readout = QtWidgets.QLabel()
        self._readout.setObjectName("readout")
        self._readout.setWordWrap(True)
        layout.addWidget(self._readout)

        return _scrollable_left(panel, width=300)

    def _slider_row(self, parent_layout, title, lo, hi, init, tip,
                    on_change, scale=1.0, suffix=""):
        """Label + value readout + slider. Returns (slider, value_label)."""
        head = QtWidgets.QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        name = QtWidgets.QLabel(title)
        name.setObjectName("param_name")
        head.addWidget(name)
        head.addStretch()
        value = QtWidgets.QLabel()
        value.setObjectName("param_value")
        head.addWidget(value)
        parent_layout.addLayout(head)

        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(init)
        slider.setToolTip(tip)
        slider.valueChanged.connect(on_change)
        parent_layout.addWidget(slider)

        slider.setProperty("scale", scale)
        slider.setProperty("suffix", suffix)
        return slider, value

    def _lbl(self, text):
        lbl = QtWidgets.QLabel(text)
        lbl.setObjectName("field_label")
        return lbl

    def _divider(self):
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setStyleSheet(f"color: {self._c['divider']}; margin: 2px 0;")
        return line

    # ── handlers ─────────────────────────────────────────────────────────────

    def _vel_spin(self, lo, hi, init, tip):
        box = QtWidgets.QDoubleSpinBox()
        box.setDecimals(2)
        box.setSingleStep(0.1)
        box.setRange(lo, hi)
        box.setValue(init)
        box.setToolTip(tip)
        box.setKeyboardTracking(False)   # apply on Enter / focus-out, not per digit
        box.setMaximumWidth(82)          # keeps the pair inside the 300 px panel
        return box

    def _set_vel(self, value):
        """Symmetric preset helper."""
        self._apply_vel_limits(-abs(value), abs(value))

    def _apply_vel_limits(self, vmin, vmax):
        self.spectrogram.set_velocity_limits(vmin, vmax)
        lo, hi = self.spectrogram.velocity_limits()
        for box, v in ((self._vel_min_box, lo), (self._vel_max_box, hi)):
            box.blockSignals(True)
            box.setValue(v)
            box.blockSignals(False)
        self._vel_value.setText(f"{lo:.2f} .. {hi:.2f} m/s")
        self._update_readout()

    def _on_vel_limits_changed(self, _=None):
        # The widget clamps and enforces a minimum span; echo whatever it
        # settled on back into the boxes so they never disagree with the plot.
        self._apply_vel_limits(self._vel_min_box.value(),
                               self._vel_max_box.value())

    def _on_time_changed(self, raw):
        self.spectrogram.set_time_window(float(raw))
        self._time_value.setText(f"{raw} s")
        self._update_readout()

    def _on_noise_changed(self, raw):
        set_dynamic_range_db(float(raw))
        self._noise_value.setText(f"-{raw} dB")

    def _on_smooth_changed(self, raw):
        self.spectrogram.set_smoothing(raw / 10.0)
        self._smooth_value.setText("off" if raw == 0 else f"{raw / 10.0:.1f}")

    def _on_method_changed(self):
        key = self._combo.currentData()
        set_method(key)
        self.spectrogram.set_method(key)
        self._sync_method_widgets()
        self._update_readout()
        self.method_changed.emit(key)

    def _sync_method_widgets(self):
        key = self._combo.currentData()
        if key == METHOD_INFINEON:
            self._desc.setText(
                "512 bins x 1 col/frame  ·  range tracked  ·  retrain models"
            )
        else:
            self._desc.setText(
                "1024 bins x 2 cols/frame  ·  MTI filtered  ·  current models"
            )
        # The noise floor only affects the Infineon color mapping.
        self._noise_lbl_row.setVisible(key == METHOD_INFINEON)

        # Nyquist differs slightly between methods; keep the boxes in range.
        max_v = self.spectrogram.max_velocity()
        lo, hi = self.spectrogram.velocity_limits()
        for box, v in ((self._vel_min_box, lo), (self._vel_max_box, hi)):
            box.blockSignals(True)
            box.setRange(-max_v, max_v)
            box.setValue(v)
            box.blockSignals(False)

        # Refresh every value label to whatever the widget actually holds.
        self._vel_value.setText(f"{lo:.2f} .. {hi:.2f} m/s")
        self._time_value.setText(f"{int(self.spectrogram.time_window())} s")
        self._noise_value.setText(f"-{self._noise_slider.value()} dB")
        sm = self._smooth_slider.value()
        self._smooth_value.setText("off" if sm == 0 else f"{sm / 10.0:.1f}")

    def _update_readout(self):
        sp = self.spectrogram
        cps = sp.columns_per_second()
        bins = sp.freq_bins()
        vis = sp.visible_bins()
        vel_per_bin = (2 * sp.max_velocity()) / bins
        # Kept short: long lines here push the 300 px panel wider than it is.
        self._readout.setText(
            f"Doppler bins   {vis} of {bins} ({vis / bins:.0%})\n"
            f"Time step      {1000.0 / cps:.0f} ms/column\n"
            f"Velocity step  {vel_per_bin:.3f} m/s/bin\n"
            f"Screen width   {int(round(sp.time_window() * cps))} columns"
        )

    # Convenience passthrough so main_window can keep calling update_frame
    def update_frame(self, batch):
        self.spectrogram.update_frame(batch)


class SpectrogramWidget(pg.GraphicsLayoutWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._method = get_method()
        self._display_seconds = float(DISPLAY_SECONDS)
        self._smoothing = 1.5
        self._apply_method_dims()
        # Independent limits, not a symmetric +/- range: a push sits mostly
        # on one side of zero, so an asymmetric window can use the whole plot.
        self._vel_min = -self._max_vel
        self._vel_max = self._max_vel
        self._col = 0
        self._setup_plot()

    # ── method-dependent geometry ─────────────────────────────────────────────

    def _apply_method_dims(self):
        """Recompute buffer size / axis scaling for the active method."""
        self._freq_bins = method_freq_bins(self._method)
        cols_per_frame  = method_cols_per_frame(self._method)
        self._max_vel   = method_max_velocity(self._method)
        self._cols_per_second = max(1, int(round(cols_per_frame / FRAME_TIME_S)))
        self._width     = max(2, int(round(self._display_seconds
                                           * self._cols_per_second)))
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
        prev_max = self._max_vel
        self._method = method
        self._apply_method_dims()
        # Keep the user's zoom across the switch; the two methods have slightly
        # different Nyquist limits, so clamp rather than reset. A window that
        # was at full range stays at full range.
        was_full = (self._vel_min <= -prev_max + 1e-6
                    and self._vel_max >= prev_max - 1e-6)
        if was_full:
            self._vel_min, self._vel_max = -self._max_vel, self._max_vel
        else:
            self._vel_min = max(-self._max_vel, self._vel_min)
            self._vel_max = min(self._max_vel, self._vel_max)
        self._col = 0
        self._rescale_axes()
        self._img.setImage(self._buffer.T, autoLevels=False)

    # ── adjustable display parameters ────────────────────────────────────────

    _MIN_VEL_SPAN = 0.2      # m/s; keeps the axis from collapsing

    def set_velocity_limits(self, vmin: float, vmax: float):
        """
        Set the velocity (Y) axis to an explicit vmin..vmax in m/s.

        View-only: the buffer still holds every Doppler bin, so this is instant
        and never discards data. Values are clamped to the radar's Nyquist
        limit, and a minimum span is enforced so the axis cannot collapse.
        """
        lim = self._max_vel
        vmin = max(-lim, min(float(vmin), lim))
        vmax = max(-lim, min(float(vmax), lim))
        if vmax - vmin < self._MIN_VEL_SPAN:
            vmax = min(lim, vmin + self._MIN_VEL_SPAN)
            if vmax - vmin < self._MIN_VEL_SPAN:
                vmin = max(-lim, vmax - self._MIN_VEL_SPAN)
        self._vel_min, self._vel_max = vmin, vmax
        self._plot.setYRange(vmin, vmax, padding=0)

    def set_velocity_range(self, vmax: float):
        """Convenience for a symmetric window: -vmax .. +vmax."""
        v = abs(float(vmax))
        self.set_velocity_limits(-v, v)

    def velocity_limits(self):
        return self._vel_min, self._vel_max

    def velocity_range(self) -> float:
        """Half-span, for callers that only care how wide the window is."""
        return (self._vel_max - self._vel_min) / 2.0

    def max_velocity(self) -> float:
        return self._max_vel

    def set_time_window(self, seconds: float):
        """
        Set how many seconds of history the X axis shows.

        Unlike the velocity zoom this resizes the rolling buffer, so the
        existing history is dropped and refills at the incoming frame rate.
        """
        seconds = max(1.0, float(seconds))
        if abs(seconds - self._display_seconds) < 1e-6:
            return
        self._display_seconds = seconds
        self._apply_method_dims()
        self._col = 0
        self._rescale_axes()
        self._img.setImage(self._buffer.T, autoLevels=False)

    def time_window(self) -> float:
        return self._display_seconds

    def set_smoothing(self, sigma: float):
        """Gaussian blur applied to the displayed image; 0 disables it."""
        self._smoothing = max(0.0, float(sigma))

    def columns_per_second(self) -> int:
        return self._cols_per_second

    def freq_bins(self) -> int:
        return self._freq_bins

    def visible_bins(self) -> int:
        """How many Doppler bins fall inside the current velocity zoom."""
        span = self._vel_max - self._vel_min
        frac = span / (2 * self._max_vel) if self._max_vel else 1.0
        return max(1, int(round(self._freq_bins * frac)))

    def reset_view(self):
        """Restore the axes to the current slider settings."""
        self._rescale_axes()

    def _rescale_axes(self):
        time_scale = self._display_seconds / self._width
        vel_scale  = (2 * self._max_vel) / self._freq_bins
        self._img.setTransform(
            QtGui.QTransform().scale(time_scale, vel_scale)
                              .translate(0, -self._freq_bins / 2)
        )
        self._plot.setXRange(0, self._display_seconds, padding=0)
        self._plot.setYRange(self._vel_min, self._vel_max, padding=0)

    def _setup_plot(self):
        self.setBackground('#00008F')

        plot = self.addPlot()
        self._plot = plot
        plot.setLabel('left', 'Velocity', units='m/s')
        plot.setLabel('bottom', 'Time', units='s')
        plot.hideButtons()
        # Scroll to zoom, drag to pan. Reset restores the slider view.
        plot.setMouseEnabled(x=True, y=True)

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
        if self._smoothing > 0:
            display = gaussian_filter(
                display, sigma=[self._smoothing, self._smoothing * 0.55]
            )
        self._img.setImage(display.T, autoLevels=False)