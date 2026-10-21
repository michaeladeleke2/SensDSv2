"""
ui/curve_fit_tab.py

Freeze the live spectrogram, trace the ridge a moving target leaves behind, and
fit a function to it.

A Newton's cradle swinging toward and away from the radar traces a sinusoid, so
the fitted frequency predicts a pendulum length the student can check against
the real one with a ruler.

Team feedback, as in the web version:
  - the picture can be the range map (distance over time, range_map_live.py)
    with MTI on or off and a configurable distance, shared with Visualize,
  - an Exponential model,
  - Auto Fit: the curve is found in the picture, and its positive and negative
    peaks give the fit its starting point,
  - the result is shown as cards a student can read: the equation, what every
    letter means, and what the fit predicts.
"""

from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from core import range_map as RM
from core.curve_explain import explain_fit, explain_peaks
from core.curve_fit import (
    MODEL_DAMPED, MODEL_EXP, MODEL_LABELS, MODEL_LINEAR, MODEL_POLYNOMIAL,
    MODEL_SINUSOID, auto_trace, clean_trace, find_peaks, fit_curve,
    peak_start, snap_to_peak,
)
from core.processing import (
    FRAME_TIME_S, METHOD_INFINEON, get_method, method_cols_per_frame,
    method_freq_bins, method_max_velocity,
)
from core.reference_image import (
    REF_MAX_SPEED_M_S, current_jet_vmin, reference_rgb, reference_spectrogram,
)
from ui import app_colors, _scrollable_left
from ui.spectrogram_widget import (
    DB_MAX, DB_MIN, DISPLAY_SECONDS, make_jet_colormap,
)

TRACE_COLOR = "#ff00ff"
PEAK_UP = "#e8590c"
PEAK_DOWN = "#0ca678"
MIN_TRACE_POINTS = 10
STILL_BAND = 0.1        # m/s around 0 that Auto Fit ignores on the spectrogram
IDLE_TEXT = "Freeze the display, trace a curve (or press Auto Fit), then fit a model to it."

# Raw frames kept for the reference drawing: one window's worth.
FRAME_WINDOW = int(round(DISPLAY_SECONDS / FRAME_TIME_S))


def _curve_fit_style(c: dict) -> str:
    return f"""
    QWidget#curvefit_root {{ background: {c['bg']}; }}
    QWidget#left_panel {{
        background: {c['panel']};
        border-right: 1px solid {c['border']};
    }}
    QLabel#heading {{
        font-size: 16px;
        font-weight: bold;
        color: {c['accent']};
    }}
    QLabel#field_label {{
        font-size: 12px;
        font-weight: bold;
        color: {c['subtext']};
    }}
    QLabel#plot_heading {{
        font-size: 14px;
        font-weight: bold;
        color: {c['accent']};
    }}
    QLabel#note {{ font-size: 11px; color: {c['faint']}; }}
    QLabel#equation {{
        font-size: 13px;
        color: {c['text']};
        font-family: Menlo, Consolas, 'DejaVu Sans Mono', 'Liberation Mono', monospace;
    }}
    QLabel#physics {{
        font-size: 12px;
        color: {c['subtext']};
        font-family: Menlo, Consolas, 'DejaVu Sans Mono', 'Liberation Mono', monospace;
    }}
    QLabel#status_ok {{ font-size: 12px; color: {c['subtext']}; }}
    QScrollArea#cf_results {{
        background: {c['panel']}; border: 1px solid {c['border']}; border-radius: 8px;
    }}
    QWidget#cf_results_body {{ background: {c['panel']}; }}
    QLabel#cf_idle {{ font-size: 16px; color: {c['subtext']}; }}
    QLabel#cf_idle_err {{ font-size: 16px; color: #c0392b; font-weight: bold; }}
    QLabel#cf_title {{ font-size: 18px; font-weight: bold; color: {c['accent']}; }}
    QLabel#cf_badge_ok, QLabel#cf_badge_warn, QLabel#cf_badge_bad {{
        font-size: 14px; font-weight: bold; padding: 4px 12px;
        border-radius: 12px; border: 1px solid;
    }}
    QLabel#cf_badge_ok {{ color: #1f9d55; border-color: #1f9d55; }}
    QLabel#cf_badge_warn {{ color: #d97706; border-color: #d97706; }}
    QLabel#cf_badge_bad {{ color: #c53030; border-color: #c53030; }}
    QFrame#cf_eq_box {{
        background: {c['bg']}; border: 1px solid {c['border']}; border-radius: 10px;
    }}
    QLabel#cf_kicker {{
        font-size: 11px; font-weight: bold; color: {c['faint']}; letter-spacing: 1px;
    }}
    QLabel#cf_eq {{
        font-size: 22px; color: {c['text']};
        font-family: 'Cambria Math', 'STIX Two Math', Cambria, 'Times New Roman', serif;
    }}
    QLabel#cf_eq_nums {{
        font-size: 19px; color: {c['accent']};
        font-family: 'Cambria Math', 'STIX Two Math', Cambria, 'Times New Roman', serif;
    }}
    QLabel#cf_note {{ font-size: 14px; color: {c['subtext']}; }}
    QLabel#cf_section {{ font-size: 15px; font-weight: bold; color: {c['text']}; }}
    QLabel#cf_group {{ font-size: 13px; font-weight: bold; color: {c['accent']}; }}
    QFrame#cf_card {{ border: 1px solid {c['border']}; border-radius: 10px; background: {c['panel']}; }}
    QFrame#cf_card_pred {{ border: none; border-radius: 10px; background: {c['hint_bg']}; }}
    QLabel#cf_sym {{
        font-size: 24px; color: {c['accent']};
        font-family: 'Cambria Math', 'STIX Two Math', Cambria, 'Times New Roman', serif;
    }}
    QLabel#cf_sym_small {{
        font-size: 15px; color: {c['accent']};
        font-family: 'Cambria Math', 'STIX Two Math', Cambria, 'Times New Roman', serif;
    }}
    QLabel#cf_pname {{ font-size: 12px; font-weight: bold; color: {c['subtext']}; letter-spacing: 1px; }}
    QLabel#cf_val {{ font-size: 22px; font-weight: bold; color: {c['text']}; }}
    QLabel#cf_meaning {{ font-size: 14px; color: {c['subtext']}; }}
    QLabel#cf_peaks {{
        font-size: 14px; color: {c['text']}; background: {c['hint_bg']};
        border-radius: 10px; padding: 10px 14px;
    }}
    QComboBox {{
        border: 1px solid {c['input_border']}; border-radius: 5px; padding: 5px 8px;
        font-size: 13px; background: {c['input_bg']}; color: {c['text']}; max-height: 30px;
    }}
    QDoubleSpinBox {{
        border: 1px solid {c['input_border']}; border-radius: 5px; padding: 4px 6px;
        font-size: 12px; background: {c['input_bg']}; color: {c['text']}; max-height: 28px;
    }}
    QLabel#status_err {{ font-size: 12px; color: #c0392b; font-weight: bold; }}
    QRadioButton, QCheckBox {{
        font-size: 13px;
        color: {c['text']};
        spacing: 6px;
    }}
    QSpinBox {{
        border: 1px solid {c['input_border']};
        border-radius: 5px;
        padding: 5px 8px;
        font-size: 13px;
        background: {c['input_bg']};
        color: {c['text']};
        max-height: 30px;
    }}
    QSpinBox:focus {{ border: 1px solid {c['accent']}; }}
    QPushButton#primary_btn {{
        background-color: {c['accent']};
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px;
        font-size: 13px;
        font-weight: bold;
    }}
    QPushButton#primary_btn:hover {{ background-color: #245080; }}
    QPushButton#primary_btn:disabled {{ background-color: #aaa; }}
    QPushButton#primary_btn:checked {{ background-color: #c0392b; }}
    QPushButton#minor_btn {{
        background: {c['panel']};
        border: 1px solid {c['input_border']};
        border-radius: 5px;
        padding: 6px 10px;
        font-size: 12px;
        color: {c['accent']};
        font-weight: bold;
    }}
    QPushButton#minor_btn:hover {{ background: {c['tab_hover']}; }}
    QPushButton#minor_btn:disabled {{ color: {c['faint']}; }}
    """


class CurveFitTab(QtWidgets.QWidget):
    """Live spectrogram or range map, freehand or automatic trace, fitted model."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._c = app_colors()
        self.setObjectName("curvefit_root")
        self.setStyleSheet(_curve_fit_style(self._c))

        self._kind = "doppler"          # the picture: "doppler" (speed) or "range" (distance)
        self._method = get_method()
        self._apply_method_dims()
        self._col = 0
        self._frozen = False
        self._frozen_spec = None
        self._reference_frozen = False
        self._raw_frames = deque(maxlen=FRAME_WINDOW)
        self._tracing = False
        self._drawing = False
        self._points = []
        self._auto = False              # the points came from Auto Fit
        self._peaks = None
        # Range map: range_map_live.py's processor on a worker thread
        self._range_thread = None
        self._range_worker = None
        self._range_hist = None         # (FRAME_WINDOW, n_bins) dB, oldest first
        self._range_m = None

        self._setup_ui()
        self._rescale_axes()
        self._refresh_image()
        self._sync_buttons()
        RM.on_settings_changed(self._on_range_settings)

    # ── live buffer ──────────────────────────────────────────────────────────

    def _apply_method_dims(self):
        self._freq_bins = method_freq_bins(self._method)
        cols_per_frame = method_cols_per_frame(self._method)
        # The Infineon picture spans the reference script's velocity axis,
        # +/-4.11 m/s for the 0.3 ms chirp (the change note's speed axis); the
        # STFT method keeps its own.
        self._max_vel = (REF_MAX_SPEED_M_S if self._method == METHOD_INFINEON
                         else method_max_velocity(self._method))
        self._cols_per_second = max(1, int(round(cols_per_frame / FRAME_TIME_S)))
        self._width = max(2, int(round(DISPLAY_SECONDS * self._cols_per_second)))
        self._buffer = np.full((self._freq_bins, self._width), DB_MIN,
                               dtype=np.float32)
        self._time_scale = DISPLAY_SECONDS / self._width
        self._vel_scale = (2 * self._max_vel) / self._freq_bins

    def on_spectrogram_frame(self, batch):
        """New columns from the radar. Ignored while frozen."""
        if self._frozen:
            return
        batch = np.asarray(batch)
        if batch.ndim != 2 or batch.size == 0:
            return
        if batch.shape[0] != self._freq_bins:
            # The method changed elsewhere; rebuild at the new geometry.
            self._method = get_method()
            self._apply_method_dims()
            self._col = 0
            self._rescale_axes()
            if batch.shape[0] != self._freq_bins:
                return
        for i in range(batch.shape[1]):
            self._buffer[:, self._col] = np.clip(batch[:, i], DB_MIN, DB_MAX)
            self._col = (self._col + 1) % self._width
        if self._kind == "doppler":
            self._refresh_image()

    def on_raw_frame(self, frame):
        """Raw frames, kept so Freeze can redraw the window the reference way,
        and fed to the range map's processor when that is the picture."""
        if self._frozen:
            return
        self._raw_frames.append(np.array(frame, copy=True))
        if self._kind == "range" and self._range_worker is not None and self.isVisible():
            QtCore.QMetaObject.invokeMethod(
                self._range_worker, "on_raw_frame",
                QtCore.Qt.ConnectionType.QueuedConnection,
                QtCore.Q_ARG(np.ndarray, np.array(frame, copy=True)),
            )

    def _display_array(self):
        """The buffer in display order, oldest column first."""
        return np.roll(self._buffer, -self._col, axis=1)

    def _refresh_image(self):
        if self._kind == "range":
            self._draw_range_image()
            return
        self._img.setImage(self._display_array().T, autoLevels=False)

    def _rescale_axes(self):
        if self._kind == "range":
            s = RM.get_settings()
            dr = self._range_step()
            self._img.setTransform(QtGui.QTransform().scale(RM.FRAME_TIME_S, dr).translate(0, -0.5))
            self._plot.setXRange(0, DISPLAY_SECONDS, padding=0)
            self._plot.setYRange(s["min"], s["max"], padding=0)
            self._plot.setLabel('left', 'Distance from radar', units='m')
            return
        self._img.setTransform(
            QtGui.QTransform().scale(self._time_scale, self._vel_scale)
                              .translate(0, -self._freq_bins / 2)
        )
        self._plot.setXRange(0, DISPLAY_SECONDS, padding=0)
        self._plot.setYRange(-self._max_vel, self._max_vel, padding=0)
        self._plot.setLabel('left', 'Velocity', units='m/s')

    # ── range map ────────────────────────────────────────────────────────────

    def _range_step(self):
        if self._range_m is None:
            self._range_m = RM.range_axis()
        return float(self._range_m[1] - self._range_m[0])

    def _start_range(self):
        from ui.range_map_view import RangeWorker
        self._stop_range()
        self._range_step()
        self._range_hist = None
        self._range_thread = QtCore.QThread(self)
        self._range_worker = RangeWorker(FRAME_WINDOW, RM.mode())
        self._range_worker.moveToThread(self._range_thread)
        self._range_worker.history_ready.connect(self._on_range_history)
        self._range_worker.failed.connect(lambda msg: self._set_status(f"Range map: {msg}", error=True))
        self._range_thread.start()

    def _stop_range(self):
        if self._range_thread is not None:
            self._range_thread.quit()
            self._range_thread.wait(2000)
        self._range_thread = None
        self._range_worker = None

    def _on_range_history(self, history, n_filled, count):
        if self._frozen or self._kind != "range":
            return
        self._range_hist = history
        self._draw_range_image()

    def _draw_range_image(self):
        """The range map as the scripts colour it, over the distance shown."""
        if self._range_hist is None:
            self._img.clear()
            return
        s = RM.get_settings()
        lo_hi = RM.shown_bins(self._range_m, s["min"], s["max"])
        rgb = RM.range_rgb(self._range_hist, lo_hi, current_jet_vmin())
        self._img.setImage(np.ascontiguousarray(rgb), levels=(0, 255))

    def _on_range_settings(self, settings, changed):
        try:
            self._sync_range_inputs()
        except RuntimeError:
            RM.remove_listener(self._on_range_settings)
            return
        if self._kind != "range":
            return
        if changed["mti"]:
            self._set_kind("range")          # a new mode starts the history again
            return
        self._rescale_axes()
        self._draw_range_image()

    def _sync_range_inputs(self):
        s = RM.get_settings()
        for w, v in ((self._mti_check, s["mti"]), (self._dist_from, s["min"]), (self._dist_to, s["max"])):
            w.blockSignals(True)
            if isinstance(w, QtWidgets.QCheckBox):
                w.setChecked(v)
            else:
                w.setValue(v)
            w.blockSignals(False)

    def _set_kind(self, kind):
        """Switch the picture; the trace and any freeze are dropped."""
        if self._trace_btn.isChecked():
            self._trace_btn.setChecked(False)
        self._kind = kind
        self._frozen = False
        self._frozen_spec = None
        self._reference_frozen = False
        self._range_opts.setVisible(kind == "range")
        idx = self._kind_combo.findData(kind)
        if self._kind_combo.currentIndex() != idx:
            self._kind_combo.blockSignals(True)
            self._kind_combo.setCurrentIndex(idx)
            self._kind_combo.blockSignals(False)
        if kind == "range":
            self._start_range()
        else:
            self._stop_range()
            self._img.setLevels([DB_MIN, DB_MAX])
        self._clear_trace(quiet=True)
        self._rescale_axes()
        self._refresh_image()
        self._spec_heading.setText(self._live_heading())
        self._sync_buttons()
        self._set_status("Range map. Live." if kind == "range" else "Spectrogram. Live.")

    def _live_heading(self):
        if self._kind == "range":
            return f"Live Range Map (MTI {'on' if RM.mode() == 'mti' else 'off'})"
        return "Live Spectrogram"

    # ── layout ───────────────────────────────────────────────────────────────

    def _setup_ui(self):
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_left())
        outer.addWidget(self._build_right(), 1)

    def _build_left(self):
        panel = QtWidgets.QWidget()
        panel.setObjectName("left_panel")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(8)

        heading = QtWidgets.QLabel("Curve Fitting")
        heading.setObjectName("heading")
        layout.addWidget(heading)
        layout.addWidget(self._divider())

        layout.addWidget(self._lbl("Picture"))
        self._kind_combo = QtWidgets.QComboBox()
        self._kind_combo.addItem("Doppler spectrogram (speed)", "doppler")
        self._kind_combo.addItem("Range map (distance)", "range")
        self._kind_combo.setToolTip(
            "Doppler spectrogram: the target's speed toward or away from the radar over time.\n"
            "Range map: the target's distance from the radar over time.")
        self._kind_combo.currentIndexChanged.connect(
            lambda _: self._set_kind(self._kind_combo.currentData()))
        layout.addWidget(self._kind_combo)
        layout.addWidget(self._build_range_opts())

        layout.addWidget(self._divider())

        layout.addWidget(self._lbl("Capture"))
        self._freeze_btn = QtWidgets.QPushButton("❄  Freeze")
        self._freeze_btn.setObjectName("primary_btn")
        self._freeze_btn.clicked.connect(self._freeze)
        layout.addWidget(self._freeze_btn)

        self._resume_btn = QtWidgets.QPushButton("▶  Resume")
        self._resume_btn.setObjectName("minor_btn")
        self._resume_btn.clicked.connect(self._resume)
        layout.addWidget(self._resume_btn)

        layout.addWidget(self._divider())

        layout.addWidget(self._lbl("Trace"))
        self._trace_btn = QtWidgets.QPushButton("✎  Start Trace")
        self._trace_btn.setObjectName("primary_btn")
        self._trace_btn.setCheckable(True)
        self._trace_btn.toggled.connect(self._on_trace_toggled)
        layout.addWidget(self._trace_btn)

        self._clear_btn = QtWidgets.QPushButton("✕  Clear Trace")
        self._clear_btn.setObjectName("minor_btn")
        self._clear_btn.clicked.connect(lambda: self._clear_trace())
        layout.addWidget(self._clear_btn)

        self._snap_check = QtWidgets.QCheckBox("Snap to strongest signal")
        self._snap_check.setChecked(True)
        self._snap_check.setToolTip(
            "Move each traced point onto the brightest bin near it before\n"
            "fitting, so the fit follows the signal rather than how steady\n"
            "your hand was."
        )
        layout.addWidget(self._snap_check)

        self._auto_btn = QtWidgets.QPushButton("✨  Auto Fit")
        self._auto_btn.setObjectName("primary_btn")
        self._auto_btn.setToolTip(
            "Freezes the picture, finds the strongest signal at each moment,\n"
            "marks its high (positive) and low (negative) peaks, and uses them\n"
            "to start the fit of the model chosen below.")
        self._auto_btn.clicked.connect(self._auto_fit)
        layout.addWidget(self._auto_btn)
        auto_note = QtWidgets.QLabel("Auto Fit traces the curve for you and starts the fit from its high and low peaks.")
        auto_note.setObjectName("note")
        auto_note.setWordWrap(True)
        layout.addWidget(auto_note)

        layout.addWidget(self._divider())

        layout.addWidget(self._lbl("Model"))
        self._model_group = QtWidgets.QButtonGroup(self)
        for key, label in ((MODEL_SINUSOID, "Sinusoid"),
                           (MODEL_DAMPED, "Damped sinusoid"),
                           (MODEL_EXP, "Exponential"),
                           (MODEL_POLYNOMIAL, "Polynomial"),
                           (MODEL_LINEAR, "Linear")):
            btn = QtWidgets.QRadioButton(label)
            btn.setProperty("model_key", key)
            if key == MODEL_SINUSOID:
                btn.setChecked(True)
            self._model_group.addButton(btn)
            layout.addWidget(btn)
        self._model_group.buttonToggled.connect(self._on_model_changed)

        self._degree_row = QtWidgets.QWidget()
        drow = QtWidgets.QHBoxLayout(self._degree_row)
        drow.setContentsMargins(0, 0, 0, 0)
        drow.setSpacing(6)
        degree_lbl = QtWidgets.QLabel("Degree")
        degree_lbl.setObjectName("field_label")
        drow.addWidget(degree_lbl)
        self._degree_spin = QtWidgets.QSpinBox()
        self._degree_spin.setRange(1, 6)
        self._degree_spin.setValue(2)
        drow.addWidget(self._degree_spin, 1)
        self._degree_row.setVisible(False)
        layout.addWidget(self._degree_row)

        self._fit_btn = QtWidgets.QPushButton("📈  Fit Curve")
        self._fit_btn.setObjectName("primary_btn")
        self._fit_btn.setEnabled(False)
        self._fit_btn.clicked.connect(self._on_fit)
        layout.addWidget(self._fit_btn)

        self._status = QtWidgets.QLabel("")
        self._status.setObjectName("status_ok")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        layout.addStretch()
        return _scrollable_left(panel, width=300)

    def _build_range_opts(self):
        box = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self._mti_check = QtWidgets.QCheckBox("MTI filter (show only moving things)")
        self._mti_check.setToolTip(
            "MTI (moving target indication) removes everything that stands still,\n"
            "so walls and furniture do not hide the moving target.\n"
            "Changing it restarts the picture.")
        self._mti_check.toggled.connect(lambda on: RM.set_settings(mti=on))
        lay.addWidget(self._mti_check)
        lay.addWidget(self._lbl("Distance shown (m)"))
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        self._dist_from = self._dist_spin(0.0, RM.RANGE_MAX_M - 0.1, 0.0)
        self._dist_to = self._dist_spin(0.1, RM.RANGE_MAX_M, RM.RANGE_MAX_M)
        for label, spin in (("From", self._dist_from), ("To", self._dist_to)):
            cap = QtWidgets.QLabel(label)
            cap.setObjectName("note")
            row.addWidget(cap)
            row.addWidget(spin, 1)
        reset = QtWidgets.QPushButton("Reset")
        reset.setObjectName("minor_btn")
        reset.setToolTip(f"Show the whole range, 0 to {RM.RANGE_MAX_M} m")
        reset.clicked.connect(lambda: RM.set_settings(min_m=0.0, max_m=RM.RANGE_MAX_M))
        row.addWidget(reset)
        lay.addLayout(row)
        note = QtWidgets.QLabel("Shared with the range map on the Visualize tab. "
                                "Narrow it to where the target moves for a cleaner Auto Fit.")
        note.setObjectName("note")
        note.setWordWrap(True)
        lay.addWidget(note)
        apply = lambda _=None: RM.set_settings(min_m=self._dist_from.value(), max_m=self._dist_to.value())
        self._dist_from.valueChanged.connect(apply)
        self._dist_to.valueChanged.connect(apply)
        self._range_opts = box
        self._sync_range_inputs()
        box.setVisible(False)
        return box

    def _dist_spin(self, lo, hi, init):
        box = QtWidgets.QDoubleSpinBox()
        box.setDecimals(2)
        box.setSingleStep(0.1)
        box.setRange(lo, hi)
        box.setValue(init)
        box.setKeyboardTracking(False)
        box.setMaximumWidth(76)
        return box

    def _build_right(self):
        panel = QtWidgets.QWidget()
        panel.setStyleSheet(f"background: {self._c['bg']};")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(6)

        self._spec_heading = QtWidgets.QLabel("Live Spectrogram")
        self._spec_heading.setObjectName("plot_heading")
        layout.addWidget(self._spec_heading)

        self._plot_widget = pg.GraphicsLayoutWidget()
        self._plot_widget.setBackground('#00008F')
        # Fifty columns across a wide plot; without this they draw as hard
        # blocks, where matplotlib (and so the reference drawing) interpolates.
        self._plot_widget.setRenderHint(
            QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
        self._plot_widget.setRenderHint(
            QtGui.QPainter.RenderHint.Antialiasing, True)
        self._plot = self._plot_widget.addPlot()
        self._plot.setLabel('left', 'Velocity', units='m/s')
        self._plot.setLabel('bottom', 'Time', units='s')
        self._plot.hideButtons()
        self._plot.setMouseEnabled(x=False, y=False)
        for axis in ('left', 'bottom'):
            ax = self._plot.getAxis(axis)
            ax.setTextPen(pg.mkPen('w'))
            ax.setPen(pg.mkPen('w'))
            ax.enableAutoSIPrefix(False)

        self._img = pg.ImageItem()
        self._plot.addItem(self._img)
        self._img.setColorMap(make_jet_colormap())
        self._img.setLevels([DB_MIN, DB_MAX])

        self._trace_curve = self._plot.plot(
            [], [], pen=pg.mkPen(TRACE_COLOR, width=2))
        self._auto_dots = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), brush=pg.mkBrush(TRACE_COLOR))
        self._plot.addItem(self._auto_dots)
        self._peak_marks = pg.ScatterPlotItem(size=14, pen=pg.mkPen('w', width=1.5))
        self._plot.addItem(self._peak_marks)

        self._plot_widget.viewport().installEventFilter(self)
        layout.addWidget(self._plot_widget, 3)

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(12)
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(6)
        fit_heading = QtWidgets.QLabel("Fitted Curve")
        fit_heading.setObjectName("plot_heading")
        col.addWidget(fit_heading)

        self._results_widget = pg.GraphicsLayoutWidget()
        self._results_widget.setBackground(self._c['panel'])
        self._results_plot = self._results_widget.addPlot()
        self._results_plot.setLabel('left', 'Velocity', units='m/s')
        self._results_plot.setLabel('bottom', 'Time', units='s')
        self._results_plot.hideButtons()
        self._results_plot.showGrid(x=True, y=True, alpha=0.2)
        for axis in ('left', 'bottom'):
            ax = self._results_plot.getAxis(axis)
            ax.setTextPen(pg.mkPen(self._c['text']))
            ax.setPen(pg.mkPen(self._c['border']))
            ax.enableAutoSIPrefix(False)

        self._fit_points = pg.ScatterPlotItem(
            size=7, pen=pg.mkPen(None), brush=pg.mkBrush(TRACE_COLOR))
        self._results_plot.addItem(self._fit_points)
        self._fit_line = self._results_plot.plot(
            [], [], pen=pg.mkPen(self._c['accent'], width=2.5))
        self._fit_peaks = pg.ScatterPlotItem(size=14, pen=pg.mkPen(self._c['panel'], width=1.5))
        self._results_plot.addItem(self._fit_peaks)
        col.addWidget(self._results_widget, 1)
        row.addLayout(col, 4)

        # The result, laid out for students: rebuilt on every fit
        self._results_area = QtWidgets.QScrollArea()
        self._results_area.setObjectName("cf_results")
        self._results_area.setWidgetResizable(True)
        self._results_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._results_area.setMinimumWidth(360)
        row.addWidget(self._results_area, 5)
        layout.addLayout(row, 3)
        self._show_idle(IDLE_TEXT)
        return panel

    def _lbl(self, text):
        w = QtWidgets.QLabel(text)
        w.setObjectName("field_label")
        return w

    def _divider(self):
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setStyleSheet(f"color: {self._c['divider']}; margin: 2px 0;")
        return line

    # ── freeze and resume ────────────────────────────────────────────────────

    def _freeze(self):
        self._frozen = True
        self._reference_frozen = False
        if self._kind == "range":
            # The live range map is already range_map_live.py's own picture;
            # rows from the nearest bin, for snapping and Auto Fit.
            if self._range_hist is not None:
                self._frozen_spec = np.ascontiguousarray(np.asarray(self._range_hist).T)
            self._spec_heading.setText(f"Frozen Range Map (MTI {'on' if RM.mode() == 'mti' else 'off'})")
            self._sync_buttons()
            self._set_status("Frozen. Start Trace and drag across the curve, or press Auto Fit.")
            return
        if get_method() == METHOD_INFINEON and self._raw_frames:
            # About a second over a full window, on the GUI thread. Say so
            # first, and take the button out of reach, rather than look hung.
            self._freeze_btn.setEnabled(False)
            self._set_status("Drawing the frozen window the reference way...")
            QtWidgets.QApplication.processEvents()
            self._reference_frozen = self._draw_reference_frozen()
        if not self._reference_frozen:
            self._frozen_spec = self._display_array()
            self._refresh_image()
        self._spec_heading.setText(
            "Frozen Spectrogram (reference script)" if self._reference_frozen
            else "Frozen Spectrogram (app view)")
        self._sync_buttons()
        self._set_status("Frozen. Start Trace and drag across the curve, or press Auto Fit.")

    def _draw_reference_frozen(self) -> bool:
        """
        Redraw the frozen window the way the reference script draws it.

        Far too slow to run live, about 0.6 s over a full window, but on freeze
        it costs nothing and the picture being traced is then the same one the
        model learns from. Rows are flipped so his orientation, the most
        negative velocity on top, lands the same way on this plot, and the
        spectrogram kept for snapping is flipped with it so a traced point and
        the bin it snaps to agree with what is on screen.
        """
        try:
            frames = np.stack([np.asarray(f) for f in self._raw_frames])
            if frames.ndim == 3:
                frames = frames[:, np.newaxis]
            spec = reference_spectrogram(frames)           # (n_frame, bins)
            missing = self._width - spec.shape[0]
            if missing > 0:
                spec = np.vstack([np.full((missing, spec.shape[1]),
                                          float(spec.min())), spec])
            elif missing < 0:
                spec = spec[-self._width:]
            rgb = reference_rgb(spec)[::-1]
            self._img.setImage(np.ascontiguousarray(rgb.transpose(1, 0, 2)))
            self._frozen_spec = np.ascontiguousarray(spec.T[::-1])
            return True
        except Exception as e:
            self._set_status(f"Reference drawing unavailable: {e}", error=True)
            return False

    def _resume(self):
        self._frozen = False
        self._frozen_spec = None
        self._reference_frozen = False
        if self._trace_btn.isChecked():
            self._trace_btn.setChecked(False)
        if self._kind == "doppler":
            self._img.setLevels([DB_MIN, DB_MAX])
        self._refresh_image()
        self._spec_heading.setText(self._live_heading())
        self._sync_buttons()
        self._set_status("Live.")

    def is_frozen(self) -> bool:
        return self._frozen

    # ── the axis of the current picture ──────────────────────────────────────

    def _axis(self):
        """(quantity, y range shown, scale per row, row of zero, y label, unit)."""
        if self._kind == "range":
            s = RM.get_settings()
            return "range", (s["min"], s["max"]), self._range_step(), 0, "Distance", "m"
        return ("velocity", (-self._max_vel, self._max_vel), self._vel_scale,
                self._freq_bins / 2, "Velocity", "m/s")

    def _time_step(self):
        return RM.FRAME_TIME_S if self._kind == "range" else self._time_scale

    # ── tracing ──────────────────────────────────────────────────────────────

    def _on_trace_toggled(self, on: bool):
        if on and not self._frozen:
            # Tracing a moving picture would fit a curve that has already gone.
            self._freeze()
        self._tracing = bool(on)
        self._drawing = False
        self._trace_btn.setText("✎  Stop Trace" if on else "✎  Start Trace")
        self._plot_widget.viewport().setCursor(
            QtCore.Qt.CursorShape.CrossCursor if on
            else QtCore.Qt.CursorShape.ArrowCursor)
        if on:
            self._set_status("Drag across the curve to trace it.")

    def eventFilter(self, obj, event):
        if self._tracing and obj is self._plot_widget.viewport():
            kind = event.type()
            if (kind == QtCore.QEvent.Type.MouseButtonPress
                    and event.button() == QtCore.Qt.MouseButton.LeftButton):
                self._drawing = True
                self._add_point(event.position())
                return True
            if kind == QtCore.QEvent.Type.MouseMove and self._drawing:
                self._add_point(event.position())
                return True
            if kind == QtCore.QEvent.Type.MouseButtonRelease and self._drawing:
                self._add_point(event.position())
                self._drawing = False
                return True
        return super().eventFilter(obj, event)

    def _add_point(self, view_pos):
        if self._auto:
            # Drawing by hand replaces what Auto Fit found
            self._points = []
            self._auto = False
            self._set_peaks(None)
            self._auto_dots.setData([], [])
        scene_pos = self._plot_widget.mapToScene(view_pos.toPoint())
        p = self._plot.vb.mapSceneToView(scene_pos)
        _, (lo, hi), *_ = self._axis()
        t = min(float(DISPLAY_SECONDS), max(0.0, float(p.x())))
        v = min(hi, max(lo, float(p.y())))
        self._points.append((t, v))
        self._trace_curve.setData([q[0] for q in self._points],
                                  [q[1] for q in self._points])
        self._sync_buttons()

    def _clear_trace(self, quiet=False):
        self._points = []
        self._auto = False
        self._trace_curve.setData([], [])
        self._auto_dots.setData([], [])
        self._fit_points.setData([], [])
        self._fit_line.setData([], [])
        self._set_peaks(None)
        self._show_idle(IDLE_TEXT)
        self._sync_buttons()
        if not quiet:
            self._set_status("Trace cleared.")

    def trace_points(self):
        return list(self._points)

    def _set_peaks(self, peaks):
        self._peaks = peaks if peaks and peaks["all"] else None
        spots = []
        for t, v, sign in (self._peaks["all"] if self._peaks else []):
            spots.append(dict(pos=(t, v), symbol='t1' if sign > 0 else 't',
                              brush=pg.mkBrush(PEAK_UP if sign > 0 else PEAK_DOWN)))
        self._peak_marks.setData(spots)
        self._fit_peaks.setData(spots)

    # ── fitting ──────────────────────────────────────────────────────────────

    def _model_key(self):
        btn = self._model_group.checkedButton()
        return btn.property("model_key") if btn else MODEL_SINUSOID

    def _on_model_changed(self, button, checked):
        if not checked:
            return
        self._degree_row.setVisible(
            button.property("model_key") == MODEL_POLYNOMIAL)

    def _on_fit(self) -> bool:
        quantity, _, scale, row0, label, unit = self._axis()
        points = list(self._points)
        if not self._auto and self._snap_check.isChecked() and self._frozen_spec is not None:
            points = snap_to_peak(points, self._frozen_spec, self._time_step(), scale,
                                  row_offset=row0)
        model = self._model_key()
        # The positive and negative peaks of the trace give a starting point
        # of their own; the fit keeps whichever start ends lower.
        peaks = None
        extra = []
        if model in (MODEL_SINUSOID, MODEL_DAMPED):
            t, v = clean_trace(points)
            if t.size >= 3 and np.all(np.isfinite(v)):
                peaks = find_peaks(t, v)
                start = peak_start(model, peaks)
                if start is not None:
                    extra.append(start)
        result = fit_curve(points, model, degree=self._degree_spin.value(),
                           quantity=quantity, extra_starts=extra)
        if not result.ok:
            self._fit_points.setData([], [])
            self._fit_line.setData([], [])
            self._set_peaks(None)
            self._show_idle(result.message, error=True)
            self._set_status(result.message, error=True)
            return False

        self._results_plot.setLabel('left', label, units=unit)
        self._fit_points.setData(result.t, result.v)
        smooth_t = np.linspace(float(result.t[0]), float(result.t[-1]), 400)
        self._fit_line.setData(smooth_t, result.predict(smooth_t))
        self._set_peaks(peaks)
        self._results_plot.enableAutoRange()
        self._show_results(result, self._peaks)
        self._set_status(
            f"Fitted {len(result.t)} points, R² = {result.r_squared:.3f}")
        return True

    def _auto_fit(self):
        """Freeze, find the curve in the picture, then fit it from its peaks."""
        if self._trace_btn.isChecked():
            self._trace_btn.setChecked(False)
        if not self._frozen:
            self._freeze()
        if self._frozen_spec is None:
            self._set_status("Nothing to fit yet: connect the radar and let the picture fill first.", error=True)
            return
        quantity, (lo, hi), scale, row0, *_ = self._axis()
        pts = auto_trace(self._frozen_spec, self._time_step(), scale, row_offset=row0,
                         y_range=(lo, hi), dead_zone=STILL_BAND if self._kind == "doppler" else 0.0)
        if len(pts) < MIN_TRACE_POINTS:
            self._clear_trace(quiet=True)
            msg = (f"Auto Fit found a clear signal in only {len(pts)} moment{'' if len(pts) == 1 else 's'}; "
                   f"it needs {MIN_TRACE_POINTS}. "
                   + ("Try turning MTI on, or narrow the distance shown to where the target is."
                      if self._kind == "range" else
                      "Move the target in front of the radar, then freeze again."))
            self._show_idle(msg, error=True)
            self._set_status(msg, error=True)
            return
        self._points = pts
        self._auto = True
        self._trace_curve.setData([], [])
        self._auto_dots.setData([p[0] for p in pts], [p[1] for p in pts])
        self._sync_buttons()
        if self._on_fit():
            self._set_status(f"Auto Fit: {len(pts)} points found, R² = "
                             f"{self._last_r2:.3f}")

    # ── the result, for students ─────────────────────────────────────────────

    def _results_body(self):
        body = QtWidgets.QWidget()
        body.setObjectName("cf_results_body")
        lay = QtWidgets.QVBoxLayout(body)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(12)
        self._results_area.setWidget(body)       # the old body is deleted
        return lay

    def _show_idle(self, text, error=False):
        lay = self._results_body()
        lbl = QtWidgets.QLabel(text)
        lbl.setObjectName("cf_idle_err" if error else "cf_idle")
        lbl.setWordWrap(True)
        lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lay.addStretch()
        lay.addWidget(lbl)
        lay.addStretch()

    def _label(self, text, name, rich=True):
        lbl = QtWidgets.QLabel(text)
        lbl.setObjectName(name)
        lbl.setWordWrap(True)
        if rich:
            lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
        return lbl

    def _card(self, item, predicted=False):
        card = QtWidgets.QFrame()
        card.setObjectName("cf_card_pred" if predicted else "cf_card")
        lay = QtWidgets.QVBoxLayout(card)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(4)
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(10)
        if item.get("sym"):
            top.addWidget(self._label(item["sym"], "cf_sym_small" if predicted else "cf_sym"))
        top.addWidget(self._label(item["name"].upper(), "cf_pname", rich=False), 1)
        lay.addLayout(top)
        lay.addWidget(self._label(
            f"{item['value']} <span style='font-size:14px; font-weight:normal; color:{self._c['subtext']}'>{item['unit']}</span>",
            "cf_val"))
        lay.addWidget(self._label(item["meaning"], "cf_meaning"))
        return card

    def _grid(self, items, predicted=False):
        w = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(w)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(10)
        for i, item in enumerate(items):
            grid.addWidget(self._card(item, predicted), i // 2, i % 2)
        for c in (0, 1):
            grid.setColumnStretch(c, 1)
        return w

    def _show_results(self, r, peaks):
        self._last_r2 = r.r_squared
        x = explain_fit(r)
        lay = self._results_body()

        head = QtWidgets.QHBoxLayout()
        head.addWidget(self._label(f"Your fitted curve: {MODEL_LABELS.get(r.model, r.model)}", "cf_title", rich=False), 1)
        badge = self._label(f"{x['quality']}   R² = {r.r_squared:.3f}", f"cf_badge_{x['level']}", rich=False)
        badge.setWordWrap(False)
        badge.setToolTip("R² (R squared) measures how well the curve matches your points.")
        head.addWidget(badge, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        lay.addLayout(head)

        box = QtWidgets.QFrame()
        box.setObjectName("cf_eq_box")
        blay = QtWidgets.QVBoxLayout(box)
        blay.setContentsMargins(16, 12, 16, 12)
        blay.setSpacing(4)
        blay.addWidget(self._label("THE MODEL", "cf_kicker", rich=False))
        blay.addWidget(self._label(x["general"], "cf_eq"))
        blay.addWidget(self._label("WITH YOUR NUMBERS", "cf_kicker", rich=False))
        blay.addWidget(self._label(x["numbers"], "cf_eq_nums"))
        lay.addWidget(box)

        lay.addWidget(self._label(
            f"<i>t</i> is the time in seconds and {x['y']} is the {x['what']} at that time, "
            f"in {x['unit']}. R² is 1 for a perfect match and 0 when the curve is no better "
            f"than a flat line.", "cf_note"))
        lay.addWidget(self._label("What each letter means", "cf_section", rich=False))
        lay.addWidget(self._grid(x["params"]))
        if x["results"]:
            lay.addWidget(self._label("What the fit tells you", "cf_section", rich=False))
            lay.addWidget(self._grid(x["results"], predicted=True))
        text = explain_peaks(peaks, x["unit"])
        if text:
            lay.addWidget(self._label(text, "cf_peaks"))
        lay.addStretch()

    # ── state ────────────────────────────────────────────────────────────────

    def _sync_buttons(self):
        self._freeze_btn.setEnabled(not self._frozen)
        self._resume_btn.setEnabled(self._frozen)
        self._clear_btn.setEnabled(bool(self._points))
        self._fit_btn.setEnabled(len(self._points) >= MIN_TRACE_POINTS)

    def _set_status(self, msg: str, error: bool = False):
        self._status.setObjectName("status_err" if error else "status_ok")
        self._status.style().unpolish(self._status)
        self._status.style().polish(self._status)
        self._status.setText(msg)

    def showEvent(self, event):
        super().showEvent(event)
        self._sync_range_inputs()

    def shutdown(self):
        RM.remove_listener(self._on_range_settings)
        self._stop_range()
