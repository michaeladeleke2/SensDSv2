"""
ui/curve_fit_tab.py

Freeze the live spectrogram, trace the ridge a moving target leaves behind, and
fit a function to it.

A Newton's cradle swinging toward and away from the radar traces a sinusoid, so
the fitted frequency predicts a pendulum length the student can check against
the real one with a ruler.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from core.curve_fit import (
    MODEL_DAMPED, MODEL_LINEAR, MODEL_POLYNOMIAL, MODEL_SINUSOID,
    fit_curve, snap_to_peak,
)
from core.processing import (
    FRAME_TIME_S, get_method, method_cols_per_frame, method_freq_bins,
    method_max_velocity,
)
from ui import app_colors, _scrollable_left
from ui.spectrogram_widget import (
    DB_MAX, DB_MIN, DISPLAY_SECONDS, make_jet_colormap,
)

TRACE_COLOR = "#ff00ff"
MIN_TRACE_POINTS = 10


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
    """Live spectrogram, freehand trace, fitted model."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._c = app_colors()
        self.setObjectName("curvefit_root")
        self.setStyleSheet(_curve_fit_style(self._c))

        self._method = get_method()
        self._apply_method_dims()
        self._col = 0
        self._frozen = False
        self._frozen_spec = None
        self._tracing = False
        self._drawing = False
        self._points = []

        self._setup_ui()
        self._rescale_axes()
        self._refresh_image()
        self._sync_buttons()

    # ── live buffer ──────────────────────────────────────────────────────────

    def _apply_method_dims(self):
        self._freq_bins = method_freq_bins(self._method)
        cols_per_frame = method_cols_per_frame(self._method)
        self._max_vel = method_max_velocity(self._method)
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
        self._refresh_image()

    def _display_array(self):
        """The buffer in display order, oldest column first."""
        return np.roll(self._buffer, -self._col, axis=1)

    def _refresh_image(self):
        self._img.setImage(self._display_array().T, autoLevels=False)

    def _rescale_axes(self):
        self._img.setTransform(
            QtGui.QTransform().scale(self._time_scale, self._vel_scale)
                              .translate(0, -self._freq_bins / 2)
        )
        self._plot.setXRange(0, DISPLAY_SECONDS, padding=0)
        self._plot.setYRange(-self._max_vel, self._max_vel, padding=0)

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
        self._clear_btn.clicked.connect(self._clear_trace)
        layout.addWidget(self._clear_btn)

        self._snap_check = QtWidgets.QCheckBox("Snap to strongest signal")
        self._snap_check.setChecked(True)
        self._snap_check.setToolTip(
            "Move each traced point onto the brightest bin near it before\n"
            "fitting, so the fit follows the signal rather than how steady\n"
            "your hand was."
        )
        layout.addWidget(self._snap_check)

        layout.addWidget(self._divider())

        layout.addWidget(self._lbl("Model"))
        self._model_group = QtWidgets.QButtonGroup(self)
        for key, label in ((MODEL_SINUSOID, "Sinusoid"),
                           (MODEL_DAMPED, "Damped sinusoid"),
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

        self._plot_widget.viewport().installEventFilter(self)
        layout.addWidget(self._plot_widget, 2)

        fit_heading = QtWidgets.QLabel("Fitted Curve")
        fit_heading.setObjectName("plot_heading")
        layout.addWidget(fit_heading)

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
            [], [], pen=pg.mkPen(self._c['accent'], width=2))
        layout.addWidget(self._results_widget, 1)

        self._equation = QtWidgets.QLabel(
            "Freeze the display, trace a curve, then fit a model to it.")
        self._equation.setObjectName("equation")
        self._equation.setWordWrap(True)
        layout.addWidget(self._equation)

        self._physics = QtWidgets.QLabel("")
        self._physics.setObjectName("physics")
        self._physics.setWordWrap(True)
        self._physics.setVisible(False)
        layout.addWidget(self._physics)

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
        self._frozen_spec = self._display_array()
        self._spec_heading.setText("Frozen Spectrogram")
        self._sync_buttons()
        self._set_status("Frozen. Start Trace, then drag across the curve.")

    def _resume(self):
        self._frozen = False
        self._frozen_spec = None
        if self._trace_btn.isChecked():
            self._trace_btn.setChecked(False)
        self._spec_heading.setText("Live Spectrogram")
        self._sync_buttons()
        self._set_status("Live.")

    def is_frozen(self) -> bool:
        return self._frozen

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
        scene_pos = self._plot_widget.mapToScene(view_pos.toPoint())
        p = self._plot.vb.mapSceneToView(scene_pos)
        t = min(float(DISPLAY_SECONDS), max(0.0, float(p.x())))
        v = min(self._max_vel, max(-self._max_vel, float(p.y())))
        self._points.append((t, v))
        self._trace_curve.setData([q[0] for q in self._points],
                                  [q[1] for q in self._points])
        self._sync_buttons()

    def _clear_trace(self):
        self._points = []
        self._trace_curve.setData([], [])
        self._fit_points.setData([], [])
        self._fit_line.setData([], [])
        self._equation.setText(
            "Freeze the display, trace a curve, then fit a model to it.")
        self._physics.setVisible(False)
        self._sync_buttons()
        self._set_status("Trace cleared.")

    def trace_points(self):
        return list(self._points)

    # ── fitting ──────────────────────────────────────────────────────────────

    def _model_key(self):
        btn = self._model_group.checkedButton()
        return btn.property("model_key") if btn else MODEL_SINUSOID

    def _on_model_changed(self, button, checked):
        if not checked:
            return
        self._degree_row.setVisible(
            button.property("model_key") == MODEL_POLYNOMIAL)

    def _on_fit(self):
        points = list(self._points)
        if self._snap_check.isChecked() and self._frozen_spec is not None:
            points = snap_to_peak(points, self._frozen_spec,
                                  self._time_scale, self._vel_scale)
        result = fit_curve(points, self._model_key(),
                           degree=self._degree_spin.value())
        if not result.ok:
            self._fit_points.setData([], [])
            self._fit_line.setData([], [])
            self._physics.setVisible(False)
            self._equation.setText(result.message)
            self._set_status(result.message, error=True)
            return

        self._fit_points.setData(result.t, result.v)
        smooth_t = np.linspace(float(result.t[0]), float(result.t[-1]), 400)
        self._fit_line.setData(smooth_t, result.predict(smooth_t))
        self._results_plot.enableAutoRange()

        self._equation.setText(
            f"{result.equation}\n{result.param_text}\n"
            f"R² = {result.r_squared:.4f}"
        )
        self._physics.setText(result.physics)
        self._physics.setVisible(bool(result.physics))
        self._set_status(
            f"Fitted {len(result.t)} points, R² = {result.r_squared:.3f}")

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
