"""
ui/range_map_view.py

The live range map, drawn by range_map_live.py itself.

As with the reference spectrogram (ui/reference_view.py), the script's own
LiveRangeProcessor turns each radar frame into a range profile, off the GUI
thread, and its own LiveRangePlot draws the rolling picture inside a Qt layout
on a 5 Hz timer.

The distance shown is applied by handing the plot only the bins inside it:
the plot is built with that part of the range axis and drawn with those
columns of the history, so its colour limits come from what is on screen.
Changing the distance rebuilds the plot (a new figure) and redraws the last
history; changing MTI or the history length starts a new processor.
"""

import numpy as np
from PyQt6 import QtCore, QtWidgets

from core import range_map as RM
from core.reference_image import current_jet_vmin
from ui import app_colors


class RangeWorker(QtCore.QObject):
    """LiveRangeProcessor per frame, off the GUI thread."""

    history_ready = QtCore.pyqtSignal(np.ndarray, int, int)    # history, n_filled, count
    failed = QtCore.pyqtSignal(str)

    def __init__(self, history_length: int, mode: str):
        super().__init__()
        self._history_length = history_length
        self._mode = mode
        self._proc = None
        self._count = 0

    @QtCore.pyqtSlot(np.ndarray)
    def on_raw_frame(self, frame: np.ndarray):
        try:
            if frame.ndim == 3:
                frame = frame[RM.ANTENNA]
            if self._proc is None:
                _, live = RM.import_range_map()
                n_chirp, n_sample = frame.shape
                self._proc = live.LiveRangeProcessor(
                    n_sample=n_sample, n_chirp=n_chirp,
                    mode=self._mode, history_length=self._history_length)
            history, n_filled = self._proc.process_frame(np.asarray(frame))
            self._count += 1
            self.history_ready.emit(history.copy(), n_filled, self._count)
        except Exception as e:
            self.failed.emit(str(e))


class RangeMapView(QtWidgets.QWidget):
    """Hosts range_map_live.py's own live figure."""

    _REDRAW_MS = 200

    error = QtCore.pyqtSignal(str)

    def __init__(self, history_length: int = 50, parent=None):
        super().__init__(parent)
        self._c = app_colors()
        self._history_length = history_length
        self._mode = RM.mode()
        self._jet_vmin = current_jet_vmin()
        self._plot = None
        self._canvas = None
        self._thread = None
        self._worker = None
        self._pending = None
        self._last = None
        self._ok = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._layout = layout
        try:
            RM.import_range_map()
            self._range_m = RM.range_axis()
            self._build_plot()
            self._ok = True
        except Exception as e:
            msg = QtWidgets.QLabel(f"Range map unavailable.\n\n{e}\n\n"
                                   "It needs matplotlib:  pip install matplotlib")
            msg.setWordWrap(True)
            msg.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            msg.setStyleSheet(f"color: {self._c['faint']}; font-size: 12px;")
            layout.addWidget(msg, 1)
            return

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(self._REDRAW_MS)
        self._timer.timeout.connect(self._redraw)
        self._thread = QtCore.QThread(self)
        self._worker = RangeWorker(history_length, self._mode)
        self._worker.moveToThread(self._thread)
        self._worker.history_ready.connect(self._on_history)
        self._worker.failed.connect(self.error)
        self._thread.start()
        self._timer.start()
        RM.on_settings_changed(self._on_settings)

    # ── the plot ─────────────────────────────────────────────────────────────

    def _bins(self):
        s = RM.get_settings()
        return RM.shown_bins(self._range_m, s["min"], s["max"])

    def _build_plot(self):
        _, live = RM.import_range_map()
        lo, hi = self._bins()
        title = "Range map (live, moving)" if self._mode == "mti" else "Range map (live, raw)"
        # LiveRangePlot turns pyplot's interactive mode on and opens its figure
        # in a window of its own; that window is hidden and the canvas adopted
        # here, as the reference view does.
        self._plot = live.LiveRangePlot(
            history_length=self._history_length,
            range_m=self._range_m[lo:hi + 1],
            frame_repetition_time_s=RM.FRAME_TIME_S,
            jet_vmin=self._jet_vmin,
            title=title,
        )
        self._canvas = self._plot.fig.canvas
        win = getattr(self._canvas.manager, "window", None)
        if win is not None:
            win.hide()
        self._canvas.setParent(self)
        self._layout.addWidget(self._canvas, 1)
        self._lo_hi = (lo, hi)

    def _drop_plot(self):
        if self._plot is None:
            return
        self._layout.removeWidget(self._canvas)
        try:
            self._plot.close()
        except Exception:
            pass
        self._canvas.deleteLater()
        self._plot = None
        self._canvas = None

    def _on_settings(self, settings, changed):
        # A new MTI mode needs a new processor; the Visualize tab rebuilds the
        # whole view for that. A new distance only needs a new plot.
        if self._ok and changed["distance"]:
            self._drop_plot()
            self._build_plot()
            if self._last is not None:
                self._pending = self._last
                self._redraw()

    # ── frame intake ─────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ok

    @property
    def history_length(self) -> int:
        return self._history_length

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def jet_vmin(self) -> float:
        return self._jet_vmin

    @property
    def redraws_per_second(self) -> float:
        return 1000.0 / self._REDRAW_MS

    @QtCore.pyqtSlot(np.ndarray)
    def on_raw_frame(self, frame: np.ndarray):
        if not self._ok or self._worker is None:
            return
        QtCore.QMetaObject.invokeMethod(
            self._worker, "on_raw_frame",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(np.ndarray, np.array(frame, copy=True)),
        )

    def _on_history(self, history, n_filled, count):
        self._pending = (history, n_filled, count)

    def _redraw(self):
        if self._pending is None or self._plot is None:
            return
        history, n_filled, count = self._pending
        self._pending = None
        self._last = (history, n_filled, count)
        lo, hi = self._lo_hi
        try:
            self._plot.draw(history[:, lo:hi + 1], n_filled, count)
        except Exception as e:
            self.error.emit(str(e))

    # ── teardown ─────────────────────────────────────────────────────────────

    def shutdown(self):
        RM.remove_listener(self._on_settings)
        if getattr(self, "_timer", None) is not None:
            self._timer.stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)
            self._thread = None
        self._worker = None
        self._drop_plot()

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)
