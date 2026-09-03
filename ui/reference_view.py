"""
ui/reference_view.py

Embeds the reference live Doppler spectrogram so it can be compared side by
side with the app's own Infineon SDK view.

core/doppler_spectrogram_live.py is a byte-for-byte copy of the script it came
from and is never edited. This module only feeds it radar frames and hosts its
matplotlib canvas inside the Qt layout, so what appears on screen is drawn by
that script's own LiveDopplerProcessor and LiveSpectrogramPlot.

Two things run off the GUI thread or on a timer, neither of which changes what
the reference code produces:
  * process_frame() runs on a worker thread (~10 ms per frame)
  * the canvas is redrawn on a timer (~30 ms per redraw), so a slow repaint
    cannot stall the radar loop. Every frame still reaches its history buffer.
"""

import numpy as np
from PyQt6 import QtCore, QtWidgets

from ui import app_colors

# Frames per second the radar delivers; used to size the reference history
# buffer so both panels cover the same number of seconds.
_FPS = 10


def _import_reference():
    """
    Import the reference script, with matplotlib pointed at Qt first.

    Done lazily so the app starts without matplotlib installed and only pays
    the import cost when the comparison view is actually switched on.
    """
    import matplotlib
    matplotlib.use("QtAgg", force=False)
    from core import doppler_spectrogram_live as ref
    return ref


class _RefWorker(QtCore.QObject):
    """Runs the reference processor's per-frame DSP off the GUI thread."""

    history_ready = QtCore.pyqtSignal(np.ndarray, int)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, history_length: int):
        super().__init__()
        self._history_length = history_length
        self._proc = None
        self._count = 0
        self._latest = None
        self._lock = QtCore.QMutex()

    @QtCore.pyqtSlot(np.ndarray)
    def on_raw_frame(self, frame: np.ndarray):
        try:
            if frame.ndim == 3:
                frame = frame[0]           # antenna 0, the reference default
            if self._proc is None:
                ref = _import_reference()
                n_chirp, n_sample = frame.shape
                self._proc = ref.LiveDopplerProcessor(
                    n_sample=n_sample,
                    n_chirp=n_chirp,
                    history_length=self._history_length,
                )
            history, _ = self._proc.process_frame(np.asarray(frame))
            self._count += 1
            self.history_ready.emit(history, self._count)
        except Exception as e:
            self.failed.emit(str(e))


class ReferenceSpectrogramView(QtWidgets.QWidget):
    """Hosts the reference script's own matplotlib figure."""

    _REDRAW_MS = 200      # 5 Hz; a full redraw costs ~30 ms

    error = QtCore.pyqtSignal(str)

    def __init__(self, history_length: int = 100, parent=None):
        super().__init__(parent)
        self._c = app_colors()
        self._history_length = history_length
        self._plot = None
        self._canvas = None
        self._worker = None
        self._thread = None
        self._pending = None
        self._ok = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self._layout = layout

        try:
            ref = _import_reference()
            self._plot = ref.LiveSpectrogramPlot(
                history_length=history_length,
                max_speed_m_s=6.19405905,     # the reference default
                jet_vmin=-20.0,
                orientation=ref.ORIENT_FRAME_X,
            )
            self._canvas = self._plot.fig.canvas
            # pyplot gave the figure its own window; hide it and adopt the
            # canvas into this layout instead.
            win = getattr(self._canvas.manager, "window", None)
            if win is not None:
                win.hide()
            self._canvas.setParent(self)
            layout.addWidget(self._canvas, 1)
            self._ok = True
        except Exception as e:
            msg = QtWidgets.QLabel(
                "Reference view unavailable.\n\n"
                f"{e}\n\n"
                "It needs matplotlib:  pip install matplotlib"
            )
            msg.setWordWrap(True)
            msg.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            msg.setStyleSheet(f"color: {self._c['faint']}; font-size: 12px;")
            layout.addWidget(msg, 1)
            return

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(self._REDRAW_MS)
        self._timer.timeout.connect(self._redraw)

        self._thread = QtCore.QThread(self)
        self._worker = _RefWorker(history_length)
        self._worker.moveToThread(self._thread)
        self._worker.history_ready.connect(self._on_history)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()
        self._timer.start()

    # ── frame intake ─────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ok

    @QtCore.pyqtSlot(np.ndarray)
    def on_raw_frame(self, frame: np.ndarray):
        """Hand a radar frame to the reference processor (queued to its thread)."""
        if not self._ok or self._worker is None:
            return
        QtCore.QMetaObject.invokeMethod(
            self._worker, "on_raw_frame",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(np.ndarray, np.array(frame, copy=True)),
        )

    def _on_history(self, history: np.ndarray, count: int):
        # Keep only the newest; the timer decides when to actually repaint.
        self._pending = (history, count)

    def _on_failed(self, msg: str):
        self.error.emit(msg)

    def _redraw(self):
        if self._pending is None or self._plot is None:
            return
        history, count = self._pending
        self._pending = None
        try:
            self._plot.draw(history, frame_end=count)
        except Exception as e:
            self.error.emit(str(e))

    # ── teardown ─────────────────────────────────────────────────────────────

    def shutdown(self):
        if getattr(self, "_timer", None) is not None:
            self._timer.stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)
            self._thread = None
        self._worker = None
        if self._plot is not None:
            try:
                self._plot.close()
            except Exception:
                pass
            self._plot = None

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)
