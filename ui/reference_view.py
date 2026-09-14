"""
ui/reference_view.py

The reference Doppler spectrogram, drawn by the reference script itself.

core/doppler_spectrogram_live.py is a byte-for-byte copy of the script it came
from and is never edited. This module only feeds it radar data and hosts its
matplotlib figures inside Qt layouts, so what appears on screen is drawn by
that script's own code:

  ReferenceSpectrogramView   the live, scrolling plot, via LiveDopplerProcessor
                             and LiveSpectrogramPlot. The Visualize tab's main
                             view for the Infineon SDK method.
  ReferenceRecordedView      one finished capture, via compute_recorded and
                             plot_recorded_spectrogram. The Collect tab's
                             "Last Captured Gesture" for the Infineon SDK method.

Heavy work runs off the GUI thread and the live canvas is redrawn on a timer.
Neither changes what the reference code produces.
"""

import numpy as np
from PyQt6 import QtCore, QtWidgets

from ui import app_colors

# Frames per second the radar delivers; used to size the reference history
# buffer so it covers the seconds the time window asks for.
_FPS = 10

# The reference script's own defaults, passed to it unchanged. Named once here
# so the live view, the capture view and the readout cannot disagree.
REF_MAX_SPEED_M_S = 6.19405905
REF_JET_VMIN = -20.0
REF_ANTENNA = 0


def _import_reference():
    """
    Import the reference script, with matplotlib pointed at Qt first.

    Done lazily so the app starts without matplotlib installed and only pays
    the import cost when a reference view is actually built.
    """
    import matplotlib
    matplotlib.use("QtAgg", force=False)
    from core import doppler_spectrogram_live as ref
    return ref


# ── live ──────────────────────────────────────────────────────────────────────

class _RefWorker(QtCore.QObject):
    """Runs the reference processor's per-frame DSP off the GUI thread."""

    history_ready = QtCore.pyqtSignal(np.ndarray, int)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, history_length: int):
        super().__init__()
        self._history_length = history_length
        self._proc = None
        self._count = 0

    @QtCore.pyqtSlot(np.ndarray)
    def on_raw_frame(self, frame: np.ndarray):
        try:
            if frame.ndim == 3:
                frame = frame[REF_ANTENNA]
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
    """Hosts the reference script's own live matplotlib figure."""

    # 5 Hz. A redraw costs 50-85 ms on a fast Mac at main-view sizes, and the
    # redraw runs on the GUI thread, so drawing at the full 10 fps would leave
    # a Surface little time for anything else. Every frame still reaches the
    # history buffer; only how often the picture is repainted changes.
    _REDRAW_MS = 200

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
                max_speed_m_s=REF_MAX_SPEED_M_S,
                jet_vmin=REF_JET_VMIN,
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

    @property
    def history_length(self) -> int:
        """Frames of history the plot spans; fixed when the view is built."""
        return self._history_length

    @property
    def redraws_per_second(self) -> float:
        return 1000.0 / self._REDRAW_MS

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


# ── one finished capture ─────────────────────────────────────────────────────

class _CaptureSignals(QtCore.QObject):
    """Carries a finished computation from the thread pool to the GUI thread."""

    ready = QtCore.pyqtSignal(object, int)
    failed = QtCore.pyqtSignal(str)


class _CaptureJob(QtCore.QRunnable):
    """compute_recorded() for one capture, on a pool thread."""

    def __init__(self, ref, cube, token, signals):
        super().__init__()
        self._ref = ref
        self._cube = cube
        self._token = token
        self._signals = signals

    def run(self):
        try:
            spectrogram, _ = self._ref.compute_recorded(
                self._cube, antenna=REF_ANTENNA)
            outcome = ("ready", spectrogram)
        except Exception as e:
            outcome = ("failed", str(e))
        try:
            if outcome[0] == "ready":
                self._signals.ready.emit(outcome[1], self._token)
            else:
                self._signals.failed.emit(outcome[1])
        except RuntimeError:
            # The view was destroyed while this was computing, as happens when
            # the app closes mid-capture. An exception escaping run() would
            # abort the process, so there is nothing to do but stop quietly.
            pass


class ReferenceRecordedView(QtWidgets.QWidget):
    """
    One finished capture, drawn by the reference script's offline path.

    compute_recorded() turns the raw cube into the spectrogram on a pool thread,
    since it takes a quarter of a second or more per capture. Then
    plot_recorded_spectrogram() draws it on the GUI thread, where Qt requires
    figures to be made. Each new capture replaces the previous figure, which is
    closed so figures do not pile up over a long collection session.

    plot_recorded_spectrogram() lays its axes over the whole figure, so the
    image fills the panel with no tick labels or title visible. That is the
    reference script's own layout and is left exactly as it is.
    """

    error = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._c = app_colors()
        self._ref = None
        self._fig = None
        self._canvas = None
        self._token = 0

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._layout = layout

        self._placeholder = QtWidgets.QLabel(
            "Captured gestures appear here, drawn by the reference script."
        )
        self._placeholder.setWordWrap(True)
        self._placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(
            f"color: {self._c['faint']}; font-size: 12px;")
        layout.addWidget(self._placeholder, 1)

        # These signals live on the GUI thread, so a pool thread emitting them
        # is delivered back here as a queued call.
        self._signals = _CaptureSignals(self)
        self._signals.ready.connect(self._on_ready)
        self._signals.failed.connect(self._on_failed)

        try:
            # Imported here, on the GUI thread, so pyplot and its Qt backend are
            # set up before any pool thread touches the module.
            self._ref = _import_reference()
        except Exception as e:
            self._placeholder.setText(
                "Reference view unavailable.\n\n"
                f"{e}\n\n"
                "It needs matplotlib:  pip install matplotlib"
            )

    @property
    def is_ready(self) -> bool:
        return self._ref is not None

    @property
    def figure(self):
        """The figure currently shown, or None before the first capture."""
        return self._fig

    def show_capture(self, cube: np.ndarray):
        """Draw a raw capture shaped (n_frame, n_ant, n_chirp, n_sample)."""
        if self._ref is None:
            return
        self._token += 1
        job = _CaptureJob(self._ref, np.array(cube, copy=True),
                          self._token, self._signals)
        QtCore.QThreadPool.globalInstance().start(job)

    def _on_ready(self, spectrogram, token):
        # Captures can finish out of order when they arrive close together;
        # only the most recent one is worth drawing.
        if token != self._token:
            return
        import matplotlib.pyplot as plt
        # The reference script turns pyplot's interactive mode on for its live
        # plot, and in interactive mode every new figure pops up in a window of
        # its own. Hold it off for this one figure: that changes whether a
        # window flashes up, not what is drawn.
        was_interactive = plt.isinteractive()
        plt.ioff()
        try:
            fig = self._ref.plot_recorded_spectrogram(
                spectrogram,
                max_speed_m_s=REF_MAX_SPEED_M_S,
                jet_vmin=REF_JET_VMIN,
            )
        except Exception as e:
            self.error.emit(str(e))
            return
        finally:
            if was_interactive:
                plt.ion()

        canvas = fig.canvas
        win = getattr(canvas.manager, "window", None)
        if win is not None:
            win.hide()
        self._drop_figure()
        canvas.setParent(self)
        self._placeholder.hide()
        self._layout.addWidget(canvas, 1)
        self._fig, self._canvas = fig, canvas
        canvas.draw_idle()

    def _on_failed(self, msg: str):
        self.error.emit(msg)

    def _drop_figure(self):
        if self._fig is None:
            return
        import matplotlib.pyplot as plt
        self._layout.removeWidget(self._canvas)
        plt.close(self._fig)
        self._canvas.deleteLater()
        self._fig = None
        self._canvas = None

    def shutdown(self):
        self._token += 1          # anything still computing is now stale
        self._drop_figure()
        self._placeholder.show()

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)
