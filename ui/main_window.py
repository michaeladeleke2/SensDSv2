import sys
import os
import threading
import time as _time
import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui
from ui import app_colors
from ui.gamification import GamificationManager, GamificationBar
from ui.spectrogram_widget import SpectrogramWidget
from ui.collect_tab import CollectTab
from ui.train_tab import TrainTab
from ui.test_tab import TestTab
from ui.results_tab import ResultsTab
from ui.vex_aim_tab import VexAimTab
from core.radar import RadarStream
from core.processing import SpectrogramProcessor


HINTS = [
    "Try waving your hand slowly toward the sensor.",
    "A fast swipe left or right creates a diagonal streak.",
    "Pushing your hand forward produces a strong positive velocity burst.",
    "Hold still — notice how the display stays flat at 0 m/s.",
    "Positive velocity = moving toward the radar. Negative = moving away.",
]


def _app_style(c: dict, compact: bool = False) -> str:
    """
    compact=True is used on screens shorter than 800 px (e.g. Surface tablets).
    It reduces font sizes and button padding so the topbar takes only 64 px instead of 96 px,
    freeing ~32 px of content height on small displays.
    """
    btn_pad   = "7px 16px"  if compact else "12px 28px"
    btn_font  = "14px"      if compact else "17px"
    stat_font = "13px"      if compact else "17px"
    time_font = "15px"      if compact else "19px"
    hint_font = "13px"      if compact else "16px"
    err_font  = "13px"      if compact else "16px"
    tab_pad   = "12px 0px"  if compact else "20px 0px"
    tab_font  = "14px"      if compact else "17px"
    tab_minw  = "110px"     if compact else "160px"

    return f"""
    QMainWindow {{ background-color: {c['bg']}; }}

    /* Top bar — always dark blue regardless of theme */
    QWidget#topbar {{
        background-color: #1a3a5c;
    }}
    QLabel#app_title {{
        font-size: 24px;
        font-weight: bold;
        color: #ffffff;
    }}
    QLabel#status_label {{
        font-size: {stat_font};
        color: #aac4e0;
    }}
    QLabel#timer_label {{
        font-size: {time_font};
        font-weight: bold;
        color: #ffffff;
        font-family: monospace;
    }}
    QLabel#hint_label {{
        font-size: {hint_font};
        color: #aac4e0;
        font-style: italic;
    }}
    QPushButton#connect_btn {{
        background-color: #27ae60;
        color: white;
        border: none;
        border-radius: 8px;
        padding: {btn_pad};
        font-size: {btn_font};
        font-weight: bold;
    }}
    QPushButton#connect_btn:hover {{ background-color: #2ecc71; }}
    QPushButton#disconnect_btn {{
        background-color: #c0392b;
        color: white;
        border: none;
        border-radius: 8px;
        padding: {btn_pad};
        font-size: {btn_font};
        font-weight: bold;
    }}
    QPushButton#disconnect_btn:hover {{ background-color: #e74c3c; }}
    QLabel#error_label {{
        font-size: {err_font};
        color: #f1948a;
    }}

    /* Tabs */
    QTabWidget::pane {{
        border: none;
        background: {c['bg']};
    }}
    QTabBar {{
        background: {c['bg']};
    }}
    QTabBar::tab {{
        background: {c['bg']};
        color: {c['subtext']};
        padding: {tab_pad};
        font-size: {tab_font};
        border: none;
        border-bottom: 4px solid transparent;
        min-width: {tab_minw};
    }}
    QTabBar::tab:selected {{
        color: {c['accent']};
        font-weight: bold;
        border-bottom: 4px solid {c['accent']};
        background: {c['bg']};
    }}
    QTabBar::tab:hover {{
        color: {c['accent']};
        background: {c['tab_hover']};
    }}
"""

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class _SpectrogramDisplayWorker(QtCore.QObject):
    """
    Background thread that pulls the latest radar frames from the
    SpectrogramProcessor and emits a dB spectrogram slice at ~5 Hz.

    Runs only the last 8 frames (mti=False) so the STFT takes ~1-2 ms
    instead of 10-50 ms, keeping the main thread entirely free.
    """
    frame_ready = QtCore.pyqtSignal(np.ndarray)

    _INTERVAL_S  = 0.20   # 5 Hz display rate
    _DISP_FRAMES = 8      # only last 8 frames → ~6× faster than 30
    _DISP_COLS   = 4      # 5 Hz × 4 cols = 20 cols/s scroll speed

    def __init__(self, processor: SpectrogramProcessor):
        super().__init__()
        self._processor  = processor
        self._stop_event = threading.Event()

    @QtCore.pyqtSlot()
    def run(self):
        while not self._stop_event.is_set():
            t0 = _time.monotonic()
            try:
                result = self._processor.get_streaming_result(
                    n_cols=self._DISP_COLS,
                    n_frames=self._DISP_FRAMES,
                    mti=False,           # skip MTI for display — imperceptible
                )
                if result is not None:
                    self.frame_ready.emit(result)
            except Exception:
                pass   # never let the display thread crash the app
            elapsed = _time.monotonic() - t0
            self._stop_event.wait(max(0.02, self._INTERVAL_S - elapsed))

    def stop(self):
        self._stop_event.set()


class RadarBridge(QtCore.QObject):
    frame_ready     = QtCore.pyqtSignal(np.ndarray)
    raw_frame_ready = QtCore.pyqtSignal(np.ndarray)
    error_occurred  = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        # streaming=True: push_frame_raw() is called from the radar thread
        # (deque-only, ~microseconds).  The display worker reads the deque
        # independently on its own thread at 5 Hz.
        self._processor = SpectrogramProcessor(streaming=True)
        self._stream = RadarStream(
            on_frame=self._on_frame,
            on_error=self._on_error,
        )
        self._display_worker: _SpectrogramDisplayWorker = None
        self._display_thread: QtCore.QThread = None

    def _on_frame(self, frame):
        # ── Radar background thread ──────────────────────────────────────────
        # Only emit the raw frame and push it into the deque accumulator.
        # Zero STFT work here → radar thread never falls behind at 10 fps.
        self.raw_frame_ready.emit(frame)
        self._processor.push_frame_raw(frame)

    def _on_error(self, msg):
        self.error_occurred.emit(msg)

    def start(self):
        self._stream.start()
        # Spin up the dedicated display thread
        worker = _SpectrogramDisplayWorker(self._processor)
        thread = QtCore.QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        # Cross-thread signal: frame_ready on the worker → frame_ready on self
        # Qt auto-detects the thread boundary and posts it correctly.
        worker.frame_ready.connect(self.frame_ready)
        thread.start()
        self._display_worker = worker
        self._display_thread = thread

    def stop(self):
        if self._display_worker is not None:
            self._display_worker.stop()
        if self._display_thread is not None:
            self._display_thread.quit()
            self._display_thread.wait(1000)
        self._display_worker = None
        self._display_thread = None
        self._stream.stop()


class PlaceholderTab(QtWidgets.QWidget):
    def __init__(self, title, description, icon="🔒"):
        super().__init__()
        c = app_colors()
        self.setStyleSheet(f"background: {c['bg']};")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(12)

        icon_label = QtWidgets.QLabel(icon)
        icon_label.setStyleSheet("font-size: 52px;")
        icon_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon_label)

        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet(
            f"font-size: 22px; font-weight: bold; color: {c['accent']};"
        )
        title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        desc_label = QtWidgets.QLabel(description)
        desc_label.setStyleSheet(f"font-size: 14px; color: {c['faint']};")
        desc_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        desc_label.setWordWrap(True)
        desc_label.setMaximumWidth(420)
        layout.addWidget(desc_label)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SensDSv2 — Sensing for Data Science")
        self.setWindowIcon(QtGui.QIcon(resource_path("assets/SensDSLogo.png")))
        self.setMinimumSize(860, 540)
        screen = QtGui.QGuiApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            w = min(1280, avail.width() - 40)
            h = min(820, avail.height() - 60)
            self.resize(w, h)
            self._compact = avail.height() < 800   # Surface / small-screen mode
        else:
            self.resize(1200, 740)
            self._compact = False
        self._bridge = None
        self._connected = False
        self._elapsed = 0
        self._hint_index = 0
        self._gamification_mgr = GamificationManager(self)
        self.setStyleSheet(_app_style(app_colors(), compact=self._compact))
        self._setup_ui()
        self._setup_timers()
        # Let the toast overlay the full window (set after show so geometry is valid)
        QtCore.QTimer.singleShot(0, self._init_gamification_toast)

    def _setup_ui(self):
        wrapper = QtWidgets.QWidget()
        self.setCentralWidget(wrapper)
        main_layout = QtWidgets.QVBoxLayout(wrapper)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        main_layout.addWidget(self._build_topbar())

        # Gamification bar sits between the top bar and the tab content
        self._gami_bar = GamificationBar(self._gamification_mgr, wrapper)
        main_layout.addWidget(self._gami_bar)

        main_layout.addWidget(self._build_tabs())

    def _build_topbar(self):
        topbar = QtWidgets.QWidget()
        topbar.setObjectName("topbar")
        topbar_h = 64 if self._compact else 96
        topbar.setFixedHeight(topbar_h)
        layout = QtWidgets.QHBoxLayout(topbar)
        layout.setContentsMargins(16 if self._compact else 24, 0,
                                  16 if self._compact else 24, 0)
        layout.setSpacing(12 if self._compact else 20)

        # Logo
        logo_path = resource_path("assets/SensDSLogo.png")
        if os.path.exists(logo_path):
            logo_container = QtWidgets.QLabel()
            logo_container.setStyleSheet("background: transparent;")
            logo_h = topbar_h - 8
            pixmap = QtGui.QPixmap(logo_path).scaledToHeight(
                logo_h, QtCore.Qt.TransformationMode.SmoothTransformation
            )
            logo_container.setPixmap(pixmap)
            logo_container.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            # White glow so the dark-blue logo text pops off the dark topbar
            # without needing a background pill.
            glow = QtWidgets.QGraphicsDropShadowEffect()
            glow.setBlurRadius(22)
            glow.setColor(QtGui.QColor(255, 255, 255, 200))
            glow.setOffset(0, 0)
            logo_container.setGraphicsEffect(glow)
            layout.addWidget(logo_container)
        else:
            title = QtWidgets.QLabel("SensDSv2")
            title.setObjectName("app_title")
            layout.addWidget(title)

        layout.addStretch()

        # Hint (rotates)
        self._hint_label = QtWidgets.QLabel(HINTS[0])
        self._hint_label.setObjectName("hint_label")
        layout.addWidget(self._hint_label)

        layout.addStretch()

        # Error label
        self._error_label = QtWidgets.QLabel("")
        self._error_label.setObjectName("error_label")
        self._error_label.setVisible(False)
        layout.addWidget(self._error_label)

        # Status dot + text
        self._status_label = QtWidgets.QLabel("⬤  Disconnected")
        self._status_label.setObjectName("status_label")
        layout.addWidget(self._status_label)

        # Timer
        self._timer_label = QtWidgets.QLabel("00:00")
        self._timer_label.setObjectName("timer_label")
        layout.addWidget(self._timer_label)

        # Buttons
        self._connect_btn = QtWidgets.QPushButton("Connect Radar")
        self._connect_btn.setObjectName("connect_btn")
        self._connect_btn.clicked.connect(self._on_connect)
        layout.addWidget(self._connect_btn)

        self._disconnect_btn = QtWidgets.QPushButton("Disconnect")
        self._disconnect_btn.setObjectName("disconnect_btn")
        self._disconnect_btn.clicked.connect(self._on_disconnect)
        self._disconnect_btn.setVisible(False)
        layout.addWidget(self._disconnect_btn)

        return topbar

    def _build_tabs(self):
        self._tabs = QtWidgets.QTabWidget()
        self._tabs.tabBar().setExpanding(True)

        self._spectrogram = SpectrogramWidget()
        self._tabs.addTab(self._spectrogram, "📡   Visualize")

        self._collect_tab = CollectTab()
        self._tabs.addTab(self._collect_tab, "🎙   Collect")

        self._train_tab = TrainTab()
        self._tabs.addTab(self._train_tab, "🧠   Train")

        self._test_tab = TestTab()
        self._tabs.addTab(self._test_tab, "✋   Test")

        self._results_tab = ResultsTab()
        self._tabs.addTab(self._results_tab, "📊   Results")

        self._vex_tab = VexAimTab()
        self._tabs.addTab(self._vex_tab, "🤖   VEX AIM")

        self._tabs.addTab(PlaceholderTab(
            "Resources",
            "Reference materials, gesture guides, and project documentation — coming soon.",
            "📚"
        ), "📚   Resources")

        # Wire test → results live updates
        self._test_tab.prediction_made.connect(self._results_tab.add_prediction)
        self._test_tab.model_loaded.connect(self._results_tab.set_model_info)

        # Wire test tab → gamification manager
        self._test_tab.gesture_tested.connect(self._gamification_mgr.on_prediction)
        self._test_tab.soccer_gesture_applied.connect(self._gamification_mgr.on_soccer_gesture)
        self._test_tab.maze_solved.connect(self._gamification_mgr.on_maze_solved)

        self._tabs.tabBarClicked.connect(self._on_tab_clicked)
        return self._tabs

    def _init_gamification_toast(self):
        """Attach the badge toast overlay to the central widget."""
        self._gami_bar.set_toast_parent(self.centralWidget())

    def _on_tab_clicked(self, index):
        if index == 0 or index == 1:
            return
        if index == 2:
            self._train_tab.refresh()
            return
        if index == 3:
            models_root = os.path.join(os.path.expanduser("~"), "SensDSv2_data", "models")
            has_model = os.path.isdir(models_root) and any(
                os.path.isdir(os.path.join(models_root, d))
                for d in os.listdir(models_root)
                if not d.startswith(".")
            ) if os.path.isdir(models_root) else False
            if has_model:
                self._test_tab.refresh()
                return
            self._show_soft_lock("Train a model first before testing.")
            return
        if index == 4:
            if self._test_tab._model is not None:
                return
            self._show_soft_lock("Test your model first before viewing results.")
            return
        if index == 5:
            models_root = os.path.join(os.path.expanduser("~"), "SensDSv2_data", "models")
            has_model = (
                os.path.isdir(models_root)
                and any(
                    os.path.isdir(os.path.join(models_root, d))
                    for d in os.listdir(models_root)
                    if not d.startswith(".")
                )
            ) if os.path.isdir(models_root) else False
            if has_model:
                return
            self._show_soft_lock(
                "Train and test a model before using VEX AIM."
            )
            return
        if index == 6:
            # Resources tab is always accessible
            return
        self._show_soft_lock(
            "Complete the previous steps first — this tab will unlock as you progress."
        )

    def _show_soft_lock(self, msg):
        dlg = QtWidgets.QMessageBox(self)
        dlg.setWindowTitle("Not ready yet")
        dlg.setText(msg)
        dlg.setIcon(QtWidgets.QMessageBox.Icon.Information)
        dlg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        dlg.exec()

    def _setup_timers(self):
        self._session_timer = QtCore.QTimer()
        self._session_timer.timeout.connect(self._tick_timer)

        self._hint_timer = QtCore.QTimer()
        self._hint_timer.timeout.connect(self._rotate_hint)
        self._hint_timer.start(8000)

    def _on_connect(self):
        self._clear_error()
        try:
            self._bridge = RadarBridge()
            self._bridge.frame_ready.connect(self._spectrogram.update_frame)
            self._bridge.raw_frame_ready.connect(self._collect_tab.on_raw_frame)
            self._bridge.raw_frame_ready.connect(self._test_tab.on_raw_frame)
            self._bridge.raw_frame_ready.connect(self._vex_tab.on_raw_frame)
            self._bridge.frame_ready.connect(self._vex_tab.on_spectrogram_frame)
            self._bridge.error_occurred.connect(self._on_radar_error)
            self._bridge.start()
            self._set_connected(True)
        except Exception as e:
            self._show_error(str(e))

    def _on_disconnect(self):
        if self._bridge:
            self._bridge.stop()
            self._bridge = None
        self._set_connected(False)

    def _on_radar_error(self, msg):
        self._show_error(msg)
        self._set_connected(False)

    def _set_connected(self, connected):
        self._connected = connected
        if connected:
            self._status_label.setText("⬤  Connected")
            self._status_label.setStyleSheet("font-size: 12px; color: #2ecc71;")
            self._connect_btn.setVisible(False)
            self._disconnect_btn.setVisible(True)
            self._elapsed = 0
            self._session_timer.start(1000)
        else:
            self._status_label.setText("⬤  Disconnected")
            self._status_label.setStyleSheet("font-size: 12px; color: #aac4e0;")
            self._connect_btn.setVisible(True)
            self._disconnect_btn.setVisible(False)
            self._session_timer.stop()
            self._timer_label.setText("00:00")

    def _show_error(self, msg):
        self._error_label.setText(f"⚠  {msg}")
        self._error_label.setVisible(True)

    def _clear_error(self):
        self._error_label.setText("")
        self._error_label.setVisible(False)

    def _tick_timer(self):
        self._elapsed += 1
        mins = self._elapsed // 60
        secs = self._elapsed % 60
        self._timer_label.setText(f"{mins:02d}:{secs:02d}")

    def _rotate_hint(self):
        self._hint_index = (self._hint_index + 1) % len(HINTS)
        self._hint_label.setText(HINTS[self._hint_index])

    def closeEvent(self, event):
        if self._bridge:
            self._bridge.stop()
        event.accept()
