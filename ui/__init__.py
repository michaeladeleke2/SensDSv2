from PyQt6 import QtWidgets, QtGui, QtCore
from ui.gamification import GamificationManager, GamificationBar, BadgeToast  # noqa: F401


# ─── Gesture window indicator (shared by TestTab and VexAimTab) ───────────────

class GestureWindowBar(QtWidgets.QFrame):
    """
    A slim status bar shown above the game area during continuous modes.
    Tells the student exactly what state the inference pipeline is in so
    they know when to perform a gesture vs. when to wait.

    States
    ──────
    ready    (green)  — the window is open; do a gesture now
    reading  (blue)   — inference is running; we're analysing what you did
    cooldown (orange) — a gesture just fired; wait for the countdown
    """

    _CFG = {
        "ready": {
            "bg":     "#1a4228",
            "border": "#27ae60",
            "fg":     "#2ecc71",
            "icon":   "🟢",
            "text":   "Gesture window open — do a gesture now!",
        },
        "reading": {
            "bg":     "#1a2a45",
            "border": "#2980b9",
            "fg":     "#5dade2",
            "icon":   "🔵",
            "text":   "Reading your gesture…",
        },
        "cooldown": {
            "bg":     "#3d2000",
            "border": "#e67e22",
            "fg":     "#f39c12",
            "icon":   "🟠",
            "text":   "Wait — next window opens in",
        },
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(42)
        self.setVisible(False)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(14, 0, 14, 0)
        layout.setSpacing(10)

        self._icon_lbl = QtWidgets.QLabel()
        self._icon_lbl.setFixedWidth(22)
        layout.addWidget(self._icon_lbl)

        self._text_lbl = QtWidgets.QLabel()
        self._text_lbl.setStyleSheet("font-size: 13px; font-weight: bold; border: none;")
        layout.addWidget(self._text_lbl, 1)

        self._cd_lbl = QtWidgets.QLabel()
        self._cd_lbl.setFixedWidth(46)
        self._cd_lbl.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        self._cd_lbl.setStyleSheet(
            "font-size: 14px; font-family: monospace; font-weight: bold; border: none;"
        )
        layout.addWidget(self._cd_lbl)

        self._state = ""

    def _apply(self, state: str, countdown_s: float = 0.0):
        if self._state == state and state != "cooldown":
            return   # avoid unnecessary redraws when state hasn't changed
        self._state = state
        cfg = self._CFG[state]
        self.setStyleSheet(
            f"QFrame {{ background: {cfg['bg']}; border-radius: 8px;"
            f" border: 1px solid {cfg['border']}; }}"
        )
        color_css = f"color: {cfg['fg']}; border: none;"
        self._icon_lbl.setText(cfg["icon"])
        self._icon_lbl.setStyleSheet(f"font-size: 15px; {color_css}")
        self._text_lbl.setText(cfg["text"])
        self._text_lbl.setStyleSheet(f"font-size: 13px; font-weight: bold; {color_css}")
        if state == "cooldown":
            self._cd_lbl.setText(f"{countdown_s:.1f}s")
            self._cd_lbl.setStyleSheet(
                f"font-size: 14px; font-family: monospace; font-weight: bold; {color_css}"
            )
        else:
            self._cd_lbl.setText("")

    def show_ready(self):
        self._apply("ready")
        self.setVisible(True)

    def show_reading(self):
        self._apply("reading")
        self.setVisible(True)

    def show_cooldown(self, secs: float):
        self._apply("cooldown", max(0.0, secs))
        self.setVisible(True)

    def hide_bar(self):
        self.setVisible(False)
        self._state = ""


def _scrollable_left(content_widget: QtWidgets.QWidget, width: int = 300) -> QtWidgets.QScrollArea:
    """
    Wrap a left-panel content widget in a vertically-scrollable QScrollArea.
    The content widget keeps its object name (and thus its stylesheet rules),
    while the fixed width moves to the scroll area so it stays locked on resize.
    """
    scroll = QtWidgets.QScrollArea()
    scroll.setWidget(content_widget)
    scroll.setWidgetResizable(True)
    scroll.setFixedWidth(width)
    scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    scroll.setStyleSheet("""
        QScrollArea { background: transparent; }
        QScrollBar:vertical {
            width: 6px; background: transparent; margin: 0;
        }
        QScrollBar::handle:vertical {
            background: #cccccc; border-radius: 3px; min-height: 20px;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
    """)
    return scroll


class HintCard(QtWidgets.QWidget):
    """Rotating hint card that cycles through a list of tip strings."""

    def __init__(self, hints: list, c: dict | None = None, interval_ms: int = 6000, parent=None):
        super().__init__(parent)
        self._hints = hints
        self._index = 0

        colors = c or {}
        bg = colors.get("hint_bg", "#f0f4ff")
        fg = colors.get("hint_text", "#888888")
        border = colors.get("border", "#dddddd")

        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"HintCard {{ background:{bg}; border:1px solid {border}; border-radius:6px; }}"
        )

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        icon = QtWidgets.QLabel("💡")
        icon.setFixedWidth(20)
        layout.addWidget(icon)

        self._label = QtWidgets.QLabel(hints[0] if hints else "")
        self._label.setWordWrap(True)
        self._label.setStyleSheet(f"color:{fg}; background:transparent; border:none; font-size:12px;")
        layout.addWidget(self._label, 1)

        if len(hints) > 1:
            self._timer = QtCore.QTimer(self)
            self._timer.setInterval(interval_ms)
            self._timer.timeout.connect(self._next_hint)
            self._timer.start()

    def _next_hint(self):
        self._index = (self._index + 1) % len(self._hints)
        self._label.setText(self._hints[self._index])


def is_dark_mode() -> bool:
    palette = QtWidgets.QApplication.palette()
    return palette.color(QtGui.QPalette.ColorRole.Window).lightness() < 128


def app_colors() -> dict:
    """Return a colour scheme dict adapted to the current system theme."""
    if is_dark_mode():
        return dict(
            bg='#1c1c1e',
            panel='#2c2c2e',
            border='#3c3c3e',
            input_bg='#3a3a3a',
            input_border='#555555',
            text='#f0f0f0',
            subtext='#b0b0b0',
            faint='#888888',
            accent='#5b9bd5',
            tab_hover='#2a2a2c',
            divider='#3c3c3e',
            hint_bg='#252535',
            hint_text='#a0a0a0',
            progress_bg='#3a3a3a',
        )
    return dict(
        bg='#f0f2f5',
        panel='#ffffff',
        border='#dddddd',
        input_bg='white',
        input_border='#cccccc',
        text='#333333',
        subtext='#555555',
        faint='#777777',
        accent='#1a3a5c',
        tab_hover='#e2e8f0',
        divider='#eeeeee',
        hint_bg='#f0f4ff',
        hint_text='#888888',
        progress_bg='#e0e0e0',
    )
