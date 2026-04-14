from PyQt6 import QtWidgets, QtGui, QtCore
from ui.gamification import GamificationManager, GamificationBar, BadgeToast  # noqa: F401


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
