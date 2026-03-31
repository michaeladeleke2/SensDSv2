from PyQt6 import QtWidgets, QtGui, QtCore


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


class HintCard(QtWidgets.QLabel):
    """Rotating hint label. Cycles through `hints` every `interval_ms` ms."""

    def __init__(self, hints: list, c: dict | None = None,
                 interval_ms: int = 7000, parent=None):
        super().__init__(parent)
        self._hints = hints
        self._index = 0
        self.setWordWrap(True)
        colors = c or app_colors()
        self.setStyleSheet(
            f"font-size: 12px; color: {colors['hint_text']}; "
            f"background: {colors['hint_bg']}; border-radius: 6px; padding: 10px;"
        )
        self._show()
        if len(hints) > 1:
            t = QtCore.QTimer(self)
            t.timeout.connect(self._advance)
            t.start(interval_ms)

    def _advance(self):
        self._index = (self._index + 1) % len(self._hints)
        self._show()

    def _show(self):
        self.setText(f"💡  {self._hints[self._index]}")
