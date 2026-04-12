"""Gamification — XP, levels, badges, and the persistent GamificationBar widget."""

from PyQt6 import QtWidgets, QtCore, QtGui

# ── level thresholds ───────────────────────────────────────────────────────────
# (min_xp, display_name, accent_color)
_LEVELS = [
    (0,    "Beginner",     "#95a5a6"),
    (100,  "Explorer",     "#3498db"),
    (300,  "Intermediate", "#2ecc71"),
    (600,  "Advanced",     "#f39c12"),
    (1000, "Expert",       "#9b59b6"),
]

# ── badges ─────────────────────────────────────────────────────────────────────
# key → (emoji, short_name, description)
_BADGES: dict[str, tuple[str, str, str]] = {
    "first_prediction": ("🔍", "First Look",       "Made your first gesture prediction!"),
    "high_five":        ("🖐️",  "High Five",        "Made 5 predictions!"),
    "confident_one":    ("🎯", "Sharp Eye",         "Got 80 %+ confidence on a prediction!"),
    "on_fire":          ("🔥", "On Fire",           "Made 20 predictions!"),
    "soccer_rookie":    ("⚽", "Soccer Rookie",     "First RoboSoccer gesture!"),
    "soccer_pro":       ("🥅", "Soccer Pro",        "10 RoboSoccer gestures!"),
    "maze_rookie":      ("🌀", "Maze Rookie",       "Solved your first maze!"),
    "maze_explorer":    ("🗺️",  "Maze Explorer",    "Solved 3 mazes!"),
    "maze_master":      ("🏆", "Maze Master",       "Solved 5 mazes!"),
    "three_stars":      ("⭐", "Perfect Run",        "Solved a maze with 3 stars!"),
    "speed_run":        ("⚡", "Speed Runner",       "Solved a maze in ≤ 10 moves!"),
}


# ══════════════════════════════════════════════════════════════════════════════
# GamificationManager
# ══════════════════════════════════════════════════════════════════════════════

class GamificationManager(QtCore.QObject):
    """Tracks XP, level, and badges. Emits signals when state changes."""

    xp_changed   = QtCore.pyqtSignal(int, int)   # (total_xp, level_idx)
    badge_earned = QtCore.pyqtSignal(str)         # badge key
    level_up     = QtCore.pyqtSignal(int)         # new level_idx

    def __init__(self, parent=None):
        super().__init__(parent)
        self._xp:              int       = 0
        self._level_idx:       int       = 0
        self._badges:          set[str]  = set()
        self._prediction_count: int      = 0
        self._soccer_count:    int       = 0
        self._mazes_solved:    int       = 0

    # ── private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _level_for_xp(xp: int) -> int:
        idx = 0
        for i, (threshold, *_) in enumerate(_LEVELS):
            if xp >= threshold:
                idx = i
        return idx

    def _add_xp(self, amount: int):
        old_level = self._level_idx
        self._xp += amount
        new_level = self._level_for_xp(self._xp)
        self._level_idx = new_level
        self.xp_changed.emit(self._xp, new_level)
        if new_level > old_level:
            self.level_up.emit(new_level)

    def _award(self, key: str):
        if key not in self._badges and key in _BADGES:
            self._badges.add(key)
            self.badge_earned.emit(key)

    # ── public event hooks ─────────────────────────────────────────────────────

    def on_prediction(self, gesture: str, confidence: float):
        """Call when a single-mode prediction is made."""
        self._prediction_count += 1
        self._add_xp(10)
        if self._prediction_count == 1:
            self._award("first_prediction")
        if self._prediction_count >= 5:
            self._award("high_five")
        if self._prediction_count >= 20:
            self._award("on_fire")
        if confidence >= 0.80:
            self._award("confident_one")

    def on_soccer_gesture(self, gesture: str):
        """Call when a non-idle RoboSoccer gesture is applied."""
        self._soccer_count += 1
        self._add_xp(15)
        if self._soccer_count == 1:
            self._award("soccer_rookie")
        if self._soccer_count >= 10:
            self._award("soccer_pro")

    def on_maze_solved(self, stars: int, moves: int):
        """Call when the player reaches the maze exit."""
        self._mazes_solved += 1
        bonus = {3: 15, 2: 5, 1: 0}.get(stars, 0)
        self._add_xp(25 + bonus)
        if self._mazes_solved == 1:
            self._award("maze_rookie")
        if self._mazes_solved >= 3:
            self._award("maze_explorer")
        if self._mazes_solved >= 5:
            self._award("maze_master")
        if stars == 3:
            self._award("three_stars")
        if moves <= 10:
            self._award("speed_run")

    # ── read-only accessors ────────────────────────────────────────────────────

    @property
    def xp(self) -> int:
        return self._xp

    @property
    def level_idx(self) -> int:
        return self._level_idx

    @property
    def badges(self) -> frozenset:
        return frozenset(self._badges)

    @property
    def level_xp_range(self) -> tuple[int, int]:
        """(xp_start, xp_end) for the current level band."""
        start = _LEVELS[self._level_idx][0]
        if self._level_idx + 1 < len(_LEVELS):
            end = _LEVELS[self._level_idx + 1][0]
        else:
            end = _LEVELS[-1][0] + 500
        return start, end


# ══════════════════════════════════════════════════════════════════════════════
# BadgeToast  — slides in from the top-right corner
# ══════════════════════════════════════════════════════════════════════════════

class BadgeToast(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("""
            BadgeToast {
                background: #1a3a5c;
                border-radius: 12px;
                border: 2px solid #f39c12;
            }
        """)
        self.setFixedSize(290, 68)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(10)

        self._icon_lbl = QtWidgets.QLabel("🏅")
        self._icon_lbl.setStyleSheet(
            "font-size: 28px; background: transparent; border: none;"
        )
        layout.addWidget(self._icon_lbl)

        text_col = QtWidgets.QVBoxLayout()
        text_col.setSpacing(2)

        self._title_lbl = QtWidgets.QLabel("Badge Earned!")
        self._title_lbl.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #f39c12;"
            "background: transparent; border: none;"
        )
        text_col.addWidget(self._title_lbl)

        self._desc_lbl = QtWidgets.QLabel("")
        self._desc_lbl.setStyleSheet(
            "font-size: 11px; color: #ecf0f1; background: transparent; border: none;"
        )
        self._desc_lbl.setWordWrap(True)
        text_col.addWidget(self._desc_lbl)

        layout.addLayout(text_col)
        layout.addStretch()

        # Slide animation
        self._anim = QtCore.QPropertyAnimation(self, b"pos")
        self._anim.setDuration(380)
        self._anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)

        self._out_anim = QtCore.QPropertyAnimation(self, b"pos")
        self._out_anim.setDuration(300)
        self._out_anim.setEasingCurve(QtCore.QEasingCurve.Type.InCubic)
        self._out_anim.finished.connect(self.hide)

        self._hide_timer = QtCore.QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._slide_out)

        # Queue of badges to show in sequence
        self._queue: list[str] = []
        self._showing = False
        self.hide()

    def show_badge(self, key: str):
        self._queue.append(key)
        if not self._showing:
            self._next()

    def _next(self):
        if not self._queue:
            self._showing = False
            return
        key = self._queue.pop(0)
        if key not in _BADGES:
            self._next()
            return
        self._showing = True
        icon, name, desc = _BADGES[key]
        self._icon_lbl.setText(icon)
        self._title_lbl.setText(f"🏅 Badge: {name}")
        self._desc_lbl.setText(desc)
        self._slide_in()

    def _slide_in(self):
        p = self.parent()
        if p is None:
            return
        pw, ph = p.width(), p.height()
        hidden = QtCore.QPoint(pw - self.width() - 12, -self.height() - 4)
        shown  = QtCore.QPoint(pw - self.width() - 12, 54)   # below gamification bar
        self.move(hidden)
        self.show()
        self.raise_()
        self._anim.setStartValue(hidden)
        self._anim.setEndValue(shown)
        self._anim.start()
        self._hide_timer.start(3600)

    def _slide_out(self):
        p = self.parent()
        if p is None:
            return
        pw = p.width()
        hidden = QtCore.QPoint(pw - self.width() - 12, -self.height() - 4)
        self._out_anim.setStartValue(self.pos())
        self._out_anim.setEndValue(hidden)
        # Disconnect old finished connections to avoid accumulation
        try:
            self._out_anim.finished.disconnect()
        except Exception:
            pass
        self._out_anim.finished.connect(self.hide)
        self._out_anim.finished.connect(self._next)
        self._out_anim.start()


# ══════════════════════════════════════════════════════════════════════════════
# GamificationBar  — compact strip inserted between topbar and tabs
# ══════════════════════════════════════════════════════════════════════════════

class GamificationBar(QtWidgets.QWidget):
    def __init__(self, manager: GamificationManager, parent=None):
        super().__init__(parent)
        self._mgr = manager
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(46)
        self.setStyleSheet("""
            GamificationBar {
                background: #0f2840;
                border-bottom: 1px solid #1e4a78;
            }
        """)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(16)

        # ── Level chip ─────────────────────────────────────────────────────────
        self._level_lbl = QtWidgets.QLabel("⭐  Level 1 · Beginner")
        self._level_lbl.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: #95a5a6;"
            "background: #1a3a5c; border-radius: 9px; padding: 3px 10px;"
        )
        layout.addWidget(self._level_lbl)

        # ── XP column (label + bar) ────────────────────────────────────────────
        xp_col = QtWidgets.QVBoxLayout()
        xp_col.setSpacing(2)
        xp_col.setContentsMargins(0, 0, 0, 0)

        self._xp_lbl = QtWidgets.QLabel("XP: 0 / 100")
        self._xp_lbl.setStyleSheet(
            "font-size: 10px; color: #aac4e0; font-weight: bold; background: transparent;"
        )
        xp_col.addWidget(self._xp_lbl)

        self._xp_bar = QtWidgets.QProgressBar()
        self._xp_bar.setRange(0, 100)
        self._xp_bar.setValue(0)
        self._xp_bar.setTextVisible(False)
        self._xp_bar.setFixedHeight(9)
        self._xp_bar.setMinimumWidth(220)
        self._xp_bar.setStyleSheet("""
            QProgressBar {
                background: #1e4a78;
                border-radius: 4px;
                border: none;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #3498db, stop:1 #9b59b6);
                border-radius: 4px;
            }
        """)
        xp_col.addWidget(self._xp_bar)
        layout.addLayout(xp_col)

        layout.addStretch()

        # ── Stats chips ────────────────────────────────────────────────────────
        self._badge_lbl = QtWidgets.QLabel("🏅  Badges: 0")
        self._badge_lbl.setStyleSheet(
            "font-size: 12px; color: #f39c12; font-weight: bold; background: transparent;"
        )
        layout.addWidget(self._badge_lbl)

        view_btn = QtWidgets.QPushButton("View All →")
        view_btn.setFixedHeight(28)
        view_btn.setStyleSheet("""
            QPushButton {
                background: #1a3a5c; color: #aac4e0;
                border: 1px solid #2e5f8a; border-radius: 6px;
                font-size: 11px; padding: 0 10px;
            }
            QPushButton:hover { background: #234e7a; color: white; }
        """)
        view_btn.clicked.connect(self._show_badge_panel)
        layout.addWidget(view_btn)

        # ── Wire manager signals ───────────────────────────────────────────────
        manager.xp_changed.connect(self._on_xp_changed)
        manager.badge_earned.connect(self._on_badge_earned)
        manager.level_up.connect(self._on_level_up)

        # Toast (parented to the main window widget set via set_toast_parent)
        self._toast: BadgeToast | None = None

    def set_toast_parent(self, widget: QtWidgets.QWidget):
        """Call after the window is shown so the toast can overlay correctly."""
        self._toast = BadgeToast(widget)

    # ── slots ──────────────────────────────────────────────────────────────────

    def _on_xp_changed(self, xp: int, level_idx: int):
        _, name, color = _LEVELS[level_idx]
        start, end = self._mgr.level_xp_range
        progress = max(0, min(100, int((xp - start) / max(end - start, 1) * 100)))
        self._level_lbl.setText(f"⭐  Level {level_idx + 1} · {name}")
        self._level_lbl.setStyleSheet(
            f"font-size: 12px; font-weight: bold; color: {color};"
            "background: #1a3a5c; border-radius: 9px; padding: 3px 10px;"
        )
        self._xp_bar.setValue(progress)
        self._xp_lbl.setText(f"XP: {xp} / {end}")

    def _on_badge_earned(self, key: str):
        count = len(self._mgr.badges)
        self._badge_lbl.setText(f"🏅  Badges: {count}")
        if self._toast:
            self._toast.show_badge(key)

    def _on_level_up(self, level_idx: int):
        _, name, color = _LEVELS[level_idx]
        # Flash gold for 1.5 s then restore
        self._level_lbl.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: white;"
            "background: #f39c12; border-radius: 9px; padding: 3px 10px;"
        )
        self._level_lbl.setText(f"🎉  Level Up! Level {level_idx + 1} · {name}")
        QtCore.QTimer.singleShot(
            1600, lambda: self._on_xp_changed(self._mgr.xp, level_idx)
        )

    def _show_badge_panel(self):
        dlg = _BadgePanel(self._mgr, self.window())
        dlg.exec()


# ══════════════════════════════════════════════════════════════════════════════
# Badge panel dialog
# ══════════════════════════════════════════════════════════════════════════════

class _BadgePanel(QtWidgets.QDialog):
    def __init__(self, manager: GamificationManager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Your Badges")
        self.setMinimumWidth(440)
        self.setStyleSheet("QDialog { background: #0f2840; } QLabel { color: #ecf0f1; }")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(20, 16, 20, 16)

        # Header
        hdr = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("🏅  Your Badges")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #f39c12;")
        hdr.addWidget(title)
        hdr.addStretch()
        xp_chip = QtWidgets.QLabel(f"⭐ {manager.xp} XP  ·  Level {manager.level_idx + 1}")
        xp_chip.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: #aac4e0;"
            "background: #1a3a5c; border-radius: 8px; padding: 4px 10px;"
        )
        hdr.addWidget(xp_chip)
        layout.addLayout(hdr)

        # Scrollable badge list
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        inner = QtWidgets.QWidget()
        inner.setStyleSheet("background: transparent;")
        inner_layout = QtWidgets.QVBoxLayout(inner)
        inner_layout.setSpacing(6)
        inner_layout.setContentsMargins(0, 0, 0, 0)

        earned = manager.badges
        for key, (icon, name, desc) in _BADGES.items():
            is_earned = key in earned
            row = QtWidgets.QFrame()
            row.setStyleSheet(f"""
                QFrame {{
                    background: {'#1a3a5c' if is_earned else '#0d1e2e'};
                    border: 1px solid {'#f39c12' if is_earned else '#1e3a55'};
                    border-radius: 8px;
                }}
                QLabel {{ border: none; background: transparent; }}
            """)
            rl = QtWidgets.QHBoxLayout(row)
            rl.setContentsMargins(12, 8, 12, 8)
            rl.setSpacing(12)

            icon_lbl = QtWidgets.QLabel(icon if is_earned else "🔒")
            icon_lbl.setStyleSheet(
                f"font-size: 24px; color: {'white' if is_earned else '#334'};"
            )
            rl.addWidget(icon_lbl)

            text_col = QtWidgets.QVBoxLayout()
            text_col.setSpacing(1)
            name_lbl = QtWidgets.QLabel(name)
            name_lbl.setStyleSheet(
                f"font-size: 13px; font-weight: bold;"
                f"color: {'#f39c12' if is_earned else '#445'};"
            )
            text_col.addWidget(name_lbl)
            desc_lbl = QtWidgets.QLabel(desc)
            desc_lbl.setStyleSheet(
                f"font-size: 11px; color: {'#aac4e0' if is_earned else '#334'};"
            )
            text_col.addWidget(desc_lbl)
            rl.addLayout(text_col)
            rl.addStretch()

            if is_earned:
                check = QtWidgets.QLabel("✓  Earned")
                check.setStyleSheet(
                    "font-size: 12px; color: #2ecc71; font-weight: bold;"
                )
                rl.addWidget(check)

            inner_layout.addWidget(row)

        inner_layout.addStretch()
        scroll.setWidget(inner)
        layout.addWidget(scroll)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.setFixedHeight(36)
        close_btn.setStyleSheet("""
            QPushButton {
                background: #1a3a5c; color: white; border: none;
                border-radius: 8px; padding: 0 20px; font-size: 13px;
            }
            QPushButton:hover { background: #234e7a; }
        """)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
