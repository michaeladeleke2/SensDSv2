import os
import csv
import datetime
import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui
from ui import HintCard


RESULTS_DIR = os.path.join(os.path.expanduser("~"), "SensDSv2_data", "results")

RESULTS_STYLE = """
    QWidget#results_root { background: #f0f2f5; }

    QFrame#res_card {
        background: white;
        border: 1px solid #dddddd;
        border-radius: 8px;
    }
    QLabel#res_heading {
        font-size: 16px;
        font-weight: bold;
        color: #1a3a5c;
    }
    QLabel#res_section_lbl {
        font-size: 12px;
        font-weight: bold;
        color: #1a3a5c;
    }
    QLabel#res_subtext {
        font-size: 11px;
        color: #888;
    }
    QLabel#res_model_pill {
        font-size: 12px;
        font-weight: bold;
        color: #1a3a5c;
        background: #e8f0fb;
        border-radius: 4px;
        padding: 3px 10px;
    }
    QLabel#res_summary {
        font-size: 12px;
        color: #555;
    }
    QTableWidget {
        background: white;
        border: none;
        font-size: 12px;
        gridline-color: #eee;
    }
    QTableWidget::item { padding: 4px 8px; }
    QTableWidget::item:selected { background: #e8f0fb; color: #1a3a5c; }
    QHeaderView::section {
        background: #f7f8fc;
        color: #1a3a5c;
        font-size: 12px;
        font-weight: bold;
        padding: 6px 8px;
        border: none;
        border-bottom: 2px solid #ddd;
    }
    QPushButton#res_btn {
        background: white;
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 5px 14px;
        font-size: 12px;
        color: #1a3a5c;
        font-weight: bold;
    }
    QPushButton#res_btn:hover { background: #f0f0f0; }
    QPushButton#res_btn:disabled { color: #aaa; border-color: #ddd; }
    QPushButton#res_clear_btn {
        background: white;
        border: 1px solid #f5c6c6;
        border-radius: 5px;
        padding: 5px 14px;
        font-size: 12px;
        color: #c0392b;
        font-weight: bold;
    }
    QPushButton#res_clear_btn:hover { background: #fff5f5; }
    QPushButton#res_clear_btn:disabled { color: #ccc; border-color: #eee; }
"""


def _make_card():
    frame = QtWidgets.QFrame()
    frame.setObjectName("res_card")
    layout = QtWidgets.QVBoxLayout(frame)
    layout.setContentsMargins(14, 12, 14, 12)
    layout.setSpacing(6)
    return frame, layout


def _section_label(text):
    lbl = QtWidgets.QLabel(text)
    lbl.setObjectName("res_section_lbl")
    return lbl


def _sub_label(text):
    lbl = QtWidgets.QLabel(text)
    lbl.setObjectName("res_subtext")
    return lbl


# ─── confusion matrix ────────────────────────────────────────────────────────

class ConfusionMatrixWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._classes: list = []
        self._matrix: np.ndarray = np.zeros((0, 0), dtype=int)
        self.setMinimumSize(150, 120)

    def set_classes(self, classes: list):
        self._classes = list(classes)
        self._matrix = np.zeros((len(classes), len(classes)), dtype=int)
        self.update()

    def record(self, actual: str, predicted: str):
        if actual not in self._classes or predicted not in self._classes:
            return
        self._matrix[self._classes.index(actual), self._classes.index(predicted)] += 1
        self.update()

    def reset(self):
        self._matrix = np.zeros_like(self._matrix)
        self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        n = len(self._classes)
        w, h = self.width(), self.height()

        if n == 0:
            p.setPen(QtGui.QColor("#aaa"))
            p.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter,
                       "No predictions yet.")
            return

        navy = QtGui.QColor("#1a3a5c")

        # ── dynamic label sizing ──────────────────────────────────────────────
        # Measure the longest row label so we can allocate exact space.
        fm_font = QtGui.QFont()
        fm_font.setPixelSize(11)
        fm = QtGui.QFontMetrics(fm_font)
        longest = max((cls.replace("_", " ") for cls in self._classes),
                      key=lambda s: fm.horizontalAdvance(s))
        row_label_w = fm.horizontalAdvance(longest) + 10   # px for row labels
        col_label_h = fm.horizontalAdvance(longest) + 10   # rotated → same budget
        axis_title_w = 18   # width of rotated "Actual →" text
        axis_title_h = 18

        left_margin  = axis_title_w + row_label_w
        top_margin   = axis_title_h + col_label_h

        avail_w = max(1, w - left_margin - 4)
        avail_h = max(1, h - top_margin - 4)
        cell = max(12, min(avail_w // n, avail_h // n))

        grid_w = cell * n
        grid_h = cell * n
        # Centre the grid in the available space
        x0 = left_margin + max(0, (avail_w - grid_w) // 2)
        y0 = top_margin  + max(0, (avail_h - grid_h) // 2)

        max_val = int(self._matrix.max()) if self._matrix.max() > 0 else 1

        lbl_font = QtGui.QFont()
        lbl_font.setPixelSize(max(8, min(11, cell // 4)))
        val_font = QtGui.QFont()
        val_font.setPixelSize(max(8, min(13, cell // 3)))
        val_font.setBold(True)
        axis_font = QtGui.QFont()
        axis_font.setPixelSize(10)
        axis_font.setBold(True)

        # ── cells ──
        for r in range(n):
            for c in range(n):
                val = int(self._matrix[r, c])
                t = val / max_val
                red   = int(255 + (26  - 255) * t)
                green = int(255 + (58  - 255) * t)
                blue  = int(255 + (92  - 255) * t)
                cx, cy = x0 + c * cell, y0 + r * cell
                p.setPen(QtGui.QPen(QtGui.QColor("#e0e0e0"), 1))
                p.setBrush(QtGui.QColor(red, green, blue))
                p.drawRect(cx, cy, cell - 1, cell - 1)
                p.setFont(val_font)
                p.setPen(QtCore.Qt.GlobalColor.white if t > 0.45 else navy)
                p.drawText(QtCore.QRectF(cx, cy, cell - 1, cell - 1),
                           QtCore.Qt.AlignmentFlag.AlignCenter, str(val))

        # ── column labels (top, rotated -55°) ──
        p.setFont(lbl_font)
        p.setPen(navy)
        for c, cls in enumerate(self._classes):
            label = cls.replace("_", " ")
            cx = x0 + c * cell + cell // 2
            p.save()
            p.translate(cx, y0 - 4)
            p.rotate(-55)
            p.drawText(QtCore.QRectF(-(col_label_h + 4), -10,
                                     col_label_h + 4, 14),
                       QtCore.Qt.AlignmentFlag.AlignRight |
                       QtCore.Qt.AlignmentFlag.AlignVCenter, label)
            p.restore()

        # ── row labels (left) ──
        for r, cls in enumerate(self._classes):
            label = cls.replace("_", " ")
            cy = y0 + r * cell
            p.setFont(lbl_font)
            p.setPen(navy)
            p.drawText(QtCore.QRectF(axis_title_w, cy, row_label_w - 4, cell),
                       QtCore.Qt.AlignmentFlag.AlignRight |
                       QtCore.Qt.AlignmentFlag.AlignVCenter, label)

        # ── axis titles ──
        p.setFont(axis_font)
        p.setPen(QtGui.QColor("#777"))
        # "Predicted →" across the top
        p.drawText(QtCore.QRectF(x0, 2, grid_w, axis_title_h),
                   QtCore.Qt.AlignmentFlag.AlignCenter, "Predicted →")
        # "Actual →" rotated on the far left
        p.save()
        p.translate(axis_title_w // 2, y0 + grid_h // 2)
        p.rotate(-90)
        p.drawText(QtCore.QRectF(-30, -8, 60, 16),
                   QtCore.Qt.AlignmentFlag.AlignCenter, "Actual →")
        p.restore()


# ─── per-class accuracy bars ─────────────────────────────────────────────────

class AccuracyBarsWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._classes: list = []
        self._correct: dict = {}
        self._total: dict = {}
        self.setMinimumHeight(60)

    def set_classes(self, classes: list):
        self._classes = list(classes)
        self._correct = {c: 0 for c in classes}
        self._total   = {c: 0 for c in classes}
        self.update()

    def record(self, actual: str, predicted: str):
        if actual not in self._classes:
            return
        self._total[actual] = self._total.get(actual, 0) + 1
        if actual == predicted:
            self._correct[actual] = self._correct.get(actual, 0) + 1
        self.update()

    def reset(self):
        self._correct = {c: 0 for c in self._classes}
        self._total   = {c: 0 for c in self._classes}
        self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        n = len(self._classes)

        if n == 0:
            p.setPen(QtGui.QColor("#aaa"))
            p.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter,
                       "No predictions yet.")
            return

        margin  = 5
        label_w = 100
        pct_w   = 48
        cnt_w   = 40
        bar_w   = max(20, w - margin * 2 - label_w - pct_w - cnt_w)
        spacing = margin
        bar_h   = max(10, (h - spacing * (n + 1)) // n)

        font = QtGui.QFont()
        font.setPixelSize(11)
        p.setFont(font)
        navy = QtGui.QColor("#1a3a5c")

        for i, cls in enumerate(self._classes):
            y   = spacing + i * (bar_h + spacing)
            tot = self._total.get(cls, 0)
            cor = self._correct.get(cls, 0)
            acc = cor / tot if tot > 0 else 0.0
            bar_px = int(bar_w * acc)

            color = (navy if acc >= 0.7
                     else QtGui.QColor("#e67e22") if acc >= 0.4
                     else QtGui.QColor("#c0392b"))

            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(QtGui.QColor("#eeeeee"))
            p.drawRoundedRect(margin + label_w, y, bar_w, bar_h, 3, 3)
            if bar_px > 0:
                p.setBrush(color)
                p.drawRoundedRect(margin + label_w, y, bar_px, bar_h, 3, 3)

            p.setPen(QtGui.QColor("#333"))
            p.drawText(QtCore.QRectF(margin, y, label_w - 4, bar_h),
                       QtCore.Qt.AlignmentFlag.AlignRight |
                       QtCore.Qt.AlignmentFlag.AlignVCenter,
                       cls.replace("_", " "))

            p.setPen(QtGui.QColor("#555"))
            pct_str = f"{acc:.0%}" if tot > 0 else "—"
            p.drawText(QtCore.QRectF(margin + label_w + bar_w + 4, y, pct_w, bar_h),
                       QtCore.Qt.AlignmentFlag.AlignLeft |
                       QtCore.Qt.AlignmentFlag.AlignVCenter, pct_str)

            p.setPen(QtGui.QColor("#999"))
            cnt_str = f"({cor}/{tot})" if tot > 0 else ""
            p.drawText(QtCore.QRectF(margin + label_w + bar_w + pct_w + 2, y, cnt_w, bar_h),
                       QtCore.Qt.AlignmentFlag.AlignLeft |
                       QtCore.Qt.AlignmentFlag.AlignVCenter, cnt_str)


# ─── tab ─────────────────────────────────────────────────────────────────────

class ResultsTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("results_root")
        self.setStyleSheet(RESULTS_STYLE)
        self._classes: list = []
        self._history: list = []
        self._setup_ui()

    # ── public API ────────────────────────────────────────────────────────────

    def set_model_info(self, model_name: str, classes: list):
        self._classes = list(classes)
        self._model_pill.setText(f"Model:  {model_name}")
        self._classes_lbl.setText("Gestures:  " + "  ·  ".join(classes))
        self._matrix.set_classes(classes)
        self._acc_bars.set_classes(classes)
        self._history.clear()
        self._table.setRowCount(0)
        self._summary_lbl.setText("")
        self._export_btn.setEnabled(False)
        self._clear_btn.setEnabled(False)

    def add_prediction(self, gesture: str, confidence: float, threshold: float,
                       actual=None):
        dt = datetime.datetime.now()
        self._history.append((dt, gesture, confidence, threshold, actual))

        # Confusion matrix and accuracy bars only update when the student
        # confirmed their actual gesture (Single Prediction mode only).
        if actual is not None:
            self._matrix.record(actual, gesture)
            self._acc_bars.record(actual, gesture)

        row = self._table.rowCount()
        self._table.insertRow(row)

        def _item(text, align=None, color=None, bold=False):
            it = QtWidgets.QTableWidgetItem(text)
            if align:
                it.setTextAlignment(align)
            if color:
                it.setForeground(QtGui.QColor(color))
            if bold:
                f = QtGui.QFont()
                f.setBold(True)
                it.setFont(f)
            return it

        ok = confidence >= threshold
        center = QtCore.Qt.AlignmentFlag.AlignCenter
        actual_text = actual.replace("_", " ") if actual else "—"

        self._table.setItem(row, 0, _item(dt.strftime("%H:%M:%S"), center))
        self._table.setItem(row, 1,
            _item(gesture.replace("_", " "), color="#1a3a5c", bold=True))
        self._table.setItem(row, 2, _item(f"{confidence:.1%}", center))
        self._table.setItem(row, 3, _item(actual_text, center))
        self._table.setItem(row, 4,
            _item("✓  Confident" if ok else "⚠  Low", center,
                  "#27ae60" if ok else "#e67e22"))

        self._table.scrollToBottom()
        self._export_btn.setEnabled(True)
        self._clear_btn.setEnabled(True)

        n = len(self._history)
        above = sum(1 for _, _, c, t, _ in self._history if c >= t)
        self._summary_lbl.setText(
            f"{n} prediction{'s' if n != 1 else ''}  —  "
            f"{above} confident  ({above / n:.0%})"
        )

    # ── UI ────────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(14, 10, 14, 10)
        root.setSpacing(8)
        root.addWidget(self._build_header())
        root.addLayout(self._build_middle(), 3)
        root.addWidget(self._build_history(), 2)

    def _build_header(self):
        frame, layout = _make_card()
        layout.setContentsMargins(14, 10, 14, 10)
        inner = QtWidgets.QHBoxLayout()
        inner.setSpacing(12)

        heading = QtWidgets.QLabel("Results")
        heading.setObjectName("res_heading")
        inner.addWidget(heading)

        self._model_pill = QtWidgets.QLabel("No model loaded.")
        self._model_pill.setObjectName("res_model_pill")
        inner.addWidget(self._model_pill)

        self._classes_lbl = QtWidgets.QLabel("")
        self._classes_lbl.setObjectName("res_subtext")
        self._classes_lbl.setStyleSheet(
            "font-size: 12px; color: #555; border: none;"
        )
        inner.addWidget(self._classes_lbl, 1)

        self._summary_lbl = QtWidgets.QLabel("")
        self._summary_lbl.setObjectName("res_summary")
        self._summary_lbl.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        inner.addWidget(self._summary_lbl)

        layout.addLayout(inner)
        frame.setFixedHeight(50)
        return frame

    def _build_middle(self):
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)

        # ── confusion matrix ──
        cf_frame, cf_layout = _make_card()
        cf_layout.addWidget(_section_label("Confusion Matrix"))
        cf_layout.addWidget(_sub_label(
            "Each row is what you actually did · each column is what the model guessed"
        ))
        cf_layout.addWidget(_sub_label(
            "Populated from Single Prediction mode only."
        ))
        self._matrix = ConfusionMatrixWidget()
        cf_layout.addWidget(self._matrix, 1)
        row.addWidget(cf_frame, 3)

        # ── right column: accuracy bars + hint ──
        right_col = QtWidgets.QVBoxLayout()
        right_col.setSpacing(8)

        acc_frame, acc_layout = _make_card()
        acc_layout.addWidget(_section_label("Accuracy per Gesture"))
        acc_layout.addWidget(_sub_label(
            "Green ≥ 70%  ·  Orange ≥ 40%  ·  Red < 40%  ·  shows correct / total"
        ))
        self._acc_bars = AccuracyBarsWidget()
        acc_layout.addWidget(self._acc_bars, 1)
        right_col.addWidget(acc_frame, 2)

        hint_frame, hint_layout = _make_card()
        hint_layout.addWidget(HintCard([
            "Confusion matrix: each box shows how many times the model guessed "
            "that column's gesture when you actually did the row's gesture. "
            "Dark diagonal = great — it means the model got it right!",
            "Accuracy per gesture: how often the model was correct for each move. "
            "If 'push' is low, try collecting more push samples and retraining.",
            "Confidence: the model gives every gesture a score from 0–100%. "
            "A high score (green ✓) means it's sure. Low (orange ⚠) means it's guessing.",
            "Threshold: you set the minimum confidence before an action happens. "
            "60% is a safe starting point — lower it if the robot feels unresponsive.",
            "Export CSV saves the full prediction log as a spreadsheet "
            "you can open in Excel or Google Sheets.",
            "The diagonal of the confusion matrix should be the darkest column. "
            "Off-diagonal dark boxes mean the model is mixing up those two gestures.",
        ]))
        right_col.addWidget(hint_frame, 1)

        row.addLayout(right_col, 2)
        return row

    def _build_history(self):
        frame, layout = _make_card()

        header = QtWidgets.QHBoxLayout()
        header.addWidget(_section_label("Prediction History"))
        header.addStretch()

        self._clear_btn = QtWidgets.QPushButton("🗑  Clear")
        self._clear_btn.setObjectName("res_clear_btn")
        self._clear_btn.setEnabled(False)
        self._clear_btn.clicked.connect(self._clear_history)
        header.addWidget(self._clear_btn)

        self._export_btn = QtWidgets.QPushButton("⬇  Export CSV")
        self._export_btn.setObjectName("res_btn")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_csv)
        header.addWidget(self._export_btn)

        layout.addLayout(header)

        self._table = QtWidgets.QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["Time", "Predicted", "Confidence", "Actual", "Result"]
        )
        hh = self._table.horizontalHeader()
        hh.setStretchLastSection(False)
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self._table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(True)
        self._table.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        layout.addWidget(self._table, 1)

        return frame

    # ── actions ───────────────────────────────────────────────────────────────

    def _clear_history(self):
        self._history.clear()
        self._table.setRowCount(0)
        self._matrix.reset()
        self._acc_bars.reset()
        self._summary_lbl.setText("")
        self._export_btn.setEnabled(False)
        self._clear_btn.setEnabled(False)

    def _export_csv(self):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        fname = datetime.datetime.now().strftime("results_%Y%m%d_%H%M%S.csv")
        path = os.path.join(RESULTS_DIR, fname)
        try:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "timestamp", "predicted", "confidence",
                    "actual", "above_threshold"
                ])
                for dt, gesture, conf, thr, actual in self._history:
                    w.writerow([
                        dt.strftime("%Y-%m-%d %H:%M:%S"),
                        gesture,
                        f"{conf:.4f}",
                        actual if actual else "",
                        "yes" if conf >= thr else "no",
                    ])
            QtWidgets.QMessageBox.information(self, "Exported", f"Saved to:\n{path}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Export Failed", str(e))
