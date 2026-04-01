import os
import csv
import datetime
import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui


RESULTS_DIR = os.path.join(os.path.expanduser("~"), "SensDSv2_data", "results")

RESULTS_STYLE = """
    QWidget#results_root { background: #f0f2f5; }

    QLabel#res_heading {
        font-size: 16px;
        font-weight: bold;
        color: #1a3a5c;
    }
    QLabel#res_field_label {
        font-size: 12px;
        font-weight: bold;
        color: #555;
    }
    QLabel#res_subtext {
        font-size: 12px;
        color: #777;
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
        border-bottom: 1px solid #ddd;
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
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 5px 14px;
        font-size: 12px;
        color: #c0392b;
        font-weight: bold;
    }
    QPushButton#res_clear_btn:hover { background: #fff0f0; }
"""


def _card(layout_type=None):
    """White card with border and rounded corners."""
    frame = QtWidgets.QFrame()
    frame.setStyleSheet(
        "QFrame { background: white; border: 1px solid #ddd; border-radius: 8px; }"
    )
    inner = (layout_type or QtWidgets.QVBoxLayout)(frame)
    inner.setContentsMargins(14, 12, 14, 12)
    inner.setSpacing(8)
    return frame, inner


# ─── confusion matrix ────────────────────────────────────────────────────────

class ConfusionMatrixWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._classes: list = []
        self._matrix: np.ndarray = np.zeros((0, 0), dtype=int)
        self.setMinimumSize(200, 200)

    def set_classes(self, classes: list):
        n = len(classes)
        self._classes = classes
        self._matrix = np.zeros((n, n), dtype=int)
        self.update()

    def record(self, actual: str, predicted: str):
        if actual not in self._classes or predicted not in self._classes:
            return
        r = self._classes.index(actual)
        c = self._classes.index(predicted)
        self._matrix[r, c] += 1
        self.update()

    def reset(self):
        self._matrix = np.zeros_like(self._matrix)
        self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        n = len(self._classes)
        if n == 0:
            p.setPen(QtGui.QColor("#aaa"))
            p.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter,
                       "No predictions yet.")
            return

        w, h = self.width(), self.height()
        label_margin = 80   # space for axis labels
        top_margin = 60

        cell_w = (w - label_margin) / n
        cell_h = (h - top_margin) / n
        cell_sz = min(cell_w, cell_h)
        # recompute grid bounds centred
        grid_w = cell_sz * n
        grid_h = cell_sz * n
        x0 = label_margin + max(0, (w - label_margin - grid_w) / 2)
        y0 = top_margin

        max_val = self._matrix.max() if self._matrix.max() > 0 else 1
        navy = QtGui.QColor("#1a3a5c")

        lbl_font = QtGui.QFont()
        lbl_font.setPixelSize(max(9, int(cell_sz * 0.28)))
        val_font = QtGui.QFont()
        val_font.setPixelSize(max(8, int(cell_sz * 0.32)))
        val_font.setBold(True)
        axis_font = QtGui.QFont()
        axis_font.setPixelSize(11)
        axis_font.setBold(True)

        # ── cells ──
        for r in range(n):
            for c in range(n):
                val = self._matrix[r, c]
                t = val / max_val
                # white → navy gradient
                red   = int(255 + (26  - 255) * t)
                green = int(255 + (58  - 255) * t)
                blue  = int(255 + (92  - 255) * t)
                bg = QtGui.QColor(red, green, blue)
                cx = x0 + c * cell_sz
                cy = y0 + r * cell_sz
                p.fillRect(QtCore.QRectF(cx, cy, cell_sz - 1, cell_sz - 1), bg)
                # value text — white on dark cells, navy on light
                p.setFont(val_font)
                txt_color = QtCore.Qt.GlobalColor.white if t > 0.5 else navy
                p.setPen(txt_color)
                p.drawText(
                    QtCore.QRectF(cx, cy, cell_sz - 1, cell_sz - 1),
                    QtCore.Qt.AlignmentFlag.AlignCenter,
                    str(val),
                )

        # ── column labels (predicted) ──
        p.setFont(lbl_font)
        p.setPen(navy)
        for c, cls in enumerate(self._classes):
            cx = x0 + c * cell_sz + cell_sz / 2
            label = cls.replace("_", " ")
            p.save()
            p.translate(cx, y0 - 6)
            p.rotate(-35)
            p.drawText(QtCore.QRectF(-50, -14, 100, 14),
                       QtCore.Qt.AlignmentFlag.AlignRight, label)
            p.restore()

        # ── row labels (actual) ──
        for r, cls in enumerate(self._classes):
            cy = y0 + r * cell_sz + cell_sz / 2
            label = cls.replace("_", " ")
            p.setFont(lbl_font)
            p.setPen(navy)
            p.drawText(
                QtCore.QRectF(0, cy - cell_sz / 2, label_margin - 6, cell_sz),
                QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter,
                label,
            )

        # ── axis titles ──
        p.setFont(axis_font)
        p.setPen(QtGui.QColor("#555"))
        # "Predicted" across the top
        p.drawText(
            QtCore.QRectF(x0, 2, grid_w, 16),
            QtCore.Qt.AlignmentFlag.AlignCenter, "Predicted →"
        )
        # "Actual" rotated on the left
        p.save()
        p.translate(10, y0 + grid_h / 2)
        p.rotate(-90)
        p.drawText(QtCore.QRectF(-40, -10, 80, 20),
                   QtCore.Qt.AlignmentFlag.AlignCenter, "Actual →")
        p.restore()


# ─── per-class accuracy bars ─────────────────────────────────────────────────

class AccuracyBarsWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._classes: list = []
        self._correct: dict = {}
        self._total: dict = {}
        self.setMinimumHeight(80)

    def set_classes(self, classes: list):
        self._classes = classes
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
        if not self._classes:
            p = QtGui.QPainter(self)
            p.setPen(QtGui.QColor("#aaa"))
            p.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter,
                       "No predictions yet.")
            return

        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        n = len(self._classes)
        navy = QtGui.QColor("#1a3a5c")

        label_w = 100
        pct_w   = 46
        margin  = 6
        bar_area_w = w - margin * 2 - label_w - pct_w
        total_spacing = margin * (n + 1)
        bar_h = max(10, (h - total_spacing) // n)

        font = QtGui.QFont()
        font.setPixelSize(12)
        p.setFont(font)

        for i, cls in enumerate(self._classes):
            y = margin + i * (bar_h + margin)
            tot = self._total.get(cls, 0)
            cor = self._correct.get(cls, 0)
            acc = cor / tot if tot > 0 else 0.0

            bar_px = int(bar_area_w * acc)
            bar_color = navy if acc >= 0.7 else (
                QtGui.QColor("#e67e22") if acc >= 0.4 else QtGui.QColor("#c0392b")
            )

            # Background track
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(QtGui.QColor("#e0e0e0"))
            p.drawRoundedRect(margin + label_w, y, bar_area_w, bar_h, 3, 3)

            if bar_px > 0:
                p.setBrush(bar_color)
                p.drawRoundedRect(margin + label_w, y, bar_px, bar_h, 3, 3)

            # Label
            p.setPen(QtGui.QColor("#333"))
            p.drawText(
                QtCore.QRectF(margin, y, label_w - 4, bar_h),
                QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter,
                cls.replace("_", " "),
            )

            # Percentage and count
            pct_txt = f"{acc:.0%}" if tot > 0 else "—"
            p.setPen(QtGui.QColor("#555"))
            p.drawText(
                QtCore.QRectF(margin + label_w + bar_area_w + 4, y, pct_w - 4, bar_h),
                QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
                pct_txt,
            )


# ─── main tab ────────────────────────────────────────────────────────────────

class ResultsTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("results_root")
        self.setStyleSheet(RESULTS_STYLE)

        self._classes: list = []
        self._history: list = []   # list of (dt, gesture, confidence, threshold)

        self._setup_ui()

    # ── public API ────────────────────────────────────────────────────────────

    def set_model_info(self, model_name: str, classes: list):
        self._classes = list(classes)
        self._model_name_lbl.setText(f"Model:  {model_name}")
        self._classes_lbl.setText("  ·  ".join(classes))
        self._matrix.set_classes(classes)
        self._acc_bars.set_classes(classes)
        self._history.clear()
        self._table.setRowCount(0)
        self._export_btn.setEnabled(False)
        self._clear_btn.setEnabled(False)

    def add_prediction(self, gesture: str, confidence: float, threshold: float):
        dt = datetime.datetime.now()
        self._history.append((dt, gesture, confidence, threshold))

        # Update confusion matrix and accuracy bars.
        # We don't know ground truth in live use, so record gesture vs gesture
        # (diagonal entries — every prediction is its own "actual" row).
        # This shows how often each class is predicted, and per-class counts.
        self._matrix.record(gesture, gesture)
        self._acc_bars.record(gesture, gesture)

        # Table row
        row = self._table.rowCount()
        self._table.insertRow(row)

        ts_item = QtWidgets.QTableWidgetItem(dt.strftime("%H:%M:%S"))
        ts_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        gesture_item = QtWidgets.QTableWidgetItem(gesture.replace("_", " "))
        gesture_item.setForeground(QtGui.QColor("#1a3a5c"))
        gesture_item.setFont(self._bold_font())

        conf_item = QtWidgets.QTableWidgetItem(f"{confidence:.1%}")
        conf_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        ok = confidence >= threshold
        flag_item = QtWidgets.QTableWidgetItem("✓" if ok else "⚠")
        flag_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        flag_item.setForeground(
            QtGui.QColor("#27ae60") if ok else QtGui.QColor("#e67e22")
        )

        for col, item in enumerate([ts_item, gesture_item, conf_item, flag_item]):
            self._table.setItem(row, col, item)

        self._table.scrollToBottom()
        self._export_btn.setEnabled(True)
        self._clear_btn.setEnabled(True)

        # Update summary label
        n = len(self._history)
        above = sum(1 for _, _, c, t in self._history if c >= t)
        self._summary_lbl.setText(
            f"{n} prediction{'s' if n != 1 else ''}  —  "
            f"{above} above threshold  ({above/n:.0%})"
        )

    # ── UI construction ───────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        root.addWidget(self._build_model_bar())
        root.addLayout(self._build_middle_row(), 3)
        root.addWidget(self._build_history_section(), 2)

    def _build_model_bar(self):
        frame, layout = _card(QtWidgets.QHBoxLayout)
        layout.setContentsMargins(14, 10, 14, 10)

        lbl = QtWidgets.QLabel("Results")
        lbl.setObjectName("res_heading")
        layout.addWidget(lbl)

        layout.addSpacing(20)

        self._model_name_lbl = QtWidgets.QLabel("No model loaded.")
        self._model_name_lbl.setObjectName("res_field_label")
        layout.addWidget(self._model_name_lbl)

        layout.addSpacing(12)

        self._classes_lbl = QtWidgets.QLabel("")
        self._classes_lbl.setObjectName("res_subtext")
        self._classes_lbl.setWordWrap(False)
        layout.addWidget(self._classes_lbl, 1)

        self._summary_lbl = QtWidgets.QLabel("")
        self._summary_lbl.setObjectName("res_subtext")
        self._summary_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight |
                                       QtCore.Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._summary_lbl)

        frame.setFixedHeight(52)
        return frame

    def _build_middle_row(self):
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(10)

        # Left — confusion matrix
        cf_frame, cf_layout = _card()
        cf_lbl = QtWidgets.QLabel("Confusion Matrix")
        cf_lbl.setObjectName("res_field_label")
        cf_layout.addWidget(cf_lbl)

        note = QtWidgets.QLabel(
            "Row = actual gesture  ·  Column = predicted gesture"
        )
        note.setObjectName("res_subtext")
        note.setStyleSheet("font-size: 11px; color: #999; border: none;")
        cf_layout.addWidget(note)

        self._matrix = ConfusionMatrixWidget()
        cf_layout.addWidget(self._matrix, 1)

        row.addWidget(cf_frame, 3)

        # Right — per-class accuracy
        acc_frame, acc_layout = _card()
        acc_lbl = QtWidgets.QLabel("Per-Class Accuracy")
        acc_lbl.setObjectName("res_field_label")
        acc_layout.addWidget(acc_lbl)

        acc_note = QtWidgets.QLabel(
            "Green ≥ 70%  ·  Orange ≥ 40%  ·  Red < 40%"
        )
        acc_note.setStyleSheet("font-size: 11px; color: #999; border: none;")
        acc_layout.addWidget(acc_note)

        self._acc_bars = AccuracyBarsWidget()
        acc_layout.addWidget(self._acc_bars, 1)

        row.addWidget(acc_frame, 2)

        return row

    def _build_history_section(self):
        frame, layout = _card()

        header_row = QtWidgets.QHBoxLayout()

        hist_lbl = QtWidgets.QLabel("Prediction History")
        hist_lbl.setObjectName("res_field_label")
        header_row.addWidget(hist_lbl)
        header_row.addStretch()

        self._clear_btn = QtWidgets.QPushButton("🗑  Clear History")
        self._clear_btn.setObjectName("res_clear_btn")
        self._clear_btn.setEnabled(False)
        self._clear_btn.clicked.connect(self._clear_history)
        header_row.addWidget(self._clear_btn)

        self._export_btn = QtWidgets.QPushButton("⬇  Export CSV")
        self._export_btn.setObjectName("res_btn")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_csv)
        header_row.addWidget(self._export_btn)

        layout.addLayout(header_row)

        self._table = QtWidgets.QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(
            ["Time", "Gesture", "Confidence", "Above Threshold"]
        )
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self._table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.horizontalHeader().setSectionResizeMode(
            2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.horizontalHeader().setSectionResizeMode(
            3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
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
                w.writerow(["timestamp", "gesture", "confidence", "above_threshold"])
                for dt, gesture, conf, thr in self._history:
                    w.writerow([
                        dt.strftime("%Y-%m-%d %H:%M:%S"),
                        gesture,
                        f"{conf:.4f}",
                        "yes" if conf >= thr else "no",
                    ])
            QtWidgets.QMessageBox.information(
                self, "Exported", f"Saved to:\n{path}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Export Failed", str(e))

    def _bold_font(self):
        f = QtGui.QFont()
        f.setBold(True)
        return f
