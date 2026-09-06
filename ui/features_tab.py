"""
ui/features_tab.py

Interactive scatterplot of physical radar features — distance, speed,
acceleration, Doppler spread — with CSV export for CODAP.

Pick any feature for X and any for Y and the plot redraws from cached values,
so students can hunt for the pair that separates their gestures. Export writes
the same numbers shown on screen, in two shapes: one row per gesture sample,
and one row per radar frame.
"""

import os

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from core import physical_features as PF
from ui import (HintCard, _scrollable_left, app_colors, is_dark_mode,
                zoom_button_row)

CLASS_COLORS_LIGHT = [
    "#e74c3c", "#2980b9", "#27ae60", "#f39c12",
    "#8e44ad", "#16a085", "#d35400", "#2c3e50",
]
CLASS_COLORS_DARK = [
    "#ff6b6b", "#5dade2", "#2ecc71", "#f5b041",
    "#bb8fce", "#48c9b0", "#f0932b", "#aab7b8",
]


def class_colors():
    return CLASS_COLORS_DARK if is_dark_mode() else CLASS_COLORS_LIGHT


def _style(c: dict) -> str:
    return f"""
    QWidget#feat_root {{ background: {c['bg']}; }}
    QWidget#left_panel {{
        background: {c['panel']};
        border-right: 1px solid {c['border']};
    }}
    QWidget#detail_panel {{
        background: {c['panel']};
        border-left: 1px solid {c['border']};
    }}
    QLabel#detail_meta {{ font-size: 12px; color: {c['text']}; }}
    QLabel#feat_name {{ font-size: 11px; color: {c['subtext']}; }}
    QLabel#feat_val  {{ font-size: 12px; color: {c['text']}; }}
    QLabel#img_frame {{
        background: {c['input_bg']};
        border: 1px solid {c['input_border']};
        border-radius: 5px;
        color: {c['faint']};
        font-size: 11px;
    }}
    QLabel#heading {{
        font-size: 16px; font-weight: bold; color: {c['accent']};
    }}
    QLabel#field_label {{
        font-size: 12px; font-weight: bold; color: {c['subtext']};
    }}
    QLabel#status_ok  {{ font-size: 12px; color: #27ae60; font-weight: bold; }}
    QLabel#status_err {{ font-size: 12px; color: #e74c3c; font-weight: bold; }}
    QLabel#note {{ font-size: 11px; color: {c['subtext']}; }}
    QLabel#hintmsg {{ font-size: 11px; color: {c['faint']}; }}
    QComboBox {{
        border: 1px solid {c['input_border']}; border-radius: 5px;
        padding: 5px 8px; font-size: 12px;
        background: {c['input_bg']}; color: {c['text']}; max-height: 30px;
    }}
    QComboBox::drop-down {{ border: none; }}
    QComboBox QAbstractItemView {{
        background: {c['panel']}; color: {c['text']};
        border: 1px solid {c['border']};
        selection-background-color: {c['accent']}; selection-color: white;
    }}
    QPushButton#primary_btn {{
        background-color: {c['accent']}; color: white; border: none;
        border-radius: 6px; padding: 10px; font-size: 13px; font-weight: bold;
    }}
    QPushButton#primary_btn:hover {{ background-color: #245080; }}
    QPushButton#primary_btn:disabled {{
        background-color: {c['progress_bg']}; color: {c['faint']};
    }}
    QPushButton#minor_btn {{
        background: {c['panel']}; border: 1px solid {c['input_border']};
        border-radius: 5px; padding: 6px 10px; font-size: 12px;
        color: {c['accent']}; font-weight: bold;
    }}
    QPushButton#minor_btn:hover {{ background: {c['tab_hover']}; }}
    QPushButton#minor_btn:disabled {{ color: {c['faint']}; }}
    QProgressBar {{
        border: none; border-radius: 4px; background: {c['progress_bg']};
        max-height: 8px; text-align: center; color: transparent;
    }}
    QProgressBar::chunk {{
        background-color: {c['accent']}; border-radius: 4px;
    }}
    """


class FeatureWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int, int, str)
    finished = QtCore.pyqtSignal(list)
    error = QtCore.pyqtSignal(str)

    def __init__(self, samples):
        super().__init__()
        self._samples = list(samples)
        self._running = False

    def stop(self):
        self._running = False

    @QtCore.pyqtSlot()
    def run(self):
        self._running = True
        out = []
        try:
            total = len(self._samples)
            for i, s in enumerate(self._samples, 1):
                if not self._running:
                    return
                self.progress.emit(i, total, s["name"])
                try:
                    cube = np.load(s["path"])
                except Exception as e:
                    self.error.emit(f"Could not read {s['name']}:\n{e}")
                    return
                if cube.ndim != 4:
                    continue
                summary, series = PF.extract(cube)
                out.append({
                    "student": s["student"],
                    "gesture": s["gesture"],
                    "name": s["name"],
                    "path": s["path"],
                    # Resolved here rather than on click so the disk check
                    # stays off the UI thread.
                    "png": PF.png_for_sample(s["path"]),
                    "summary": summary,
                    "series": series,
                })
            if not out:
                self.error.emit("No usable raw samples found.")
                return
            self.finished.emit(out)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class FeaturesTab(QtWidgets.QWidget):
    # Detail column width, and the box the sample's PNG is drawn into. The
    # PNG is saved 400x300, so 264x198 shows it at native aspect with no crop.
    DETAIL_W = 300
    IMG_W = 264
    IMG_H = 198

    def __init__(self):
        super().__init__()
        self._c = app_colors()
        self.setObjectName("feat_root")
        self.setStyleSheet(_style(self._c))

        dark = is_dark_mode()
        self._plot_bg = "#1a1a2e" if dark else "#ffffff"
        self._axis_pen = pg.mkPen("#888899" if dark else "#999999")
        self._axis_color = "#cccccc" if dark else "#555555"
        self._title_color = "#e6e6e6" if dark else "#1a3a5c"
        self._legend_bg = (30, 30, 46, 235) if dark else (255, 255, 255, 235)
        self._legend_border = "#4a4a5a" if dark else "#cccccc"
        self._legend_text = "#e6e6e6" if dark else "#333333"

        self._worker = None
        self._thread = None
        self._samples = []
        self._records = []
        self._selected = None
        self._highlight = None

        self._setup_ui()
        self.refresh()

    # ── layout ───────────────────────────────────────────────────────────────

    def _setup_ui(self):
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_left())
        outer.addWidget(self._build_right(), 1)
        outer.addWidget(self._build_detail())

    def _build_left(self):
        panel = QtWidgets.QWidget()
        panel.setObjectName("left_panel")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(8)

        heading = QtWidgets.QLabel("Physical Features")
        heading.setObjectName("heading")
        layout.addWidget(heading)
        layout.addWidget(self._divider())

        layout.addWidget(self._lbl("Dataset"))
        self._status = QtWidgets.QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._refresh_btn = QtWidgets.QPushButton("↻  Refresh")
        self._refresh_btn.setObjectName("minor_btn")
        self._refresh_btn.clicked.connect(self.refresh)
        layout.addWidget(self._refresh_btn)

        self._extract_btn = QtWidgets.QPushButton("▶  Extract Features")
        self._extract_btn.setObjectName("primary_btn")
        self._extract_btn.clicked.connect(self._start)
        layout.addWidget(self._extract_btn)

        self._progress = QtWidgets.QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        layout.addWidget(self._divider())

        layout.addWidget(self._lbl("Plot Axes"))
        self._x_combo = QtWidgets.QComboBox()
        self._y_combo = QtWidgets.QComboBox()
        for key, label, unit in PF.SUMMARY_FEATURES:
            text = f"{label} ({unit})" if unit else label
            self._x_combo.addItem(text, key)
            self._y_combo.addItem(text, key)
        self._x_combo.setCurrentIndex(
            PF.SUMMARY_KEYS.index("range_travel_m"))
        self._y_combo.setCurrentIndex(
            PF.SUMMARY_KEYS.index("radial_speed_max_ms"))
        self._x_combo.currentIndexChanged.connect(self._replot)
        self._y_combo.currentIndexChanged.connect(self._replot)

        layout.addWidget(self._small("X axis"))
        layout.addWidget(self._x_combo)
        layout.addWidget(self._small("Y axis"))
        layout.addWidget(self._y_combo)

        layout.addWidget(self._small("Color by"))
        self._color_combo = QtWidgets.QComboBox()
        self._color_combo.addItem("Gesture", "gesture")
        self._color_combo.addItem("Student", "student")
        self._color_combo.currentIndexChanged.connect(self._replot)
        layout.addWidget(self._color_combo)

        layout.addWidget(self._divider())

        layout.addWidget(self._lbl("Export for CODAP"))
        note = QtWidgets.QLabel(
            "Summary: one line for each gesture you recorded, holding its "
            "four measurements. Use this to compare gestures.\n\n"
            "Frames: one line for every picture the radar took, ten per "
            "second. Use this to see how distance and speed changed while "
            "a single gesture was happening."
        )
        note.setObjectName("note")
        note.setWordWrap(True)
        layout.addWidget(note)

        self._export_summary_btn = QtWidgets.QPushButton("⤓  Summary CSV")
        self._export_summary_btn.setObjectName("minor_btn")
        self._export_summary_btn.clicked.connect(self._export_summary)
        self._export_summary_btn.setEnabled(False)
        layout.addWidget(self._export_summary_btn)

        self._export_frames_btn = QtWidgets.QPushButton("⤓  Frames CSV")
        self._export_frames_btn.setObjectName("minor_btn")
        self._export_frames_btn.clicked.connect(self._export_frames)
        self._export_frames_btn.setEnabled(False)
        layout.addWidget(self._export_frames_btn)

        layout.addStretch()

        layout.addWidget(HintCard([
            "Every number here has real units (meters and m/s), so you can "
            "sanity check them against what your hand actually did.",
            "A push moves toward the radar, so 'Distance traveled' is large. "
            "A swipe moves sideways, so it stays small.",
            "The radar measures motion toward and away from itself. Sideways "
            "movement barely registers, which is why a swipe looks slow here.",
            "Distance against speed separates the gestures better than two "
            "distance features or two speed features.",
            "Click a point to see the spectrogram it came from. A point sitting "
            "far from the rest of its group is usually a recording that went "
            "wrong, and the picture will show you how.",
        ], c=self._c))

        self._msg = QtWidgets.QLabel("")
        self._msg.setObjectName("hintmsg")
        self._msg.setWordWrap(True)
        layout.addWidget(self._msg)

        return _scrollable_left(panel, width=300)

    def _build_right(self):
        panel = QtWidgets.QWidget()
        panel.setStyleSheet(f"background: {self._c['bg']};")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        self._plot = pg.PlotWidget()
        self._plot.setBackground(self._plot_bg)
        self._plot.setTitle("Physical Feature Scatterplot",
                            color=self._title_color, size="11pt", bold=True)
        for ax in ("left", "bottom"):
            axis = self._plot.getAxis(ax)
            axis.setPen(self._axis_pen)
            axis.setTextPen(pg.mkPen(self._axis_color))
            # Units are already in the label text; without this pyqtgraph
            # rescales small values and appends its own "(x0.001)".
            axis.enableAutoSIPrefix(False)
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        # An unbacked legend sits directly on top of the points and is hard to
        # read; give it a solid panel and a border so it reads as an overlay.
        self._legend = self._plot.addLegend(
            offset=(-12, 12),
            labelTextColor=self._legend_text,
            labelTextSize="10pt",
            brush=pg.mkBrush(self._legend_bg),
            pen=pg.mkPen(self._legend_border),
        )
        layout.addWidget(self._plot, 1)
        layout.addWidget(zoom_button_row(self._plot, self._c))

        self._caption = QtWidgets.QLabel(
            "Extract features to plot them."
        )
        self._caption.setObjectName("note")
        self._caption.setWordWrap(True)
        layout.addWidget(self._caption)
        return panel

    def _build_detail(self):
        """Right-hand column: the spectrogram and numbers behind one point."""
        panel = QtWidgets.QWidget()
        panel.setObjectName("detail_panel")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(8)

        heading = QtWidgets.QLabel("Selected Sample")
        heading.setObjectName("heading")
        layout.addWidget(heading)
        layout.addWidget(self._divider())

        self._detail_hint = QtWidgets.QLabel(
            "Click any point on the plot to see the recording it came from — "
            "its spectrogram picture and all of its measurements."
        )
        self._detail_hint.setObjectName("note")
        self._detail_hint.setWordWrap(True)
        layout.addWidget(self._detail_hint)

        # Everything below stays hidden until a point is picked, so the panel
        # is not a column of empty dashes on first open.
        self._detail_body = QtWidgets.QWidget()
        body = QtWidgets.QVBoxLayout(self._detail_body)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(8)

        self._img_label = QtWidgets.QLabel()
        self._img_label.setObjectName("img_frame")
        self._img_label.setFixedSize(self.IMG_W, self.IMG_H)
        self._img_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._img_label.setWordWrap(True)
        body.addWidget(self._img_label)

        self._img_caption = QtWidgets.QLabel("")
        self._img_caption.setObjectName("hintmsg")
        self._img_caption.setWordWrap(True)
        body.addWidget(self._img_caption)

        self._detail_meta = QtWidgets.QLabel("")
        self._detail_meta.setObjectName("detail_meta")
        self._detail_meta.setWordWrap(True)
        body.addWidget(self._detail_meta)

        body.addWidget(self._divider())
        body.addWidget(self._lbl("Measurements"))

        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        self._feat_rows = {}
        for row, (key, label, unit) in enumerate(PF.SUMMARY_FEATURES):
            name = QtWidgets.QLabel(label)
            name.setObjectName("feat_name")
            name.setWordWrap(True)
            value = QtWidgets.QLabel("—")
            value.setObjectName("feat_val")
            value.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight
                | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
            grid.addWidget(name, row, 0)
            grid.addWidget(value, row, 1)
            self._feat_rows[key] = (name, value, unit)
        grid.setColumnStretch(0, 1)
        body.addLayout(grid)

        self._axis_note = QtWidgets.QLabel("")
        self._axis_note.setObjectName("hintmsg")
        self._axis_note.setWordWrap(True)
        body.addWidget(self._axis_note)

        self._open_sample_btn = QtWidgets.QPushButton("🗀  Show in Folder")
        self._open_sample_btn.setObjectName("minor_btn")
        self._open_sample_btn.clicked.connect(self._open_selected_folder)
        body.addWidget(self._open_sample_btn)

        self._detail_body.setVisible(False)
        layout.addWidget(self._detail_body)
        layout.addStretch()

        return _scrollable_left(panel, width=self.DETAIL_W)

    def _lbl(self, text):
        w = QtWidgets.QLabel(text)
        w.setObjectName("field_label")
        return w

    def _small(self, text):
        w = QtWidgets.QLabel(text)
        w.setObjectName("hintmsg")
        return w

    def _divider(self):
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setStyleSheet(f"color: {self._c['divider']}; margin: 2px 0;")
        return line

    # ── dataset ──────────────────────────────────────────────────────────────

    def refresh(self):
        self._samples = PF.scan_samples()
        n = len(self._samples)
        gestures = sorted({s["gesture"] for s in self._samples})
        students = sorted({s["student"] for s in self._samples})

        if n:
            self._status.setObjectName("status_ok")
            self._status.setText(
                f"✓  {n} raw samples\n"
                f"    {len(gestures)} gestures: {', '.join(gestures)}\n"
                f"    {len(students)} students: {', '.join(students)}"
            )
            self._msg.setText("")
        else:
            self._status.setObjectName("status_err")
            self._status.setText("✗  No raw samples found")
            self._msg.setText(
                "Physical features need the raw radar cube "
                "(sample_NNN_raw.npy), which only newer captures save. "
                "Collect fresh samples to use this tab."
            )
        self._status.style().unpolish(self._status)
        self._status.style().polish(self._status)
        self._extract_btn.setEnabled(n > 0)

    # ── extraction ───────────────────────────────────────────────────────────

    def _start(self):
        self._extract_btn.setEnabled(False)
        self._refresh_btn.setEnabled(False)
        self._progress.setMaximum(len(self._samples))
        self._progress.setValue(0)
        self._progress.setVisible(True)

        self._worker = FeatureWorker(self._samples)
        self._thread = QtCore.QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._thread.start()

    def _on_progress(self, i, total, name):
        self._progress.setValue(i)
        self._msg.setText(f"Extracting {i}/{total}: {name}")

    def _on_finished(self, records):
        self._cleanup_thread()
        self._records = records
        # The old selection points at a record object that no longer exists.
        self._clear_selection()
        self._progress.setVisible(False)
        self._extract_btn.setEnabled(True)
        self._refresh_btn.setEnabled(True)
        self._export_summary_btn.setEnabled(True)
        self._export_frames_btn.setEnabled(True)
        frames = sum(len(r["series"]["time_s"]) for r in records)
        self._msg.setText(
            f"Extracted {len(records)} samples ({frames} frames total)."
        )
        self._replot()

    def _on_error(self, msg):
        self._cleanup_thread()
        self._progress.setVisible(False)
        self._extract_btn.setEnabled(True)
        self._refresh_btn.setEnabled(True)
        self._msg.setText(f"✗  {msg.splitlines()[0]}")

    def _cleanup_thread(self):
        if self._thread:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None

    def stop_if_running(self):
        if self._worker is not None:
            self._worker.stop()
        self._cleanup_thread()

    # ── plotting ─────────────────────────────────────────────────────────────

    def _replot(self):
        if not self._records:
            return
        xk = self._x_combo.currentData()
        yk = self._y_combo.currentData()
        group_key = self._color_combo.currentData()

        self._plot.clear()
        self._legend.clear()
        # clear() already dropped the ring; forget it so it is not removed twice.
        self._highlight = None

        groups = sorted({r[group_key] for r in self._records})
        palette = class_colors()
        for i, g in enumerate(groups):
            rows = [r for r in self._records if r[group_key] == g]
            xs = [r["summary"][xk] for r in rows]
            ys = [r["summary"][yk] for r in rows]
            color = palette[i % len(palette)]
            # Each spot carries its own record, so a click can find its way
            # back to the sample's spectrogram and measurements.
            try:
                item = pg.ScatterPlotItem(
                    x=xs, y=ys, size=12,
                    pen=pg.mkPen(self._plot_bg, width=1),
                    brush=pg.mkBrush(color),
                    data=rows, hoverable=True,
                    tip=self._point_tip,
                )
            except TypeError:
                # Older pyqtgraph without hoverable/tip support.
                item = pg.ScatterPlotItem(
                    x=xs, y=ys, size=12,
                    pen=pg.mkPen(self._plot_bg, width=1),
                    brush=pg.mkBrush(color),
                    data=rows,
                )
            item.sigClicked.connect(self._on_point_clicked)
            self._plot.addItem(item)
            self._legend.addItem(item, f"{g}  ({len(rows)})")

        self._plot.setLabel("bottom", PF.feature_label(xk),
                            color=self._axis_color)
        self._plot.setLabel("left", PF.feature_label(yk),
                            color=self._axis_color)
        self._plot.enableAutoRange()
        self._caption.setText(
            f"{len(self._records)} samples  ·  X = {PF.feature_label(xk)}  ·  "
            f"Y = {PF.feature_label(yk)}  ·  colored by {group_key}  ·  "
            f"click a point to see its spectrogram"
        )

        # Swapping axes moves the selected sample and changes which two of its
        # measurements are the ones on screen, so redraw the panel too.
        if self._selected is not None:
            self._show_record(self._selected)

    def _point_tip(self, x, y, data):
        if not isinstance(data, dict):
            return ""
        xk = self._x_combo.currentData()
        yk = self._y_combo.currentData()
        return (
            f"{data['student']} / {data['gesture']}\n{data['name']}\n"
            f"{PF.feature_label(xk)}: {x:.4g}\n"
            f"{PF.feature_label(yk)}: {y:.4g}\n"
            f"Click to open this sample"
        )

    # ── point selection ──────────────────────────────────────────────────────

    def _on_point_clicked(self, *args):
        # sigClicked is (item, points, event) on current pyqtgraph and
        # (item, points) on older builds; points is second either way.
        points = args[1] if len(args) > 1 else ()
        if len(points) == 0:
            return
        record = points[0].data()
        if isinstance(record, dict):
            self._show_record(record)

    def _clear_selection(self):
        self._selected = None
        self._highlight = None
        self._detail_body.setVisible(False)
        self._detail_hint.setVisible(True)

    def _show_record(self, record):
        self._selected = record
        self._detail_hint.setVisible(False)
        self._detail_body.setVisible(True)

        png = record.get("png")
        pixmap = QtGui.QPixmap(png) if png else QtGui.QPixmap()
        if pixmap.isNull():
            self._img_label.setPixmap(QtGui.QPixmap())
            self._img_label.setText(
                "No spectrogram image was saved next to this recording."
            )
            self._img_caption.setText("")
        else:
            self._img_label.setText("")
            self._img_label.setPixmap(pixmap.scaled(
                self.IMG_W, self.IMG_H,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            ))
            self._img_caption.setText(
                "Time runs left to right. Velocity runs bottom to top with "
                "zero in the middle, so motion toward the radar sits above "
                "the center line. This is the picture the model is trained on."
            )

        n_frames = len(record["series"]["time_s"])
        self._detail_meta.setText(
            f"<b>{record['gesture']}</b> &nbsp;·&nbsp; {record['student']}<br>"
            f"{record['name']}<br>"
            f"{n_frames} frames &nbsp;·&nbsp; {n_frames * PF.FRAME_S:.1f} s"
        )

        xk = self._x_combo.currentData()
        yk = self._y_combo.currentData()
        for key, (name, value, unit) in self._feat_rows.items():
            v = record["summary"][key]
            value.setText(f"{v:.3f} {unit}".strip())
            on_axis = key in (xk, yk)
            # Inline styles only where a row is highlighted; an empty sheet
            # falls back to the #feat_name / #feat_val rules.
            value.setStyleSheet(
                f"color: {self._c['accent']}; font-weight: bold;"
                if on_axis else ""
            )
            name.setStyleSheet(
                f"color: {self._c['text']}; font-weight: bold;"
                if on_axis else ""
            )
        self._axis_note.setText(
            "The two highlighted rows are what the plot is showing."
        )

        self._mark_selected()

    def _mark_selected(self):
        """Ring the selected point so it is findable among its neighbors."""
        if self._highlight is not None:
            self._plot.removeItem(self._highlight)
            self._highlight = None
        if self._selected is None or not self._records:
            return
        xk = self._x_combo.currentData()
        yk = self._y_combo.currentData()
        self._highlight = pg.ScatterPlotItem(
            x=[self._selected["summary"][xk]],
            y=[self._selected["summary"][yk]],
            size=24, symbol="o",
            pen=pg.mkPen(self._c["accent"], width=3),
            brush=None,
        )
        self._highlight.setZValue(20)
        self._plot.addItem(self._highlight)

    def _open_selected_folder(self):
        if self._selected is None:
            return
        folder = os.path.dirname(self._selected["path"])
        if not os.path.isdir(folder):
            self._msg.setText("✗  That folder no longer exists.")
            return
        import subprocess
        import sys
        if sys.platform == "darwin":
            subprocess.Popen(["open", folder])
        elif sys.platform == "win32":
            subprocess.Popen(["explorer", folder])
        else:
            subprocess.Popen(["xdg-open", folder])

    # ── export ───────────────────────────────────────────────────────────────

    def _export_dir(self):
        d = os.path.join(PF.DATA_ROOT, "exports")
        os.makedirs(d, exist_ok=True)
        return d

    def _ask_path(self, default_name):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save CSV",
            os.path.join(self._export_dir(), default_name),
            "CSV files (*.csv)",
        )
        return path

    def _export_summary(self):
        path = self._ask_path("radar_features_summary.csv")
        if not path:
            return
        try:
            n = PF.write_summary_csv(self._records, path)
            self._msg.setText(f"Saved {n} rows to {os.path.basename(path)}")
        except Exception as e:
            self._msg.setText(f"✗  Export failed: {e}")

    def _export_frames(self):
        path = self._ask_path("radar_features_frames.csv")
        if not path:
            return
        try:
            n = PF.write_frames_csv(self._records, path)
            self._msg.setText(f"Saved {n} rows to {os.path.basename(path)}")
        except Exception as e:
            self._msg.setText(f"✗  Export failed: {e}")
