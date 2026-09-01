"""
ui/pca_tab.py

Compares two representations of the same radar captures: the Doppler-domain
spectrogram against the range-domain range-FFT profile. Both are projected to
2D with PCA and scored with a silhouette coefficient, so a student can see
which representation separates the gesture classes better.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from core import pca_analysis as PA
from ui import (HintCard, _scrollable_left, app_colors, is_dark_mode,
                zoom_button_row)

MIN_CLASSES = 2
MIN_SAMPLES = 6

# Fixed palette so a gesture keeps the same color in both plots.
# Two variants: the light one is too dark to read on a dark plot background
# (notably the navy), the dark one washes out on white.
CLASS_COLORS_LIGHT = [
    "#e74c3c", "#2980b9", "#27ae60", "#f39c12",
    "#8e44ad", "#16a085", "#d35400", "#2c3e50",
]
CLASS_COLORS_DARK = [
    "#ff6b6b", "#5dade2", "#2ecc71", "#f5b041",
    "#bb8fce", "#48c9b0", "#f0932b", "#aab7b8",
]


def class_colors() -> list:
    return CLASS_COLORS_DARK if is_dark_mode() else CLASS_COLORS_LIGHT


def _pca_style(c: dict) -> str:
    return f"""
    QWidget#pca_root {{ background: {c['bg']}; }}
    QWidget#left_panel {{
        background: {c['panel']};
        border-right: 1px solid {c['border']};
    }}
    QLabel#heading {{
        font-size: 16px;
        font-weight: bold;
        color: {c['accent']};
    }}
    QLabel#field_label {{
        font-size: 12px;
        font-weight: bold;
        color: {c['subtext']};
    }}
    QLabel#status_ok   {{ font-size: 12px; color: #27ae60; font-weight: bold; }}
    QLabel#status_warn {{ font-size: 12px; color: #e67e22; font-weight: bold; }}
    QLabel#status_err  {{ font-size: 12px; color: #e74c3c; font-weight: bold; }}
    QLabel#note {{
        font-size: 11px;
        color: {c['subtext']};
    }}
    QPushButton#pca_btn {{
        background-color: {c['accent']};
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px;
        font-size: 13px;
        font-weight: bold;
    }}
    QPushButton#pca_btn:hover {{ background-color: #245080; }}
    QPushButton#pca_btn:disabled {{
        background-color: {c['progress_bg']};
        color: {c['faint']};
    }}
    QPushButton#refresh_btn {{
        background: {c['panel']};
        border: 1px solid {c['input_border']};
        border-radius: 5px;
        padding: 6px 12px;
        font-size: 12px;
        color: {c['accent']};
        font-weight: bold;
    }}
    QPushButton#refresh_btn:hover {{ background: {c['tab_hover']}; }}
    QProgressBar {{
        border: none;
        border-radius: 4px;
        background: {c['progress_bg']};
        max-height: 8px;
        text-align: center;
        color: transparent;
    }}
    QProgressBar::chunk {{ background-color: {c['accent']}; border-radius: 4px; }}
    QLabel#caption {{
        font-size: 12px;
        color: {c['subtext']};
        font-family: monospace;
    }}
    QLabel#measures {{
        font-size: 11px;
        color: {c['faint']};
    }}
    QLabel#summary {{
        font-size: 13px;
        font-weight: bold;
        color: {c['accent']};
        background: {c['panel']};
        border: 1px solid {c['border']};
        border-radius: 6px;
        padding: 10px;
    }}
    """


class PcaWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int, int, str)
    finished = QtCore.pyqtSignal(np.ndarray, np.ndarray, float,
                                 np.ndarray, np.ndarray, float, list)
    error = QtCore.pyqtSignal(str)

    def __init__(self, paths, labels):
        super().__init__()
        self._paths = list(paths)
        self._labels = list(labels)
        self._running = False

    def stop(self):
        self._running = False

    @QtCore.pyqtSlot()
    def run(self):
        self._running = True
        try:
            spec_feats, rfft_feats, kept = [], [], []
            total = len(self._paths)

            for i, (path, label) in enumerate(zip(self._paths, self._labels), 1):
                if not self._running:
                    return
                import os
                self.progress.emit(i, total, os.path.basename(path))
                try:
                    cube = np.load(path)
                except Exception as e:
                    self.error.emit(f"Could not read {path}:\n{e}")
                    return
                if cube.ndim != 4:
                    continue
                spec_feats.append(PA.spectrogram_features(cube))
                rfft_feats.append(PA.range_fft_features(cube))
                kept.append(label)

            if len(kept) < MIN_SAMPLES:
                self.error.emit(
                    f"Only {len(kept)} usable samples, need at least {MIN_SAMPLES}."
                )
                return
            if len(set(kept)) < MIN_CLASSES:
                self.error.emit(
                    f"Only {len(set(kept))} gesture class found, need at least "
                    f"{MIN_CLASSES}."
                )
                return

            labels_arr = np.array(kept)
            spec_proj, spec_var = PA.pca(np.vstack(spec_feats), k=2)
            rfft_proj, rfft_var = PA.pca(np.vstack(rfft_feats), k=2)
            spec_sil = PA.silhouette(spec_proj, labels_arr)
            rfft_sil = PA.silhouette(rfft_proj, labels_arr)

            self.finished.emit(spec_proj, spec_var, float(spec_sil),
                               rfft_proj, rfft_var, float(rfft_sil), kept)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class PcaTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self._c = app_colors()
        self.setObjectName("pca_root")
        self.setStyleSheet(_pca_style(self._c))
        # pyqtgraph is styled imperatively, not by stylesheet.
        dark = is_dark_mode()
        self._plot_bg = "#1a1a2e" if dark else "#ffffff"
        self._axis_pen = pg.mkPen("#888899" if dark else "#999999")
        self._axis_color = "#cccccc" if dark else "#555555"
        self._title_color = "#e6e6e6" if dark else "#1a3a5c"
        self._point_pen = pg.mkPen(self._plot_bg, width=1)
        self._legend_bg = (30, 30, 46, 235) if dark else (255, 255, 255, 235)
        self._legend_border = "#4a4a5a" if dark else "#cccccc"
        self._legend_text = "#e6e6e6" if dark else "#333333"
        self._worker = None
        self._thread = None
        self._paths = []
        self._labels = []
        self._setup_ui()
        self.refresh()

    # ── layout ───────────────────────────────────────────────────────────────

    def _setup_ui(self):
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_left_panel())
        outer.addWidget(self._build_right_panel())

    def _build_left_panel(self):
        panel = QtWidgets.QWidget()
        panel.setObjectName("left_panel")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        heading = QtWidgets.QLabel("PCA Comparison")
        heading.setObjectName("heading")
        layout.addWidget(heading)

        layout.addWidget(self._divider())

        layout.addWidget(self._lbl("Dataset Status"))
        self._status_samples = QtWidgets.QLabel("")
        self._status_samples.setWordWrap(True)
        layout.addWidget(self._status_samples)
        self._status_classes = QtWidgets.QLabel("")
        self._status_classes.setWordWrap(True)
        layout.addWidget(self._status_classes)

        self._refresh_btn = QtWidgets.QPushButton("↻  Refresh")
        self._refresh_btn.setObjectName("refresh_btn")
        self._refresh_btn.clicked.connect(self.refresh)
        layout.addWidget(self._refresh_btn)

        layout.addWidget(self._divider())

        note = QtWidgets.QLabel(
            "This compares two ways of describing the same capture:\n\n"
            "• Doppler domain: the spectrogram, ex. how fast things moved.\n\n"
            "• Range domain: the range FFT profile, ex. how far away "
            "things were.\n\n"
            "Both are reduced to 2D with PCA. The one whose classes form "
            "tighter, more separated clusters is the better representation "
            "for telling these gestures apart."
        )
        note.setObjectName("note")
        note.setWordWrap(True)
        layout.addWidget(note)

        layout.addStretch()

        layout.addWidget(HintCard([
            "PCA finds the directions your data varies in most, then keeps "
            "the top two so it can be drawn on a flat plot.",
            "The silhouette score runs from -1 to +1. Above ~0.5 means classes "
            "sit in clean, well separated clusters.",
            "Explained variance tells you how much of the original detail "
            "survived the squash to 2D. Higher is more faithful.",
            "If neither representation separates well, the model will struggle "
            "too, so collect more samples or more distinct gestures.",
        ], c=self._c))

        self._run_btn = QtWidgets.QPushButton("▶  Run Analysis")
        self._run_btn.setObjectName("pca_btn")
        self._run_btn.clicked.connect(self._start)
        self._run_btn.setEnabled(False)
        layout.addWidget(self._run_btn)

        self._progress = QtWidgets.QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._status_msg = QtWidgets.QLabel("")
        self._status_msg.setWordWrap(True)
        self._status_msg.setStyleSheet(
            f"font-size: 11px; color: {self._c['faint']};"
        )
        layout.addWidget(self._status_msg)

        return _scrollable_left(panel, width=300)

    def _build_right_panel(self):
        panel = QtWidgets.QWidget()
        panel.setStyleSheet(f"background: {self._c['bg']};")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        # PCA scores are linear combinations of the input features, so they
        # carry the input's units. The spectrogram is uniformly dB; the range
        # feature concatenates magnitude with phase, so its units are mixed.
        self._spec_units = "dB"
        self._rfft_units = "mixed: amplitude + rad"

        layout.addWidget(self._measures(
            "Measuring:  Doppler spectrogram amplitude  ·  32x32 resized  ·  "
            "1024 features  ·  units dB"
        ))
        self._spec_plot, self._spec_legend = self._make_plot(
            "Spectrogram (Doppler domain)", self._spec_units
        )
        self._spec_caption = self._caption()
        layout.addWidget(self._spec_plot, 1)
        layout.addWidget(self._spec_caption)
        layout.addWidget(zoom_button_row(self._spec_plot, self._c))

        layout.addWidget(self._measures(
            "Measuring:  range profile magnitude + phase  ·  32x32 each  ·  "
            "2048 features  ·  units linear amplitude and radians "
            "(phase dominates the variance)"
        ))
        self._rfft_plot, self._rfft_legend = self._make_plot(
            "Range FFT (range domain)", self._rfft_units
        )
        self._rfft_caption = self._caption()
        layout.addWidget(self._rfft_plot, 1)
        layout.addWidget(self._rfft_caption)
        layout.addWidget(zoom_button_row(self._rfft_plot, self._c))

        self._summary = QtWidgets.QLabel(
            "Run the analysis to compare the two representations."
        )
        self._summary.setObjectName("summary")
        self._summary.setWordWrap(True)
        # A wrapped QLabel under-reports its height, which clipped the second
        # line; reserve room for two lines explicitly.
        self._summary.setMinimumHeight(62)
        self._summary.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        layout.addWidget(self._summary)

        return panel

    def _make_plot(self, title, units):
        plot = pg.PlotWidget()
        plot.setBackground(self._plot_bg)
        plot.setTitle(title, color=self._title_color, size="11pt", bold=True)
        # Units go in the label text rather than pyqtgraph's units= kwarg,
        # which would add SI prefixes ("kdB") that make no sense here.
        plot.setLabel("left", f"PC2  ({units})", color=self._axis_color)
        plot.setLabel("bottom", f"PC1  ({units})", color=self._axis_color)
        for ax in ("left", "bottom"):
            plot.getAxis(ax).setPen(self._axis_pen)
            plot.getAxis(ax).setTextPen(pg.mkPen(self._axis_color))
        plot.showGrid(x=True, y=True, alpha=0.25)
        plot.setMinimumHeight(180)
        # A backed legend stays readable where it overlaps the points.
        legend = plot.addLegend(
            offset=(-12, 12),
            labelTextColor=self._legend_text,
            labelTextSize="10pt",
            brush=pg.mkBrush(self._legend_bg),
            pen=pg.mkPen(self._legend_border),
        )
        return plot, legend

    def _set_axis_variance(self, plot, units, var):
        plot.setLabel("bottom", f"PC1  ({units}),  {var[0]:.1%} of variance",
                      color=self._axis_color)
        plot.setLabel("left", f"PC2  ({units}),  {var[1]:.1%} of variance",
                      color=self._axis_color)

    def _caption(self):
        lbl = QtWidgets.QLabel("")
        lbl.setObjectName("caption")
        return lbl

    def _measures(self, text):
        lbl = QtWidgets.QLabel(text)
        lbl.setObjectName("measures")
        lbl.setWordWrap(True)
        return lbl

    def _lbl(self, text):
        lbl = QtWidgets.QLabel(text)
        lbl.setObjectName("field_label")
        return lbl

    def _divider(self):
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setStyleSheet(f"color: {self._c['divider']}; margin: 2px 0;")
        return line

    # ── dataset status ───────────────────────────────────────────────────────

    def refresh(self):
        self._paths, self._labels = PA.scan_dataset()
        n = len(self._paths)
        classes = sorted(set(self._labels))

        if n >= MIN_SAMPLES:
            self._status_samples.setObjectName("status_ok")
            self._status_samples.setText(f"✓  {n} raw samples found")
        else:
            self._status_samples.setObjectName("status_err")
            self._status_samples.setText(
                f"✗  {n} raw samples (need {MIN_SAMPLES})"
            )

        if len(classes) >= MIN_CLASSES:
            self._status_classes.setObjectName("status_ok")
            self._status_classes.setText(
                f"✓  {len(classes)} gestures: {', '.join(classes)}"
            )
        else:
            self._status_classes.setObjectName("status_err")
            self._status_classes.setText(
                f"✗  {len(classes)} gesture class (need {MIN_CLASSES})"
            )

        for w in (self._status_samples, self._status_classes):
            w.style().unpolish(w)
            w.style().polish(w)

        ready = n >= MIN_SAMPLES and len(classes) >= MIN_CLASSES
        self._run_btn.setEnabled(ready)
        if not ready and n == 0:
            self._status_msg.setText(
                "No raw samples yet. Collect gestures first. Each capture "
                "saves a sample_NNN_raw.npy alongside the spectrogram."
            )
        elif not ready:
            self._status_msg.setText("Collect more samples to unlock analysis.")
        else:
            self._status_msg.setText("")

    # ── run ──────────────────────────────────────────────────────────────────

    def _start(self):
        self._run_btn.setEnabled(False)
        self._refresh_btn.setEnabled(False)
        self._progress.setMaximum(len(self._paths))
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._summary.setText("Analyzing…")

        self._worker = PcaWorker(self._paths, self._labels)
        self._thread = QtCore.QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._thread.start()

    def _on_progress(self, i, total, name):
        self._progress.setValue(i)
        self._status_msg.setText(f"Processing {i}/{total}: {name}")

    def _on_finished(self, spec_proj, spec_var, spec_sil,
                     rfft_proj, rfft_var, rfft_sil, labels):
        self._cleanup_thread()
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)
        self._refresh_btn.setEnabled(True)
        self._status_msg.setText(f"Done. {len(labels)} samples analyzed.")

        classes = sorted(set(labels))
        palette = class_colors()
        colors = {c: palette[i % len(palette)] for i, c in enumerate(classes)}

        self._draw(self._spec_plot, self._spec_legend, spec_proj, labels,
                   classes, colors)
        self._draw(self._rfft_plot, self._rfft_legend, rfft_proj, labels,
                   classes, colors)

        self._set_axis_variance(self._spec_plot, self._spec_units, spec_var)
        self._set_axis_variance(self._rfft_plot, self._rfft_units, rfft_var)

        self._spec_caption.setText(
            f"PC1 {spec_var[0]:.1%}   PC2 {spec_var[1]:.1%}   "
            f"({spec_var[0] + spec_var[1]:.1%} of detail kept)   "
            f"silhouette {spec_sil:+.3f}  (-1 to +1, higher = better separated)"
        )
        self._rfft_caption.setText(
            f"PC1 {rfft_var[0]:.1%}   PC2 {rfft_var[1]:.1%}   "
            f"({rfft_var[0] + rfft_var[1]:.1%} of detail kept)   "
            f"silhouette {rfft_sil:+.3f}  (-1 to +1, higher = better separated)"
        )
        self._summary.setText(self._summarise(spec_sil, rfft_sil))

    def _draw(self, plot, legend, proj, labels, classes, colors):
        plot.clear()
        legend.clear()
        labels = np.asarray(labels)
        for c in classes:
            mask = labels == c
            item = pg.ScatterPlotItem(
                x=proj[mask, 0], y=proj[mask, 1],
                size=11, pen=self._point_pen,
                brush=pg.mkBrush(colors[c]),
            )
            plot.addItem(item)
            legend.addItem(item, f"{c}  ({int(mask.sum())})")
        plot.enableAutoRange()

    def _summarise(self, spec_sil, rfft_sil):
        diff = abs(spec_sil - rfft_sil)
        if spec_sil > rfft_sil:
            better, worse = "Spectrogram (Doppler domain)", "Range FFT"
        else:
            better, worse = "Range FFT (range domain)", "Spectrogram"
        best = max(spec_sil, rfft_sil)

        if best < 0.1:
            quality = ("Neither representation separates these gestures "
                       "cleanly, the clusters overlap a lot.")
        elif best < 0.35:
            quality = "The better one separates them only weakly."
        elif best < 0.6:
            quality = "The better one separates them reasonably well."
        else:
            quality = "The better one separates them into clean, tight clusters."

        if diff < 0.02:
            return (f"Both representations perform about the same "
                    f"(silhouette {spec_sil:+.3f} vs {rfft_sil:+.3f}). {quality}")
        return (f"{better} separates the gestures better than {worse}. "
                f"Silhouette {max(spec_sil, rfft_sil):+.3f} vs "
                f"{min(spec_sil, rfft_sil):+.3f}, a gap of {diff:.3f}. {quality}")

    def _on_error(self, msg):
        self._cleanup_thread()
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)
        self._refresh_btn.setEnabled(True)
        self._status_msg.setText("")
        self._summary.setText(f"✗  {msg.splitlines()[0]}")

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
