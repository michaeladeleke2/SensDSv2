import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui
from scipy.ndimage import gaussian_filter


DISPLAY_SECONDS = 5
FREQ_BINS = 1024

# Radar / STFT parameters — must match core/processing.py and core/radar.py
FRAME_TIME_S = 0.15   # cfg.frame_repetition_time_s
_CHIRPS_PER_FRAME = 64
_STFT_SHIFT = 8       # SHIFT = WINDOW - NOVERLAP in processing.py
# Spectrogram columns produced per second of live radar data
COLS_PER_SECOND = int(round(_CHIRPS_PER_FRAME / (FRAME_TIME_S * _STFT_SHIFT)))
BUFFER_WIDTH = DISPLAY_SECONDS * COLS_PER_SECOND  # ~265 columns for 5 s
DB_MIN = -20
DB_MAX = 0

PRF = 1.0 / 0.0005
WAVELENGTH = 3e8 / 61e9
MAX_VELOCITY = (PRF * WAVELENGTH) / 4  # ±2.46 m/s


def make_jet_colormap():
    positions = [0.0, 0.125, 0.375, 0.625, 0.875, 1.0]
    colors = [
        (0,   0,   143, 255),
        (0,   0,   255, 255),
        (0,   255, 255, 255),
        (255, 255, 0,   255),
        (255, 0,   0,   255),
        (128, 0,   0,   255),
    ]
    return pg.ColorMap(positions, colors)


class SpectrogramWidget(pg.GraphicsLayoutWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._buffer = np.full((FREQ_BINS, BUFFER_WIDTH), DB_MIN, dtype=np.float32)
        self._col = 0
        self._setup_plot()

    def _setup_plot(self):
        self.setBackground('#00008F')

        plot = self.addPlot()
        plot.setLabel('left', 'Velocity', units='m/s')
        plot.setLabel('bottom', 'Time', units='s')
        plot.hideButtons()
        plot.setMouseEnabled(x=False, y=False)

        ax_left = plot.getAxis('left')
        ax_left.setTextPen(pg.mkPen('w'))
        ax_left.setPen(pg.mkPen('w'))
        ax_bottom = plot.getAxis('bottom')
        ax_bottom.setTextPen(pg.mkPen('w'))
        ax_bottom.setPen(pg.mkPen('w'))

        self._img = pg.ImageItem()
        plot.addItem(self._img)

        colormap = make_jet_colormap()
        self._img.setColorMap(colormap)
        self._img.setLevels([DB_MIN, DB_MAX])

        time_scale = DISPLAY_SECONDS / BUFFER_WIDTH
        vel_scale = (2 * MAX_VELOCITY) / FREQ_BINS
        self._img.setTransform(
            QtGui.QTransform().scale(time_scale, vel_scale).translate(0, -FREQ_BINS / 2)
        )

        plot.setXRange(0, DISPLAY_SECONDS, padding=0)
        plot.setYRange(-MAX_VELOCITY, MAX_VELOCITY, padding=0)

        zero_line = pg.InfiniteLine(
            pos=0,
            angle=0,
            pen=pg.mkPen(color=(255, 255, 255, 60), width=1)
        )
        plot.addItem(zero_line)

    def update_frame(self, spectrogram_batch):
        n_cols = spectrogram_batch.shape[1]
        for i in range(n_cols):
            col = np.clip(spectrogram_batch[:, i], DB_MIN, DB_MAX).astype(np.float32)
            self._buffer[:, self._col] = col
            self._col = (self._col + 1) % BUFFER_WIDTH
        display = np.roll(self._buffer, -self._col, axis=1)
        display = gaussian_filter(display.astype(np.float64), sigma=[2.0, 1.5]).astype(np.float32)
        self._img.setImage(display.T, autoLevels=False)