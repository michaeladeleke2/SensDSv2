import sys
import numpy as np
from PyQt6 import QtWidgets, QtCore
from ui.spectrogram_widget import SpectrogramWidget
from core.radar import RadarStream
from core.processing import SpectrogramProcessor


class RadarBridge(QtCore.QObject):
    frame_ready = QtCore.pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._processor = SpectrogramProcessor()
        self._stream = RadarStream(
            on_frame=self._on_frame,
            on_error=self._on_error
        )

    def _on_frame(self, frame):
        result = self._processor.push_frame(frame)
        if result is not None:
            self.frame_ready.emit(result)


    def _on_error(self, msg):
        print(f"Radar error: {msg}")

    def start(self):
        self._stream.start()

    def stop(self):
        self._stream.stop()


app = QtWidgets.QApplication(sys.argv)
window = QtWidgets.QMainWindow()
widget = SpectrogramWidget()
window.setCentralWidget(widget)
window.resize(900, 400)
window.setWindowTitle('SensDSv2 — Live Radar')
window.show()

bridge = RadarBridge()
bridge.frame_ready.connect(widget.update_frame)
bridge.start()

app.exec()
bridge.stop()