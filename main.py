import os
import sys
from PyQt6 import QtWidgets
from ui.main_window import MainWindow

if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
    lib_path = os.path.join(base_path, 'ifxradarsdk', 'lib')
    os.environ['DYLD_LIBRARY_PATH'] = lib_path

# ── PyTorch thread tuning ─────────────────────────────────────────────────────
# set_num_threads      → intra-op parallelism (matrix ops within one op)
# set_num_interop_threads → inter-op parallelism (ops running in parallel)
#
# Surface Pro 9 (i7-1265U, 10 cores, CPU-only torch):
#   We give torch 6 intra-op threads — enough for fast matrix multiplications
#   inside the ViT model — while leaving 4 cores free for the Qt event loop,
#   the radar streaming thread, and the spectrogram display thread.
#
# Mac / CUDA (hardware-accelerated): inference runs on GPU so CPU threads
#   matter less; keep at 4 to avoid unnecessary overhead.
try:
    import torch
    if sys.platform == "win32":
        # Windows / Surface: more intra-op threads → faster per-inference,
        # partially compensating for the lack of MPS/CUDA.
        torch.set_num_threads(6)
        torch.set_num_interop_threads(2)
    else:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(2)
except ImportError:
    pass  # torch not yet installed — setup_windows.bat handles this


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("SensDSv2")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
