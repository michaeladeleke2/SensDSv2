import os
import sys
from PyQt6 import QtWidgets
from ui.main_window import MainWindow

if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
    lib_path = os.path.join(base_path, 'ifxradarsdk', 'lib')
    os.environ['DYLD_LIBRARY_PATH'] = lib_path

# Limit PyTorch CPU thread usage so the UI stays responsive during inference
# and training on low-core-count machines (Surface Pro 9: i7-1265U, 10 cores).
# set_num_threads     → intra-op parallelism (matrix ops inside a single op)
# set_num_interop_threads → inter-op parallelism (running independent ops in parallel)
# 4 + 2 leaves plenty of headroom for the Qt event loop and radar streaming thread.
try:
    import torch
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
