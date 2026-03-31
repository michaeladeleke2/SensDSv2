import os
import sys
from PyQt6 import QtWidgets
from ui.main_window import MainWindow

if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
    lib_path = os.path.join(base_path, 'ifxradarsdk', 'lib')
    os.environ['DYLD_LIBRARY_PATH'] = lib_path


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("SensDSv2")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
