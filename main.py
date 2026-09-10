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


def self_test() -> int:
    """
    Import every subsystem and report what this copy of the app can do.

    Most of the heavy imports happen lazily inside worker threads, so a build
    that lost one of them starts perfectly and only fails later, on whichever
    tab a student happens to open. Run this once on a new machine — or on a
    fresh copy from the flash drive — to find out up front.

        SensDSv2.exe --self-test

    The Windows build is a windowed app, and PyInstaller gives a windowed app
    no console at all: sys.stdout is None and print() goes nowhere. So when
    there is nothing to print to, the report appears in a dialog instead.
    """
    lines = []
    code = _self_test_report(lines)
    text = "\n".join(lines)
    if sys.stdout is not None:
        print(text)
    else:
        _show_report(code, text)
    return code


def _show_report(code: int, text: str):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    box = QtWidgets.QMessageBox()
    box.setWindowTitle("SensDSv2 self-test")
    box.setIcon(QtWidgets.QMessageBox.Icon.Information if code == 0
                else QtWidgets.QMessageBox.Icon.Critical)
    box.setText("Everything required is present." if code == 0
                else "This build is missing required components.")
    # Full report under "Show Details", fixed-width so the ok / FAIL column
    # lines up.
    box.setDetailedText(text)
    box.setStyleSheet(
        "QTextEdit { font-family: Menlo, Consolas, 'DejaVu Sans Mono', "
        "monospace; font-size: 12px; min-width: 520px; min-height: 320px; }"
    )
    box.exec()


def _self_test_report(out: list) -> int:
    """Append the report lines to `out`; return 0 if all required parts load."""
    from ui.train_tab import _MODEL_OPTIONS, _app_dir, model_is_available_offline

    checks = [
        ("Qt GUI", lambda: __import__("PyQt6.QtWidgets", fromlist=["x"]), True),
        ("plotting (pyqtgraph)", lambda: __import__("pyqtgraph"), True),
        ("radar maths (scipy)", lambda: __import__("scipy.signal", fromlist=["x"]), True),
        ("PyTorch", lambda: __import__("torch"), True),
        ("torchvision", lambda: __import__("torchvision"), True),
        ("model loading (transformers)", lambda: __import__(
            "transformers", fromlist=["AutoModelForImageClassification"]), True),
        ("image handling (Pillow)", lambda: __import__("PIL.Image", fromlist=["x"]), True),
        ("reference view (matplotlib)", lambda: __import__(
            "matplotlib.backends.backend_qtagg", fromlist=["x"]), True),
        ("reference spectrogram script",
         lambda: __import__("core.doppler_spectrogram_live", fromlist=["x"]), True),
        ("VEX AIM robot", lambda: __import__("vex.aim", fromlist=["Robot"]), False),
        ("live radar (Infineon SDK)",
         lambda: __import__("ifxradarsdk.fmcw", fromlist=["DeviceFmcw"]), False),
    ]

    out.append(f"SensDSv2 self-test    frozen={getattr(sys, 'frozen', False)}")
    out.append(f"resources: {_app_dir()}")
    out.append("")

    failures = 0
    for label, probe, required in checks:
        try:
            probe()
            out.append(f"  ok    {label}")
        except Exception as exc:
            if required:
                failures += 1
                out.append(f"  FAIL  {label}: {type(exc).__name__}: {exc}")
            else:
                out.append(f"  --    {label} unavailable ({type(exc).__name__})")

    out.append("")
    for key, model_id in _MODEL_OPTIONS.items():
        offline = model_is_available_offline(model_id)
        out.append(f"  {'ok  ' if offline else '--  '}  base model {key}: "
                   + ("bundled, trains offline" if offline
                      else "not bundled, needs internet once"))

    out.append("")
    if failures:
        out.append(f"{failures} required component(s) missing. This build is broken.")
        return 1
    out.append("Everything required is present.")
    return 0


def main():
    if "--self-test" in sys.argv:
        sys.exit(self_test())

    # Before anything opens the radar: the Infineon SDK prints dropped-packet
    # notices to stdout from C++, several a second, which buries everything
    # else. This counts them and prints a summary instead. See core/sdk_log.py.
    from core import sdk_log
    sdk_log.install()

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("SensDSv2")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
