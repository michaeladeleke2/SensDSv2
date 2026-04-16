# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec — produces:
#   macOS : SensDSv2.app  (one-file app bundle, via pyinstaller main.spec)
#   Windows: dist/SensDSv2/ folder  (one-DIRECTORY build — fast startup, no temp extraction)
#
# ── Build instructions ────────────────────────────────────────────────────────
#
#  macOS (dev machine):
#    pip install -r requirements.txt
#    pyinstaller main.spec
#    → dist/SensDSv2.app
#
#  Windows (must be run ON a Windows machine or via GitHub Actions):
#    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#    pip install -r requirements.txt
#    pyinstaller main.spec
#    → dist/SensDSv2/          ← zip this entire folder and distribute
#       SensDSv2.exe            ← students double-click this
#       (+ hundreds of .dll files alongside it)
#
#  DO NOT build for Windows on macOS — PyInstaller bundles are OS-specific.
#  Use GitHub Actions (.github/workflows/build_windows.yml) to build automatically.
#
# ─────────────────────────────────────────────────────────────────────────────

import sys
from PyInstaller.utils.hooks import collect_all, collect_data_files

datas    = [('assets', 'assets')]
binaries = []
hiddenimports = []

# ── PyQt6 ────────────────────────────────────────────────────────────────────
tmp = collect_all('PyQt6')
datas += tmp[0]; binaries += tmp[1]; hiddenimports += tmp[2]

# ── Infineon Radar SDK (optional — skipped if not installed on build machine) ─
try:
    tmp = collect_all('ifxradarsdk')
    datas += tmp[0]; binaries += tmp[1]; hiddenimports += tmp[2]
    print("INFO: ifxradarsdk collected.")
except Exception:
    print("INFO: ifxradarsdk not found on this build machine.")
    print("INFO: The built app will start and run normally, but radar streaming")
    print("INFO: will not work.  To include radar support, run build_local.bat")
    print("INFO: on a Windows machine that has the Infineon SDK installed.")

# ── PyTorch ───────────────────────────────────────────────────────────────────
# torch is imported lazily inside worker threads so PyInstaller cannot detect
# it automatically.  collect_all ensures all DLLs are bundled.
# ALWAYS build Windows targets with CPU-only torch (no CUDA DLLs → no c10.dll error).
tmp = collect_all('torch')
datas += tmp[0]; binaries += tmp[1]; hiddenimports += tmp[2]

tmp = collect_all('torchvision')
datas += tmp[0]; binaries += tmp[1]; hiddenimports += tmp[2]

# ── HuggingFace Transformers ──────────────────────────────────────────────────
tmp = collect_all('transformers')
datas += tmp[0]; binaries += tmp[1]; hiddenimports += tmp[2]
datas += collect_data_files('transformers')

hiddenimports += [
    'transformers.models.auto',
    'transformers.models.vit',
    'transformers.models.convnext',
    'accelerate',
    'scipy.signal',
    'scipy.ndimage',
    # Infineon SDK — imported lazily in core/radar.py so PyInstaller won't
    # detect them via static analysis; list them explicitly so they're found
    # when the SDK *is* present on the build machine.
    'ifxradarsdk',
    'ifxradarsdk.fmcw',
    'ifxradarsdk.fmcw.types',
]

# ─────────────────────────────────────────────────────────────────────────────

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Exclude CUDA sub-packages when using CPU-only torch — shrinks bundle significantly.
    excludes=['torch.cuda', 'torchvision.models.detection'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

# ── Windows: one-DIRECTORY build ──────────────────────────────────────────────
# Distributing the dist/SensDSv2/ folder (zipped) is the correct Windows approach.
# One-file mode would extract 1-2 GB of DLLs to a temp folder on every launch,
# causing a 30-60 second black screen before the app opens.
if sys.platform == 'win32':
    exe = EXE(
        pyz,
        a.scripts,
        [],        # ← binaries NOT packed into the exe
        [],        # ← datas NOT packed into the exe
        name='SensDSv2',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        # UPX compresses DLLs but BREAKS PyTorch DLLs on Windows — disable it.
        upx=False,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        icon='assets/SensDSLogo.ico' if sys.platform == 'win32' else None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        name='SensDSv2',    # → dist/SensDSv2/
    )

# ── macOS: one-file app bundle ────────────────────────────────────────────────
else:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name='SensDSv2',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=['torch*.dylib', 'libtorch*.dylib'],
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon='assets/SensDSLogo.icns',
    )
    app = BUNDLE(
        exe,
        name='SensDSv2.app',
        icon='assets/SensDSLogo.icns',
        bundle_identifier='edu.sensds.v2',
    )
