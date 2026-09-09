# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec — produces:
#   Windows: dist/SensDSv2/ folder  (one-DIRECTORY build — fast startup)
#   macOS  : dist/SensDSv2.app
#
# ── Build instructions ────────────────────────────────────────────────────────
#
#  Windows (must run ON Windows — PyInstaller cannot cross-compile):
#    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#    pip install -r requirements.txt pyinstaller
#    pip install vendor\ifxradarsdk-*-win_amd64.whl     ← for live radar
#    pyinstaller main.spec
#    → dist\SensDSv2\   ← zip this whole folder and distribute
#
#  macOS (dev machine):
#    pip install -r requirements.txt && pyinstaller main.spec
#
#  See PACKAGING.md for hosting, USB distribution, and the radar SDK.
# ─────────────────────────────────────────────────────────────────────────────

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

SPEC_DIR = Path(SPECPATH)

datas = [('assets', 'assets')]
binaries = []
hiddenimports = []


def bundle(package, optional=False):
    """collect_all a package, or say plainly why the build will be limited."""
    try:
        d, b, h = collect_all(package)
    except Exception as exc:
        if not optional:
            raise
        print(f"SPEC: {package} not installed — {exc}")
        return False
    datas.extend(d)
    binaries.extend(b)
    hiddenimports.extend(h)
    print(f"SPEC: bundled {package} ({len(d)} data files, {len(b)} binaries)")
    return True


# ── GUI stack ────────────────────────────────────────────────────────────────
bundle('PyQt6')
bundle('pyqtgraph')

# matplotlib is not optional: core/doppler_spectrogram_live.py imports it at
# module scope and the Visualize tab's reference view draws with it. It was
# absent from earlier builds, so "Compare with reference view" failed in the
# packaged app while working fine from source.
bundle('matplotlib')

# ── Machine learning ─────────────────────────────────────────────────────────
# torch is imported lazily inside worker threads, so static analysis misses it.
bundle('torch')
bundle('torchvision')
bundle('transformers')
bundle('PIL')

# ── Infineon Radar SDK ───────────────────────────────────────────────────────
# Ships native libraries (.dll on Windows, .dylib on macOS) that must match the
# build platform. A macOS wheel in a Windows build yields an app that starts but
# can never open the radar.
if not bundle('ifxradarsdk', optional=True):
    print("SPEC: ==> live radar streaming will NOT work in this build.")
    print("SPEC:     Install this platform's ifxradarsdk wheel and rebuild.")

# ── VEX AIM robot ────────────────────────────────────────────────────────────
# ui/vex_aim_tab.py does `from vex.aim import Robot` inside a method, so
# PyInstaller never sees it. vex/settings.py then reads settings.json from
# beside itself, so the JSON has to travel with the package.
datas.append((str(SPEC_DIR / 'vex' / 'settings.json'), 'vex'))
hiddenimports += [
    'vex', 'vex.aim', 'vex.settings', 'vex.vex_globals',
    'vex.vex_messages', 'vex.vex_types',
    'websocket',                      # vex.aim's transport
]

# ── Base model for offline training ──────────────────────────────────────────
# Optional. Populate models/ with `python tools/fetch_base_model.py` before
# building and the app can train with no internet at all; without it the Train
# tab has to reach HuggingFace once.
models_dir = SPEC_DIR / 'models'
if models_dir.is_dir() and any(models_dir.iterdir()):
    datas.append((str(models_dir), 'models'))
    print(f"SPEC: bundled base models: {[p.name for p in models_dir.iterdir()]}")
else:
    print("SPEC: no models/ directory — the Train tab will need internet once.")

hiddenimports += [
    'transformers.models.auto',
    'transformers.models.vit',
    'transformers.models.convnext',
    'accelerate',
    'huggingface_hub',
    'scipy.signal',
    'scipy.signal.windows',
    'scipy.ndimage',
    'matplotlib.backends.backend_qtagg',
    'ifxradarsdk',
    'ifxradarsdk.fmcw',
    'ifxradarsdk.fmcw.types',
]

# ─────────────────────────────────────────────────────────────────────────────

a = Analysis(
    ['main.py'],
    pathex=[str(SPEC_DIR)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # NOTE: do NOT exclude torch.cuda. `import torch` imports it unconditionally
    # even in the CPU-only build, so excluding it makes the packaged app die at
    # startup with ModuleNotFoundError: No module named 'torch.cuda'. The
    # CPU-only wheel is what actually keeps CUDA out of the bundle.
    excludes=['tkinter'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

# ── Windows: one-DIRECTORY build ──────────────────────────────────────────────
# One-file mode would unpack well over a gigabyte of DLLs into a temp folder on
# every launch: a 30-60 second black screen before the window appears, every
# time. The folder build starts immediately.
if sys.platform == 'win32':
    exe = EXE(
        pyz,
        a.scripts,
        [],                     # binaries stay outside the exe
        [],                     # datas stay outside the exe
        exclude_binaries=True,
        name='SensDSv2',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        # UPX corrupts PyTorch's DLLs on Windows — leave it off.
        upx=False,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        icon='assets/SensDSLogo.ico',
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        name='SensDSv2',        # → dist/SensDSv2/
    )

# ── macOS: app bundle ─────────────────────────────────────────────────────────
else:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        [],
        exclude_binaries=True,
        name='SensDSv2',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon='assets/SensDSLogo.icns',
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        name='SensDSv2',
    )
    app = BUNDLE(
        coll,
        name='SensDSv2.app',
        icon='assets/SensDSLogo.icns',
        bundle_identifier='edu.sensds.v2',
    )
