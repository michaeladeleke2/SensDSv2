@echo off
setlocal enabledelayedexpansion
title SensDSv2 — Windows Setup

echo ============================================================
echo   SensDSv2  —  First-Time Windows Setup
echo ============================================================
echo.

:: ── 1. Verify Python is on PATH ───────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found.
    echo.
    echo Please install Python 3.10 or 3.11 from https://www.python.org/downloads/
    echo Make sure to tick "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)
echo [OK] Python found:
python --version
echo.

:: ── 2. Upgrade pip ────────────────────────────────────────────────────────────
echo [1/4] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo Done.
echo.

:: ── 3. Install CPU-only PyTorch ───────────────────────────────────────────────
::
::  IMPORTANT: We install the CPU-only build of PyTorch.
::  The default "pip install torch" installs the CUDA/GPU build which requires
::  NVIDIA drivers and ships with c10.dll, torch_cuda.dll, etc.  Those DLLs
::  will fail to load on devices without an NVIDIA GPU (Surface, Chromebook,
::  most school laptops).  The CPU build is ~250 MB smaller and runs on any PC.
::
echo [2/4] Installing PyTorch (CPU-only build — works on all Windows devices)...
python -m pip install torch torchvision ^
    --index-url https://download.pytorch.org/whl/cpu ^
    --quiet
if errorlevel 1 (
    echo [ERROR] PyTorch install failed.  Check your internet connection and try again.
    pause
    exit /b 1
)
echo Done.
echo.

:: ── 4. Install remaining dependencies ────────────────────────────────────────
echo [3/4] Installing remaining dependencies...
python -m pip install ^
    "PyQt6>=6.5.0" ^
    "pyqtgraph>=0.13.3" ^
    "numpy>=1.24.0" ^
    "scipy>=1.10.0" ^
    "transformers>=4.30.0" ^
    "Pillow>=9.0.0" ^
    "evaluate" ^
    "accelerate" ^
    "websocket-client>=1.9" ^
    --quiet
if errorlevel 1 (
    echo [ERROR] Dependency install failed.  Check your internet connection and try again.
    pause
    exit /b 1
)
echo Done.
echo.

:: ── 5. Install the Infineon radar SDK if present ─────────────────────────────
echo [4/4] Checking for Infineon Radar SDK...
if exist "ifxradarsdk" (
    python -m pip install ./ifxradarsdk --quiet
    echo Infineon SDK installed from local folder.
) else (
    python -c "import ifxradarsdk" >nul 2>&1
    if errorlevel 1 (
        echo [WARN] Infineon Radar SDK not found.
        echo        The app will start but radar streaming will be unavailable.
        echo        Copy the SDK wheel into this folder and re-run setup.
    ) else (
        echo Infineon SDK already installed.
    )
)
echo.

:: ── Done ──────────────────────────────────────────────────────────────────────
echo ============================================================
echo   Setup complete!
echo   Run the app by double-clicking  run_windows.bat
echo ============================================================
echo.
pause
endlocal
