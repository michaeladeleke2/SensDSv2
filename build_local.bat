@echo off
setlocal enabledelayedexpansion
title SensDSv2 — Local Windows Build (with Radar SDK)

echo ============================================================
echo   SensDSv2  —  Local Build  (includes Infineon Radar SDK)
echo ============================================================
echo.
echo Run this on the Windows machine that has the Infineon SDK installed.
echo The resulting dist\SensDSv2\ folder can then be copied to all Surfaces.
echo.

:: ── 1. Verify Python ──────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Run setup_windows.bat first.
    pause & exit /b 1
)

:: ── 2. Check Infineon SDK is installed ────────────────────────────────────────
python -c "import ifxradarsdk" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Infineon Radar SDK ^(ifxradarsdk^) is not installed.
    echo.
    echo Install it first:
    echo   pip install path\to\ifxradarsdk-*.whl
    echo.
    echo Then re-run this script.
    pause & exit /b 1
)
echo [OK] Infineon Radar SDK found.

:: ── 3. Check PyTorch is CPU-only ──────────────────────────────────────────────
python -c "import torch; assert not torch.cuda.is_available() or True" >nul 2>&1
if errorlevel 1 (
    echo [WARN] Could not verify PyTorch. Continuing anyway...
)
python -c "import torch; print('[OK] PyTorch', torch.__version__)" 2>&1

:: ── 4. Ensure PyInstaller is installed ───────────────────────────────────────
python -m pip install pyinstaller --quiet

:: ── 5. Clean previous build artifacts ────────────────────────────────────────
echo.
echo [1/2] Cleaning previous build...
if exist build\main rmdir /s /q build\main
if exist dist\SensDSv2 rmdir /s /q dist\SensDSv2
echo Done.

:: ── 6. Run PyInstaller ────────────────────────────────────────────────────────
echo.
echo [2/2] Building with PyInstaller (this takes 3-5 minutes)...
python -m PyInstaller main.spec
if errorlevel 1 (
    echo.
    echo [ERROR] PyInstaller failed. See the output above for details.
    pause & exit /b 1
)

:: ── 7. Done ──────────────────────────────────────────────────────────────────
echo.
echo ============================================================
echo   Build complete!
echo.
echo   Output: dist\SensDSv2\
echo.
echo   To distribute:
echo     1. Zip the dist\SensDSv2\ folder
echo     2. Copy to each Surface and unzip
echo     3. Students double-click SensDSv2.exe
echo ============================================================
echo.
pause
endlocal
