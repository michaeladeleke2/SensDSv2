@echo off
setlocal
title SensDSv2

:: ── Verify Python is available ────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.  Run setup_windows.bat first.
    pause
    exit /b 1
)

:: ── Quick dependency check ─────────────────────────────────────────────────────
python -c "import PyQt6, torch, numpy" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Required packages are missing.
    echo         Please run setup_windows.bat to install them.
    pause
    exit /b 1
)

:: ── Launch the app ───────────────────────────────────────────────────────────
cd /d "%~dp0"
python main.py
if errorlevel 1 (
    echo.
    echo [ERROR] The application exited with an error.
    echo         See the message above for details.
    pause
)
endlocal
