@echo off
echo 🚛 DrayVis Rate Estimator
echo ====================================
echo.

:: Try to find Python executable
where python >nul 2>&1
if %errorlevel% == 0 (
    echo Starting DrayVis GUI...
    python launch_gui.py
) else (
    echo ❌ Python not found in PATH
    echo.
    echo Please install Python or add it to your PATH:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation.
)

echo.
pause
