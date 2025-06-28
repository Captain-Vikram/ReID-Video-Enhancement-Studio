@echo off
REM Launch script for ReID Video Enhancement Studio (Windows)
REM Double-click this file to start the GUI application

echo.
echo ========================================
echo   ReID Video Enhancement Studio
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8 or later.
    echo    Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Check if app.py exists
if not exist "app.py" (
    echo ❌ app.py not found in current directory
    echo 💡 Please run this script from the gui directory
    pause
    exit /b 1
)

REM Run the Python launcher
echo 🚀 Starting application...
python launch_gui.py

echo.
echo 👋 Application closed.
pause
