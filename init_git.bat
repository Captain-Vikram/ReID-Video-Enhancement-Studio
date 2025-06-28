@echo off
REM Git Repository Initialization Script for Windows
REM ReID Video Enhancement Studio

echo.
echo ========================================
echo   ReID Video Enhancement Studio
echo   Git Repository Initialization
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not found in PATH
    echo Please install Python 3.8+ from: https://www.python.org/
    pause
    exit /b 1
)

REM Check if Git is available
git --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git is not installed or not found in PATH
    echo Please install Git from: https://git-scm.com/
    pause
    exit /b 1
)

echo Running Git initialization script...
echo.

REM Run the Python initialization script
python init_git.py

if errorlevel 1 (
    echo.
    echo ERROR: Git initialization failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Git Repository Ready!
echo ========================================
echo.
echo Your repository has been initialized with:
echo   - Proper .gitignore and .gitattributes
echo   - Initial commit with all project files
echo   - Development branch structure
echo   - Git aliases and configuration
echo.
echo Next steps:
echo   1. Set up remote repository on GitHub/GitLab
echo   2. git remote add origin [repository-url]
echo   3. git push -u origin main
echo.
pause
