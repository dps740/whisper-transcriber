@echo off
echo.
echo   Whisper Bulk Transcriber
echo   ========================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)

REM Install dependencies if needed
pip show faster-whisper >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
    echo.
)

echo Starting server... (browser will open automatically)
echo.
python transcribe_server.py
pause
