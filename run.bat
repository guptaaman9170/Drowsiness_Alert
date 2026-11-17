@echo off
title Driver Drowsiness and Sleep Alert System
color 0A
echo ==============================================================
echo    DRIVER DROWSINESS AND SLEEP ALERT SYSTEM
echo    Reference: OpenCV + MediaPipe / GeeksforGeeks EAR Pipeline
echo ==============================================================
echo.
echo Starting application (Press 'q' or ESC in the video window to exit)...
echo.
python main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application exited with an error.
    echo Please make sure dependencies are installed: pip install -r requirements.txt
    pause
)

