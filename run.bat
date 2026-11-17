@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app.py
) else (
    python app.py
)
if errorlevel 1 (
    echo.
    echo Khong khoi dong duoc. Xem huong dan trong README.md.
    pause
)
endlocal
