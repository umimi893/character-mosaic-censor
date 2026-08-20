@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\pythonw.exe (
  echo Run START_HERE.bat first, or install with install_gpu.bat / install.bat.
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "%~dp0run_gui.py"
exit /b 0
