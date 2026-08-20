@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\pythonw.exe (
  echo Run install_gpu.bat or install.bat first.
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m character_mosaic.gui
exit /b 0
