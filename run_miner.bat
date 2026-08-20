@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run START_HERE.bat first, or install with install_gpu.bat / install.bat.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m character_mosaic.miner_cli %*
exit /b %errorlevel%
