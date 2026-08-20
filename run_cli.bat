@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run install_gpu.bat or install.bat first.
  exit /b 1
)
.venv\Scripts\python.exe -m character_mosaic.cli %*
