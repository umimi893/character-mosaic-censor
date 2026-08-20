@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo 先に install.bat を実行してください。
  exit /b 1
)
.venv\Scripts\python.exe -m character_mosaic.cli %*
