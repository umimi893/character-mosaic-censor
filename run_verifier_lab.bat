@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" -m character_mosaic.verifier_lab
  exit /b 0
)
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m character_mosaic.verifier_lab
  exit /b %errorlevel%
)
python -m character_mosaic.verifier_lab
