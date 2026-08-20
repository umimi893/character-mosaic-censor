@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run install_gpu.bat or install.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -c "import pytest" >nul 2>nul
if errorlevel 1 (
  python -m pip install -r requirements-dev.txt
  if errorlevel 1 (
    echo Failed to install pytest.
    pause
    exit /b 1
  )
)
pytest -q
set TEST_EXIT=%ERRORLEVEL%
pause
exit /b %TEST_EXIT%
