@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo 先に install_gpu.bat または install.bat を実行してください。
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -c "import pytest" >nul 2>nul
if errorlevel 1 (
  python -m pip install -r requirements-dev.txt
  if errorlevel 1 (
    echo pytestの導入に失敗しました。
    pause
    exit /b 1
  )
)
pytest -q
set TEST_EXIT=%ERRORLEVEL%
pause
exit /b %TEST_EXIT%
