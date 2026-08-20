@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo 先に install_gpu.bat または install.bat を実行してください。
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m character_mosaic.gui
set APP_EXIT=%ERRORLEVEL%
if not "%APP_EXIT%"=="0" (
  echo.
  echo アプリがエラー終了しました。diagnose.bat を実行して確認してください。
  pause
)
exit /b %APP_EXIT%
