@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo 先に install_gpu.bat または install.bat を実行してください。
  pause
  exit /b 1
)
.venv\Scripts\python.exe diagnose.py %*
set DIAG_EXIT=%ERRORLEVEL%
echo.
echo モデル読み込みまで確認する場合:
echo   diagnose.bat --model-test
if not "%DIAG_EXIT%"=="0" echo 診断で問題が見つかりました。上のERROR/WARNINGを確認してください。
pause
exit /b %DIAG_EXIT%
