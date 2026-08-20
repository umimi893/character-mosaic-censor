@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run install_gpu.bat or install.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe diagnose.py %*
set DIAG_EXIT=%ERRORLEVEL%
echo.
echo To test model loading:
echo   diagnose.bat --model-test
if not "%DIAG_EXIT%"=="0" echo Diagnostics found a problem. Check the ERROR/WARNING above.
pause
exit /b %DIAG_EXIT%
