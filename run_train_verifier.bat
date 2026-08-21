@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo .venv\Scripts\python.exe not found.
  echo Run setup.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m character_mosaic.verifier_train_cli %*
set EXIT_CODE=%ERRORLEVEL%
echo.
echo verifier trainer exit code = %EXIT_CODE%
pause
exit /b %EXIT_CODE%
