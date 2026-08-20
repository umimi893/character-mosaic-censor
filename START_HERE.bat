@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title Character Mosaic Censor

rem Read the displayed version from pyproject.toml so release bumps cannot
rem leave this launcher showing an old hard-coded version.
set "APP_VERSION=unknown"
for /f "tokens=2 delims==" %%V in ('findstr /b /c:"version = " "pyproject.toml" 2^>nul') do set "APP_VERSION=%%V"
set "APP_VERSION=%APP_VERSION: =%"
set "APP_VERSION=%APP_VERSION:"=%"

echo ================================================
echo Character Mosaic Censor v%APP_VERSION%
echo ================================================
echo.

if not exist ".venv\Scripts\python.exe" (
  echo First-time setup will now start.
  echo This can take several minutes.
  echo.
  call install_gpu.bat
  if errorlevel 1 goto :setup_failed
)

echo.
echo Starting Character Mosaic Censor...
call run.bat
if errorlevel 1 goto :run_failed
exit /b 0

:setup_failed
echo.
echo Setup failed.
echo Review the message above, or run diagnose.bat from a terminal.
pause
exit /b 1

:run_failed
echo.
echo The application could not be started.
echo Check startup_error.log or run diagnose.bat.
pause
exit /b 1
