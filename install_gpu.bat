@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo Python Launcher ^(py^) was not found.
  echo Install Python 3.10 - 3.13 and run this installer again.
  pause
  exit /b 1
)
if not exist .venv (
  py -3.11 -m venv .venv 2>nul || py -3.12 -m venv .venv 2>nul || py -3.13 -m venv .venv 2>nul || py -3.10 -m venv .venv 2>nul
)
if not exist .venv\Scripts\python.exe (
  echo Python 3.10 - 3.13 was not found and .venv could not be created.
  echo Python 3.11 is recommended.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,14) else 1)"
if errorlevel 1 (
  echo The Python version in .venv is unsupported. Recreate it with Python 3.10 - 3.13.
  pause
  exit /b 1
)
python -m pip install --upgrade pip
rem Avoid mixing CPU and GPU ONNX Runtime packages in the same environment.
python -m pip uninstall -y onnxruntime onnxruntime-gpu >nul 2>nul
python -m pip install -r requirements-gpu.txt
if errorlevel 1 (
  echo Failed to install GPU dependencies.
  pause
  exit /b 1
)
python -m pip check
if errorlevel 1 (
  echo Dependency conflicts were found. Check the pip check output above.
  pause
  exit /b 1
)
python -c "import site,pathlib; p=pathlib.Path(site.getsitepackages()[0])/'character_mosaic_local.pth'; p.write_text(str(pathlib.Path.cwd()/'src'), encoding='utf-8'); print('path:', p)"
python diagnose.py
if errorlevel 1 (
  echo Runtime diagnostics failed. Check the output above.
  pause
  exit /b 1
)
python -c "from character_mosaic.detector import get_runtime_info; import sys; info=get_runtime_info(); print('GPU check:', info.display_text); sys.exit(0 if info.using_cuda else 2)"
if errorlevel 2 (
  echo.
  echo WARNING: CUDAExecutionProvider was not selected for inference.
  echo Check the NVIDIA driver, ONNX Runtime GPU environment, and ONNX_MODE.
  echo Run diagnose.bat for details.
  pause
  exit /b 2
)
echo.
echo GPU installation complete. Run run.bat to launch the app.
echo To test the model: diagnose.bat --model-test
pause
