@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo Python Launcher (py) が見つかりません。
  echo Python 3.10 - 3.13 をインストールしてから再実行してください。
  pause
  exit /b 1
)
if not exist .venv (
  py -3.11 -m venv .venv 2>nul || py -3.12 -m venv .venv 2>nul || py -3.13 -m venv .venv 2>nul || py -3.10 -m venv .venv 2>nul
)
if not exist .venv\Scripts\python.exe (
  echo Python 3.10 - 3.13 が見つからず、.venvを作成できませんでした。
  echo 推奨は Python 3.11 です。
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,14) else 1)"
if errorlevel 1 (
  echo この.venvのPythonは未対応です。3.10 - 3.13で作り直してください。
  pause
  exit /b 1
)
python -m pip install --upgrade pip
rem CPU版ONNX RuntimeとGPU版が同じ環境に混在しないように先に整理する。
python -m pip uninstall -y onnxruntime onnxruntime-gpu >nul 2>nul
python -m pip install -r requirements-gpu.txt
if errorlevel 1 (
  echo GPU依存の導入に失敗しました。
  pause
  exit /b 1
)
python -m pip check
if errorlevel 1 (
  echo 依存関係に競合があります。上のpip check結果を確認してください。
  pause
  exit /b 1
)
python -c "import site,pathlib; p=pathlib.Path(site.getsitepackages()[0])/'character_mosaic_local.pth'; p.write_text(str(pathlib.Path.cwd()/'src'), encoding='utf-8'); print('path:', p)"
python diagnose.py
if errorlevel 1 (
  echo ランタイム診断に失敗しました。上の結果を確認してください。
  pause
  exit /b 1
)
python -c "from character_mosaic.detector import get_runtime_info; import sys; info=get_runtime_info(); print('GPU check:', info.display_text); sys.exit(0 if info.using_cuda else 2)"
if errorlevel 2 (
  echo.
  echo WARNING: GPU版を入れましたが CUDAExecutionProvider が実際の推論先として選択されていません。
  echo NVIDIA Driver / ONNX Runtime GPU環境 / ONNX_MODE を確認してください。
  echo diagnose.bat を実行すると詳細を表示できます。
  pause
  exit /b 2
)
echo.
echo GPU版の導入完了。run.bat で起動できます。
echo モデルまで事前確認する場合: diagnose.bat --model-test
pause
