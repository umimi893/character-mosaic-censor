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
rem GPU版からCPU版へ切り替える場合のRuntime混在を防ぐ。
python -m pip uninstall -y onnxruntime-gpu >nul 2>nul
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo 依存ライブラリのインストールに失敗しました。
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
echo.
echo CPU版インストール完了。run.bat で起動できます。
echo RTX GPUを使う場合は install_gpu.bat を実行してください。
pause
