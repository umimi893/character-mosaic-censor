@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set "PYTHON="
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
if not defined PYTHON (
  py -3.11 --version >nul 2>&1 && set "PYTHON=py -3.11"
)
if not defined PYTHON (
  python --version >nul 2>&1 && set "PYTHON=python"
)
if not defined PYTHON (
  echo Python が見つかりません。先に install_gpu.bat を実行してください。
  pause
  exit /b 1
)

echo.
echo ================================================
echo Character Mosaic Censor - Development Benchmark
echo ================================================
echo.
echo ベンチマークしたい実画像フォルダを指定してください。
echo 一時出力は自動削除され、結果だけ benchmark_results に残ります。
echo.
set /p "INPUT=画像フォルダ: "
if "%INPUT%"=="" exit /b 1

set /p "LIMIT=測定枚数 [500 / 0=全部]: "
if "%LIMIT%"=="" set "LIMIT=500"

echo.
echo 開始します。処理中はこのウィンドウを閉じないでください。
echo.
%PYTHON% tools\benchmark_run.py --input "%INPUT%" --limit %LIMIT%
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (
  echo 完了しました。
  echo benchmark_results フォルダの最新 JSON をChatGPTに渡してください。
) else (
  echo ベンチマークがエラー終了しました。終了コード: %RC%
)
echo.
pause
exit /b %RC%
