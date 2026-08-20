# v0.4.0 Validation

このファイルは配布物に対して行ったローカル検証の記録です。

## PASS

- `pytest`: 48/48 pass
- `python -m compileall`: pass
- clean checkout import path: `python -m pytest` works without editable install via `pyproject.toml` pytest config
- project metadata parse (`pyproject.toml`): pass
- wheel build (`pip wheel --no-deps --no-build-isolation`): pass
- wheel contents: package entry points and `LICENSE` included
- mixed-format batch smoke: PNG / JPEG / WebP pass
- 3200px image processing smoke: pass
- subfolder preservation: pass
- output/review rescan exclusion: pass
- low-confidence censor + Review image: pass
- persistent Review manifest + HTML generation: pass
- JSONL run_start/image/run_end logging: pass
- atomic output writes: no leaked temp files in smoke
- corrupt image: non-fatal, batch continues
- detector runtime failure: fatal, batch stops after first failure
- no-detection unchanged copy: original bytes preserved
- expected-person count mismatch: original + annotated copies isolated under `_manual_review`
- shape mask: detector-box corners remain uncensored while the center receives the selected effect
- zero-result retry: horizontal/vertical flip and 90/180/270-degree coordinate mapping verified
- omitted output path: `<input>/_censored` selected automatically in GUI and CLI
- Japanese / English control-panel, preview, and advanced-settings labels: pass
- language selection persistence through `QSettings`: pass
- `run.bat`: GUI remains responsive through `pythonw.exe`, no backing `cmd.exe` remains

## Windows / RTX 5090

現在のWindows実機で以下を確認済みです。

1. PySide6 GUI起動
2. `CUDAExecutionProvider` が選択されること
3. RTX 5090名が表示されること
4. `censor_detect_v1.0_s` の初回ロードと推論成功
5. 実使用画像セットでのRecall/false positive測定は継続課題

ソフトウェアの回帰・ファイル安全性・バッチ制御は上記PASS項目まで検証済みですが、検出精度の最終保証には実際の画像分布が必要です。
