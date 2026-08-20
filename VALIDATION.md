# v0.3.1 Validation

このファイルは配布物に対して行ったローカル検証の記録です。

## PASS

- `pytest`: 36/36 pass
- `python -m compileall`: pass
- `git diff --check`: pass
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

## Windows / RTX 5090 gate

この開発環境ではWindows GUI・NVIDIA driver・CUDAExecutionProvider・実際の `detect_censors` モデル推論は起動できないため、以下はWindows実機で `diagnose.bat --model-test` により確認する項目です。

1. PySide6 GUI起動
2. `CUDAExecutionProvider` が選択されること
3. RTX 5090名が表示されること
4. `censor_detect_v1.0_s` の初回ロードと推論成功
5. 実使用画像セットでRecall/false positiveを測定

ソフトウェアの回帰・ファイル安全性・バッチ制御は上記PASS項目まで検証済みですが、検出精度の最終保証には実際の画像分布が必要です。
