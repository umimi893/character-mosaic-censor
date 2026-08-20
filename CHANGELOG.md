# Changelog

## 0.3.1

公開・共有を意識したリポジトリ品質・パッケージング改善。

### Repository quality

- `LICENSE`、`.editorconfig`、`.gitattributes` を追加。
- `docs/ARCHITECTURE.md` と `docs/RELEASE_CHECKLIST.md` を追加。
- `pyproject.toml` にREADME・classifiers・keywords・pytest設定を追加。
- GUI entry pointを`project.gui-scripts`へ分離。
- `python -m character_mosaic` とCLI `--version` を追加。
- 肥大化していた`pipeline.py`を公開Facade化し、config / processor / logging / review / storageへ責務分離。
- README/ATTRIBUTIONを第三者へ共有できる体裁へ整理。

- checkout直後の `python -m pytest` がeditable installなしで動くようpytest設定を追加。
- CLI/parserの回帰テストを追加し、テスト数を36へ拡張。

## 0.3.0

完成版候補に向けた処理・GUI・ファイル安全性の安定化。

### 検出 / Recall

- 全体検出 + 2x2/3x3重複タイル + 左右反転TTAを維持。
- TTA/タイル間の重複BBoxを単純に捨てず、IoU/IoSで同一候補を判定して**BBoxのunionを採用**。
  - 低confidence側のBBoxが広い場合もモザイク範囲を失わない。
- BBoxを画像境界へclipし、不正サイズのBBoxを除外。
- Detector/Pipeline設定値の検証を追加。
- 推論エンジンのモデルロード等で継続不能なRuntimeErrorが出た場合、全画像で同じ失敗を繰り返さずバッチを停止。

### ファイル安全性

- 変更画像、未変更コピー、Review HTML/manifestを**一時ファイル + `os.replace`**で原子的に保存。
  - 強制終了や書き込みエラー時に半端な出力を残しにくくした。
- 出力/Reviewフォルダが入力フォルダの親になる危険な構成を拒否。
- 実行前に出力先の書き込み可否を確認。
- 再エンコード時もPNG text（`parameters`等）、ICC/DPI、EXIF/XMPを可能な範囲で保持。
- EXIF Orientation適用後の二重回転を防止。
- RGBAモザイクで半透明alphaを不必要に増加させる合成処理を修正。

### Review / ログ

- `review/manifest.json` を追加。
- 停止→再実行や既存出力Skip後も、以前のReview項目が `review/index.html` から消えない。
- 上書き再検出でReview不要になった画像は古いReview画像・manifest項目を削除。
- JSONLを実行終了時まとめ書きから、**画像ごとにflush/fsyncするストリーミングログ**へ変更。
- JSONLに `run_start` / `image` / `run_end` を記録。
- 実際のモザイク適用範囲 `censor_boxes`、fatal error、各種集計を記録。

### GUI

- 巨大画像を各推論パスごとにフル解像度でQtへ送る処理を廃止。
  - Preview画像だけ最大1600pxへ縮小し、BBox座標は原寸座標系のまま正確に表示。
  - 高速GPU時のPreview Signalを約8fpsへ抑制し、UIイベント詰まりを軽減。
- 画像走査中はindeterminate progress、走査完了時点で総枚数を即表示。
- 詳細設定ダイアログを追加。
  - 2x2/3x3開始サイズ、Overlap、Model IoU、Merge IoU/IoS、未検出Review、Previewサイズ、JPEG品質。
- QSettingsでフォルダ・主要設定・詳細設定・ウィンドウ位置・Splitter位置を保存。
- 出力/Review/今回のログを開くボタンを追加。
- CUDA provider、ONNX Runtime、Python、Provider一覧を表示。
- CUDAが存在するだけでなく実際の選択先かを判定し、`ONNX_MODE=cpu`等のCPU強制も警告。

### 診断

- `diagnose.bat` / `diagnose.py` を追加。
- `diagnose.bat --model-test` で実際に `censor_detect_v1.0_s` をロードして1回推論できる。

### テスト

- Core regression tests: 34 tests。
- GUIコードを含む `compileall` を実施。
- GitHub Actionsは追加していない。
