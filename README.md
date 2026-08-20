# Character Mosaic Censor

**Character Mosaic Censor** is a Windows desktop application for automatically detecting and censoring sensitive anatomical regions in anime, CG, and AI-generated character images. Processing stays local, and the application is designed around **recall-first detection**: uncertain candidates can still be censored and routed to a review queue instead of being silently ignored.

> **Status:** v0.4.0 / pre-release candidate. Core processing and regression tests are validated locally; representative-image recall validation is still required before calling the detector production-ready.

## Highlights

- Native-looking desktop GUI built with **PySide6**
- Live Japanese / English UI switching with a persisted language choice
- Large live preview showing the image, detector boxes, confidence, and censor region
- `QThread` worker so inference does not freeze the UI
- `dghs-imgutils` `detect_censors` as the default anime/CG detector
- Full-frame + overlapping 2x2 / 3x3 tiled inference
- Zero-detection fallback using horizontal/vertical flips and 90/180/270-degree rotations
- Expected-person count checks with a dedicated manual-review quarantine
- Recall-oriented cross-pass box union/merge
- Mosaic / Blur / Black censor modes
- Low-confidence Review queue with persistent HTML report
- Crash-tolerant JSONL run logs
- Subfolder preservation and output/review rescan protection
- Atomic writes and metadata preservation where supported
- CPU and NVIDIA CUDA/ONNX Runtime diagnostics
- CLI entry point using the same processing pipeline
- Console-free GUI launch through `run.bat`
- No GitHub Actions dependency; validation is local by design

## Application layout

The desktop UI uses a preview-first layout:

```text
┌───────────────────────────────────────────┬─────────────────────────┐
│                                           │ Input / Output / Review │
│                                           │                         │
│             Live image preview            │ Run / Stop              │
│                                           │                         │
│        detector BBox + confidence         │ Detection settings      │
│        expanded censor region             │ Censor settings         │
│        post-process result                │ Runtime / GPU status    │
│                                           │                         │
├───────────────────────────────────────────┴─────────────────────────┤
│ progress / current file / detection state                           │
└─────────────────────────────────────────────────────────────────────┘
```

The preview can switch between **Original / Detection / Censored**, and BBox overlays can be toggled independently.

The right-side language selector switches the visible UI between Japanese and English immediately. Advanced settings are grouped into Basic, Large images, and Duplicate merging tabs, with an explanation under every non-obvious option.

For large images, the preview copy is downscaled while all detection coordinates remain in the source image coordinate system. This avoids sending a full 3K/8K image through Qt after every detector pass.

## Detection strategy

The default detector is `dghs-imgutils==0.19.0` with `censor_detect_v1.0_s`.

The application currently filters to the detector's female genital-region class (`pussy`) by default. That class name comes from the upstream model and is kept internally for compatibility.

### Default recall-oriented settings

| Setting | Default |
|---|---:|
| Detection confidence | `0.12` |
| Expected people per image | `1` |
| Review threshold | `0.30` |
| Model | `v1.0 / standard(s)` |
| Model NMS IoU | `0.70` |
| Padding | `15% + 12px` |
| 2x2 tiled inference | long side `>= 1200px` |
| 3x3 tiled inference | long side `>= 3000px` |
| Tile overlap | `16%` |
| Flip/rotation retry after zero detections | ON |
| Cross-pass merge IoU | `0.45` |
| Cross-pass nested IoS | `0.70` |

Confidence handling:

- `score >= 0.30` → normal censoring
- `0.12 <= score < 0.30` → censor **and** add to Review
- below `0.12` → not returned by the detector

### Cross-pass box merge

The same object may be detected by the full-frame pass, a tile, and a flipped pass. Matching boxes are not resolved by simply keeping the highest-confidence box.

When boxes strongly overlap, the application uses their **union bounding box** and keeps the highest confidence. This is deliberate: if one pass finds a slightly wider region, the censor coverage should not shrink just because another pass scored higher.

## Censor modes

The GUI supports:

- **Mosaic** — default
- **Blur**
- **Black**

The censor region is expanded beyond the raw detector box using both a fixed pixel margin and a box-size ratio margin.
Because the detector provides boxes rather than segmentation contours, effects are applied through an antialiased oval mask inside each expanded box instead of filling all four corners.

## Detection-count mismatch workflow

The GUI accepts the expected number of people per image. If the final `pussy` detection count differs after all retries, the source is copied to `<output>/_manual_review/original/` and an annotated reference is written to `<output>/_manual_review/annotated/`. This quarantine is created independently of the normal confidence-based Review setting.

## Review workflow

Low-confidence detections can be copied into a persistent Review tree.

```text
review/
├─ index.html
├─ manifest.json
└─ <mirrored source folders>/...
```

Review images include:

- raw detector BBox
- class and confidence
- actual expanded censor region

`manifest.json` persists across runs, so stopping and resuming does not make previous Review cards disappear. If an image is reprocessed and no longer qualifies for Review, the stale Review entry is removed.

For recall evaluation, **Review images with no detections** can also be enabled from Advanced Settings. This is off by default because it can produce a very large review queue.

## Logging and failure handling

Each run writes a JSONL file under `logs/`.

Recorded data includes:

- run configuration
- input/output/review paths
- detections and source pass
- raw and expanded censor boxes
- Review state
- skip / cancel / error / fatal error
- per-image processing time
- run summary

Records are flushed and `fsync`'d after each image where supported. If the application or machine stops unexpectedly, completed records remain much easier to inspect.

A corrupt input image is treated as a per-file error and the batch continues. A detector/runtime failure such as a model-load error is treated as fatal so the same broken inference does not repeat across hundreds of files.

## Stop and resume behavior

Stop is cooperative rather than process-kill based.

1. The active inference pass is allowed to return.
2. If the current image has not completed the intended detection pass set, that image is **not** written as a completed output.
3. The batch stops.
4. With overwrite disabled, a new run skips already completed outputs and continues naturally.

## File safety and metadata

The pipeline aims to avoid producing damaged or unnecessarily altered files:

- writes use a temporary sibling followed by `os.replace`
- output/review trees are excluded from input recursion
- unsafe input/output/review parent-child layouts are rejected
- images without detections are copied byte-for-byte when possible
- re-encoded PNG text fields such as Stable Diffusion `parameters` are retained where supported
- ICC, DPI, EXIF, and WebP XMP are retained where Pillow supports them
- EXIF Orientation is removed after pixel normalization to prevent double rotation
- RGBA processing preserves existing alpha instead of alpha-compositing the censored region over itself

## Installation on Windows

Python **3.11** is recommended. Python 3.10-3.13 is accepted by the project metadata.

### NVIDIA GPU installation

```text
1. install_gpu.bat
2. diagnose.bat
3. run.bat
```

The GPU installer uses the upstream `dghs-imgutils[gpu]` extra.

A healthy GPU environment should show information similar to:

```text
CUDA / NVIDIA GeForce RTX 5090
ONNX Runtime <version> / Python 3.11.x
Providers: CUDAExecutionProvider, CPUExecutionProvider
```

To verify the actual detector model load and one test inference:

```bat
diagnose.bat --model-test
```

The first model load may require network access if the upstream model has not been cached yet.

### CPU installation

```text
1. install.bat
2. diagnose.bat
3. run.bat
```

## CLI

The GUI is the primary interface, but the same pipeline is available from the command line.

```bat
run_cli.bat "D:\input" "D:\output" --review "D:\review"
```

The output argument may be omitted; in that case `D:\input\_censored` is created automatically. Use `--people N` to set the expected people count.

Example with more aggressive recall settings:

```bat
run_cli.bat "D:\input" "D:\output" --review "D:\review" ^
  --detect-threshold 0.08 ^
  --auto-threshold 0.30 ^
  --padding 16 ^
  --padding-ratio 0.18
```

Installed entry points:

```text
character-mosaic --help
character-mosaic --version
character-mosaic-gui
```

`python -m character_mosaic` also launches the desktop application.

## Local validation

The repository intentionally does not use GitHub Actions at this stage.

From a checkout with development dependencies available:

```text
python -m pytest -q
python -m compileall -q src diagnose.py run_cli.py run_gui.py
git diff --check
```

On Windows, `test_local.bat` runs the local regression suite.

See:

- [`VALIDATION.md`](VALIDATION.md) — current validation record
- [`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md) — release gate
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — module boundaries and data flow
- [`CHANGELOG.md`](CHANGELOG.md) — release history
- [`ATTRIBUTION.md`](ATTRIBUTION.md) — third-party components

## Repository structure

```text
.
├─ src/character_mosaic/
│  ├─ detector.py
│  ├─ pipeline.py
│  ├─ image_ops.py
│  ├─ types.py
│  ├─ cli.py
│  ├─ gui.py
│  ├─ ui/
│  │  ├─ main_window.py
│  │  ├─ preview_widget.py
│  │  ├─ control_panel.py
│  │  └─ settings_dialog.py
│  └─ workers/
│     └─ batch_worker.py
├─ tests/
├─ docs/
├─ diagnose.py
├─ install.bat
├─ install_gpu.bat
└─ pyproject.toml
```

The GUI, worker, pipeline, detector, and image operations are intentionally separate. A future detector can be introduced without coupling model-specific code to the desktop UI.

## Detector extension

A future custom YOLO implementation should target the existing detector boundary:

```python
class Detector:
    def detect(self, image, progress=None, stop_requested=None) -> list[Detection]:
        ...
```

A custom single-class model becomes justified if representative real images show repeated misses in cases such as:

- closed/low-contrast target appearance
- open appearance
- small target regions
- side or oblique views
- partial occlusion by hair, hands, or other objects
- long-shot compositions

Detector quality cannot be honestly declared complete without testing against the actual image distribution. The application therefore includes no-detection Review, persistent Review state, and detailed logs specifically to support that evaluation.

## Validation status

Current source-level regression validation includes **48 tests** plus compile checks. Windows GUI startup, console-free launch, language switching, and an RTX 5090 model inference pass are verified; representative-image recall evaluation remains required.

See [`VALIDATION.md`](VALIDATION.md) for the exact boundary between verified behavior and pending hardware/model validation.

## Licensing and third-party components

Original application code is currently **all rights reserved**. See [`LICENSE`](LICENSE).

The application depends on third-party software and an upstream detection model with their own license terms. No model weights or third-party source trees are vendored into this repository. See [`ATTRIBUTION.md`](ATTRIBUTION.md).
