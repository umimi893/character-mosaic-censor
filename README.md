# Character Mosaic Censor

**Character Mosaic Censor** is a Windows desktop application that automatically detects and censors sensitive anatomical regions in anime, CG, and AI-generated character images. Processing is performed locally on your PC, with visual body-analysis diagnostics and a Review workflow for uncertain detector results.

**Version:** 1.3.0  
**Platform:** Windows 10 / 11  
**Recommended:** Python 3.11 + NVIDIA GPU

## Features

- PySide6 desktop GUI with Japanese / English display switching
- Large live preview for original, detected, **body-analysis**, and censored images
- Drag-and-drop one image onto the preview for an immediate single-image test
- Anime/CG detection powered by `dghs-imgutils`
- Multi-signal body-region reasoning using person/head/face/eye BBoxes plus DWPose body keypoints
- Derived pelvis, knee, armpit, and torso/back regions for false-positive suppression
- Candidate-level `KEEP` / `SUPPRESS` evidence with clickable diagnostics
- Full-frame detection plus large-image tiled inference and retry passes
- Mosaic / Blur / Black censor modes
- GUI reruns overwrite generated results by default so settings can be adjusted and the same folder processed again without deleting prior output folders
- Low-confidence Review output and manual-review quarantine for suspicious detection counts
- Recursive folder processing while preserving subfolder structure
- Safe temporary-file writes to reduce incomplete/corrupted outputs
- NVIDIA CUDA / ONNX Runtime diagnostics
- CLI using the same processing pipeline
- No cloud upload of the images being processed

## Quick start on Windows

1. Install **64-bit Python 3.11** from python.org. During installation, enable the Python Launcher if offered.
2. Download or clone this repository.
3. Run **`START_HERE.bat`**.
4. On the first launch, the GPU environment is created in `.venv`; after setup, the GUI starts.

`START_HERE.bat` reads the current application version directly from `pyproject.toml`, so the displayed version stays synchronized with releases.

If the GUI does not open, check `startup_error.log` in the repository folder or run `diagnose.bat` from a terminal.

## Basic usage

1. Select the folder containing source images.
2. Leave the output lock off to automatically use `<input>\_censored`, or lock/select a custom output folder when needed.
3. Keep **Overwrite existing outputs** enabled for normal reruns. Re-running refreshes output, Review, and manual-review artifacts; JSONL logs remain as separate history files.
4. Adjust censor mode or detection settings only when necessary.
5. Click **Run**.
6. Use **Body analysis** in the preview to inspect detected people/body parts and candidate decisions.
7. Check Review/manual-review items before publishing or distributing the results.

Supported input formats include PNG, JPEG, WebP, and BMP for preview drag-and-drop.

## Single-image drag and drop

Drag one image from Explorer directly onto the large preview area to process only that image with the current settings.

- If the output folder is **not locked**, the dropped image is written to `<image folder>\_censored`.
- If the output folder is explicitly locked, the locked output folder is used.
- A custom Review folder is respected; otherwise the usual sibling `review` folder is used.
- The same detector, body analysis, GPU diagnostics, logging, Review logic, and save pipeline are used as a normal folder run.

This is intended for quickly retesting a troublesome image after changing thresholds or body-analysis behavior.

## Rerunning the same folder

Version 1.3 changes the normal GUI workflow so generated results can be refreshed without manually deleting output folders first.

**Overwrite existing outputs** is enabled by default. On the first v1.3 launch, older settings that inherited the previous automatic `overwrite=false` default are migrated once to the new default. After that migration, if you deliberately turn overwrite off, that choice is preserved.

When a rerun changes the result:

- the output image is replaced with the latest result,
- a stale Review image is removed when the image no longer needs Review,
- stale `_manual_review/edit`, `reference_bbox`, and `auto_censored` files are removed when the image is no longer over-detected,
- logs are **not** overwritten; every run keeps its own JSONL history file.

## Body-region reasoning

The normal censor detector runs first. For every candidate, the application can gather additional evidence from:

- anime person BBoxes,
- head BBoxes,
- face BBoxes,
- eye BBoxes,
- DWPose shoulder/hip/knee and other body keypoints,
- derived pelvis-safe, knee, armpit, and torso/back regions,
- the original detector confidence and inference source.

The final product policy uses two main decisions:

- **KEEP** — retain the candidate and censor it.
- **SUPPRESS** — remove a candidate that has strong body-region evidence of being a false positive.

The lower-level anatomy pass can still generate an internal `REVIEW` signal, but in v1.3 a body-analysis Review candidate with **no reliable pelvis evidence** is treated as a false positive and becomes `SUPPRESS`. General low-confidence detector Review remains unchanged.

### Safety rules

- A reliable pelvis remains strong positive evidence.
- Pelvis evidence from **another person** can protect a candidate even if it overlaps a different person's face/head/torso. This is important for close-contact and oral compositions.
- Strong eye+face+head overlap can suppress an obvious facial false positive.
- Knee/armpit suppression is used only when reliable pose evidence agrees.
- A conservative shoulder-to-hip **torso/back BBox** can suppress clear waist/back/torso false positives.
- The torso/back BBox is skipped for nearly-horizontal bodies because an axis-aligned torso estimate becomes unsafe in those poses.
- Missing or weak body information keeps the original detector result.
- If an auxiliary helper model fails, that helper is disabled for the rest of the batch and processing continues with the remaining evidence.

For troubleshooting, set `CMC_ANATOMY_FILTER=0` before launch to disable the extra body reasoning and return to the base detector behavior.

The first run that needs a helper may download additional upstream person/head/face/eye/pose model files. Images are still processed locally; only upstream model files may be downloaded.

## Body analysis preview

The **Body analysis** view can display:

- person, head, face, and eye BBoxes,
- pose skeleton lines and keypoints,
- pelvis-safe, knee, armpit, and torso/back regions,
- candidate BBoxes colored by decision.

The preview hint uses violet for the torso/back region. Click a candidate to see its positive and negative evidence. JSONL logs also record `anatomy_filter_status`, `body_regions`, `pose_points`, `pose_edges`, `candidate_evidence`, suppressed candidates, and suppression reasons.

## Review behavior

The detector is intentionally recall-oriented, but no automatic detector is perfect. Low-confidence detections can be written to a Review folder, and excessive final detection counts can be isolated for manual checking.

For important batches, review the uncertain images before considering the output final. False positives and missed regions are both possible, especially with unusual poses, occlusion, very small targets, or images outside the detector's training distribution.

## Installation options

### NVIDIA GPU

```text
install_gpu.bat
diagnose.bat
run.bat
```

`install_gpu.bat` creates a local `.venv` and installs the GPU dependencies. The first detector/body-analysis run may download upstream model data if it is not already cached.

To test the actual model load:

```bat
diagnose.bat --model-test
```

### CPU

```text
install.bat
diagnose.bat
run.bat
```

CPU inference is supported but substantially slower.

## CLI

```bat
run_cli.bat "D:\input" "D:\output" --review "D:\review"
```

The output argument can be omitted to use `<input>\_censored`.

Installed entry points:

```text
character-mosaic --help
character-mosaic --version
character-mosaic-gui
```

## Privacy

Image processing runs locally. The application does not require uploading source images to an external service. Network access may still be used by dependencies to obtain upstream model files on first use.

## Development

```text
python -m pytest -q
python -m compileall -q src diagnose.py run_cli.py run_gui.py
```

`test_local.bat` runs the local regression suite on Windows. GitHub Actions are intentionally not required for normal use.

## Project structure

```text
src/character_mosaic/   application source
tests/                  regression tests
docs/ARCHITECTURE.md    architecture notes
models/README.md         model directory notes
```

## License and third-party software

Original project code is currently **all rights reserved**; see [`LICENSE`](LICENSE). Public availability of the source code does not grant additional redistribution or modification rights beyond that license.

Third-party libraries and upstream detection models have their own terms. See [`ATTRIBUTION.md`](ATTRIBUTION.md). No third-party model weights are committed to this repository.
