# Character Mosaic Censor

**Character Mosaic Censor** is a Windows desktop application that automatically detects and censors sensitive anatomical regions in anime, CG, and AI-generated character images. Processing is local, with visual body-analysis diagnostics, rerunnable output, and an optional local experience system focused on reducing recurring false positives.

**Version:** 1.4.0  
**Platform:** Windows 10 / 11  
**Recommended:** Python 3.11 + NVIDIA GPU

## Features

- PySide6 desktop GUI with Japanese / English display switching
- Large live preview for original, detected, **body-analysis**, and censored images
- Drag-and-drop one image onto the preview for an immediate single-image test
- Anime/CG censor detection powered by `dghs-imgutils`
- Person/head/face/eye BBoxes plus DWPose skeleton analysis
- **Geometry v2** suppression for upper-back/shoulder-blade, torso, armpit, thigh, and lower-leg false positives
- Directional groin-positive geometry to protect plausible targets without broadly protecting the lower back/waist
- Candidate-level `KEEP` / `SUPPRESS` evidence with clickable diagnostics
- Conservative **Negative Memory** for recurring near-identical false positives
- Local SQLite **Experience Store** that can remember compact candidate evidence while the GUI is used
- Resumable mining of noisy legacy image folders and ZIP archives without modifying the source material
- Automatic duplicate/corrupt-file handling and GOLD / SILVER / QUARANTINE pseudo-label tiers
- Optional idle-GPU gating for large background mining runs
- Full-frame detection plus large-image tiled inference and production retry passes
- Mosaic / Blur / Black censor modes
- GUI reruns overwrite generated results by default
- Low-confidence Review output and manual-review quarantine for excessive final detection counts
- Safe temporary-file writes and JSONL run history
- NVIDIA CUDA / ONNX Runtime diagnostics
- Normal processing CLI plus a standalone corpus-mining CLI
- No cloud upload of the images being processed

## Quick start on Windows

1. Install **64-bit Python 3.11** from python.org. During installation, enable the Python Launcher if offered.
2. Download or clone this repository.
3. Run **`START_HERE.bat`**.
4. On the first launch, the GPU environment is created in `.venv`; after setup, the GUI starts.

`START_HERE.bat` reads the application version directly from `pyproject.toml`.

If the GUI does not open, check `startup_error.log` in the repository folder or run `diagnose.bat` from a terminal.

## Basic usage

1. Select the folder containing source images.
2. Leave the output lock off to use `<input>\_censored`, or lock/select a custom output folder.
3. Keep **Overwrite existing outputs** enabled for normal reruns.
4. Leave **Remember candidate evidence while processing** enabled if you want the local false-positive memory to grow automatically.
5. Click **Run**.
6. Use **Body analysis** to inspect skeleton/body regions and candidate decisions.
7. Check Review/manual-review items when the output is important.

Normal batch input supports PNG, JPEG, and WebP. The preview drop target also accepts BMP for single-image testing.

## Geometry v2: back, armpit, and leg false positives

Version 1.4 adds a second pose-geometry layer aimed specifically at common visible false mosaics.

From reliable DWPose landmarks it derives:

- a shoulder/hip torso polygon for torso and upper-back/shoulder-blade candidates,
- armpit zones using shoulder-to-hip position plus arm context,
- pose-aligned thigh tubes from hip to knee,
- pose-aligned lower-leg tubes from knee to ankle,
- a narrow directional groin-positive region extending below the hip line.

This is intentionally more anatomical than a single axis-aligned body BBox. Bent or diagonal legs remain representable because limb decisions use distance from the actual pose bone.

### Geometry safety rules

Geometry v2 is intentionally allowed to do nothing when evidence is uncertain:

- A candidate that cannot be assigned to a reliable person is kept.
- Missing pose landmarks remove the corresponding suppression rule instead of guessing.
- If a candidate is matched to multiple people, all matched people must support a hard-negative body classification before geometry v2 suppresses it.
- Reliable pelvis evidence from another person protects close-contact candidates.
- A candidate inside the directional groin-positive zone is kept before torso/armpit/leg suppression is considered.
- Existing face/head and lower-level body-analysis safeguards remain in the pipeline.

The **Body analysis** preview shows the new torso-v2, armpit, thigh, and lower-leg regions so the reason for a suppression is visible rather than hidden.

## Automatic experience capture

The desktop GUI enables **Remember candidate evidence while processing** by default.

When enabled, successful normal processing can write a compact local record to:

```text
%LOCALAPPDATA%\CharacterMosaicCensor\learning\
```

The store contains:

- `experience.sqlite3` — source/candidate metadata and evidence,
- small candidate crops for selected high-confidence examples,
- no automatic copy of the original source image.

Learning storage is **best effort**. A full disk, damaged learning database, or crop-save error is not allowed to fail the normal censoring operation.

The core Python/CLI configuration remains side-effect-free by default; automatic capture is a desktop-GUI default and can be turned off in the GUI.

## Negative Memory

Version 1.4 can immediately benefit from repeated high-confidence false-positive examples without waiting for a separately trained neural network.

A candidate may be suppressed by Negative Memory only when:

- several near-identical perceptual fingerprints already exist as **GOLD negative** experience,
- the current candidate has no pelvis/groin-positive evidence,
- the repeated evidence crosses a conservative match threshold.

This mechanism is deliberately asymmetric. A memory-based suppression is **not** promoted back into GOLD training evidence, so one remembered mistake cannot recursively manufacture more trusted evidence and amplify itself.

Negative Memory is intended for recurring visual patterns in a user's own image library, not as a replacement for the normal detector/body reasoning.

## Mining old image libraries

Use **Mine legacy image folders…** in the GUI to point the app at large, messy collections of old generated images.

The corpus does **not** have to be clean training data. It is treated as a mine from which the current detector extracts interesting candidates.

The miner supports:

- recursive PNG / JPEG / WebP discovery,
- images stored inside ZIP files without manually extracting them,
- exact SHA-256 duplicate detection,
- corrupt/unreadable image skipping,
- minimum-size and extreme-size safety checks,
- resumable processing through the Experience Store,
- optional compact candidate-crop storage,
- optional waiting until NVIDIA GPU utilization falls below a selected threshold,
- cooperative stop and later resume.

Source images and ZIP archives are never modified by the miner.

### Pseudo-label tiers

The miner deliberately does not force every candidate into a training label.

- **GOLD negative** — a known hard-negative body/face reason has strong evidence.
- **SILVER** — useful but less trustworthy evidence.
- **QUARANTINE** — conflicting or ambiguous evidence; stored for analysis but not treated as trusted training data.

This matters for noisy legacy libraries containing broken anatomy, text overlays, failed generations, or unusual compositions. Having many images makes it preferable to discard ambiguity rather than pretend an uncertain automatic label is ground truth.

### Mining performance

Production image processing keeps its normal zero-detection flip/rotation retries. Legacy mining does not run those expensive retries after a zero-result image because a zero-result image contributes no hard-negative candidate. Large-image tiling remains available to the miner.

## What v1.4 learning is — and is not

Version 1.4 adds persistent experience, automatic hard-negative mining, and conservative repeated-negative memory. It does **not** silently train and auto-promote a new deep neural network in the background.

That distinction is intentional: automatically training a verifier entirely from its own pseudo-labels without a trusted evaluation set can reinforce mistakes. The Experience Store is designed so a future learned verifier or optional semantic hair model can consume accumulated GOLD/SILVER data without changing the stable production detector first.

No new mandatory ML framework was added in v1.4, so existing installations do not need a PyTorch-based segmentation stack just to gain the new geometry/mining functionality.

## Single-image drag and drop

Drag one image from Explorer directly onto the large preview area to process only that image with the current settings.

- If the output folder is **not locked**, the dropped image is written to `<image folder>\_censored`.
- If the output folder is explicitly locked, the locked output folder is used.
- A custom Review folder is respected; otherwise the usual sibling `review` folder is used.
- The same detector, body analysis, Negative Memory, logging, Review logic, and save pipeline are used as a normal folder run.

## Rerunning the same folder

**Overwrite existing outputs** is enabled by default in the GUI. Older settings that inherited the former automatic `overwrite=false` default are migrated once; an explicit later OFF choice is preserved.

When a rerun changes the result:

- the output image is replaced with the latest result,
- a stale Review image is removed when it is no longer needed,
- stale `_manual_review/edit`, `reference_bbox`, and `auto_censored` files are removed when no longer required,
- JSONL logs are not overwritten; each run keeps its own history file.

## Body-region reasoning pipeline

The normal detector is still the candidate source. The default desktop processing chain is conceptually:

```text
Anime censor detector
  -> person/head/face/eye + DWPose anatomy reasoning
  -> v1.3 torso/body reasoning
  -> geometry v2 hard-negative reasoning
  -> conservative Negative Memory (when learning is enabled)
  -> KEEP / SUPPRESS
  -> censor output
```

For troubleshooting, set `CMC_ANATOMY_FILTER=0` before launch to disable the extra anatomy helper logic and return toward base detector behavior.

The first run that needs an upstream helper may download person/head/face/eye/pose model files. Images themselves remain local.

## Review behavior

The detector is recall-oriented, but no automatic detector is perfect. Low-confidence retained detections can be written to Review, and excessive final detection counts can be isolated for manual checking.

False positives and missed regions remain possible, especially with unusual poses, occlusion, extremely small targets, or images outside the detector's training distribution.

## Installation options

### NVIDIA GPU

```text
install_gpu.bat
diagnose.bat
run.bat
```

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

Normal processing:

```bat
run_cli.bat "D:\input" "D:\output" --review "D:\review"
```

The output argument can be omitted to use `<input>\_censored`.

Installed entry points:

```text
character-mosaic --help
character-mosaic --version
character-mosaic-gui
character-mosaic-mine "F:\old_images" "D:\archives"
```

Useful miner options:

```text
character-mosaic-mine "F:\old_images" --max-gpu-util 20
character-mosaic-mine "F:\old_images" --no-idle-wait
character-mosaic-mine "F:\old_images" --max-images 5000
character-mosaic-mine "F:\old_images" --no-zip
```

## Privacy

Image processing and the Experience Store are local. The application does not upload source images to an external service. Dependencies may use network access to download upstream inference model files on first use.

## Development

```text
python -m pytest -q
python -m compileall -q src diagnose.py run_cli.py run_gui.py
```

`test_local.bat` runs the local regression suite on Windows. GitHub Actions are not required for normal use.

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
