# Character Mosaic Censor

**Character Mosaic Censor** is a Windows desktop application that automatically detects and censors sensitive anatomical regions in anime, CG, and AI-generated character images. Processing is performed locally on your PC, with a review workflow for uncertain or suspicious results.

**Version:** 1.2.0  
**Platform:** Windows 10 / 11  
**Recommended:** Python 3.11 + NVIDIA GPU

## Features

- PySide6 desktop GUI with Japanese / English display switching
- Large live preview for original, detected, **body-analysis**, and censored images
- Anime/CG detection powered by `dghs-imgutils`
- Multi-signal body-region reasoning using person/head/face/eye BBoxes plus DWPose body keypoints
- Candidate-level `KEEP` / `REVIEW` / `SUPPRESS` decisions with visible evidence
- Full-frame detection plus large-image tiled inference and retry passes
- Mosaic / Blur / Black censor modes
- Low-confidence Review output and manual-review quarantine for suspicious detections
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
3. Adjust censor mode or detection settings only when necessary.
4. Click **Run**.
5. Use **Body analysis** in the preview to inspect detected people/body parts and candidate decisions.
6. Check Review/manual-review items before publishing or distributing the results.

Supported input formats include PNG, JPEG, and WebP.

## Body-region reasoning

Version 1.2 expands the anatomy check into a body-region reasoning layer. The normal censor detector still runs first. For every candidate, the application can then gather additional evidence from:

- anime person BBoxes,
- head BBoxes,
- face BBoxes,
- eye BBoxes,
- DWPose shoulder/hip/knee and other body keypoints,
- derived pelvis-safe, knee, and armpit regions,
- the original detector confidence and inference source.

The body layer does **not** treat a face, head, knee, or any other body part as an unconditional exclusion zone. Candidate evidence is combined into one of three decisions:

- **KEEP** — retain the candidate and censor it.
- **REVIEW** — retain/censor it, but route the image through Review because the body evidence is ambiguous.
- **SUPPRESS** — remove only a strong hard-negative candidate.

### Recall-first safety rules

- A candidate near a reliable pelvis is protected.
- Pelvis evidence from **another person** can protect a candidate even if it overlaps a different person's face/head. This is important for close-contact and oral compositions.
- Face/head overlap by itself is never enough to auto-suppress. It becomes a Review signal.
- A strongly confirmed eye+face+head overlap can be suppressed as an obvious facial false positive.
- Knee/armpit suppression is used only when reliable pose evidence agrees and the candidate is clearly separated from the pelvis.
- Missing or weak body information keeps the original detector result.
- If an auxiliary helper model fails, that helper is disabled for the rest of the batch and processing continues with the remaining evidence.

For troubleshooting, set `CMC_ANATOMY_FILTER=0` before launch to disable the extra body reasoning and return to the base detector behavior.

The first run that needs a helper may download additional upstream person/head/face/eye/pose model files. Images are still processed locally; only upstream model files may be downloaded.

## Body analysis preview

The **Body analysis** view can display:

- person, head, face, and eye BBoxes,
- pose skeleton lines and keypoints,
- pelvis-safe, knee, and armpit regions,
- candidate BBoxes colored by `KEEP`, `REVIEW`, or `SUPPRESS`.

Click a candidate in this view to see its positive and negative evidence. JSONL logs also record `anatomy_filter_status`, `body_regions`, `pose_points`, `pose_edges`, `candidate_evidence`, suppressed candidates, and suppression reasons.

## Review behavior

The detector is intentionally recall-oriented, but no automatic detector is perfect. Low-confidence detections and body-analysis `REVIEW` decisions can be written to a Review folder, and suspicious detection-count results can be isolated for manual checking.

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
