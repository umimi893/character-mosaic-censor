# Character Mosaic Censor

**Character Mosaic Censor** is a Windows desktop application that automatically detects and censors sensitive anatomical regions in anime, CG, and AI-generated character images. Processing is performed locally on your PC, with a review workflow for uncertain or suspicious results.

**Version:** 1.0.0  
**Platform:** Windows 10 / 11  
**Recommended:** Python 3.11 + NVIDIA GPU

## Features

- PySide6 desktop GUI with Japanese / English display switching
- Large live preview for original, detected, and censored images
- Anime/CG detection powered by `dghs-imgutils`
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

Windows may display an **Unknown publisher / Open File - Security Warning** for the batch files because they are not code-signed. If you downloaded this repository from the official GitHub page and trust the copy, choose **Run**.

If the GUI does not open, check `startup_error.log` in the repository folder or run `diagnose.bat` from a terminal.

## Basic usage

1. Select the folder containing source images.
2. Choose an output folder, or use the automatically suggested `_censored` folder.
3. Adjust censor mode or detection settings only when necessary.
4. Click **Run**.
5. Check Review/manual-review items before publishing or distributing the results.

Supported input formats include PNG, JPEG, and WebP.

## Review behavior

The detector is intentionally recall-oriented, but no automatic detector is perfect. Low-confidence detections can be written to a Review folder, and suspicious detection-count results can be isolated for manual checking.

For important batches, review the uncertain images before considering the output final. False positives and missed regions are both possible, especially with unusual poses, occlusion, very small targets, or images outside the detector's training distribution.

## Installation options

### NVIDIA GPU

```text
install_gpu.bat
diagnose.bat
run.bat
```

`install_gpu.bat` creates a local `.venv` and installs the GPU dependencies. The first detector run may download upstream model data if it is not already cached.

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
