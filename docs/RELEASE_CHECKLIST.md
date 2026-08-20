# Release checklist

This project currently avoids GitHub Actions by design. Run release validation locally.

## Source validation

- [ ] `python -m pytest -q`
- [ ] `python -m compileall -q src diagnose.py run_cli.py run_gui.py`
- [ ] `git diff --check`
- [ ] Confirm `git status` contains no generated outputs, models, virtual environments, or logs
- [ ] Confirm README and CHANGELOG version match `character_mosaic.__version__`

## Windows validation

- [ ] Create a clean `.venv` with Python 3.11
- [ ] Run `install_gpu.bat`
- [ ] Run `diagnose.bat`
- [ ] Confirm `CUDAExecutionProvider` is selected
- [ ] Run `diagnose.bat --model-test`
- [ ] Launch `run.bat`
- [ ] Verify start/stop, progress, preview switching, and settings persistence
- [ ] Process a mixed PNG/JPEG/WebP folder
- [ ] Verify output subfolders, Review HTML, manifest, and JSONL logs

## Detection-quality gate

Before calling a build production-ready, evaluate a representative real-image set and record:

- missed target regions
- false positives
- low-confidence Review volume
- small-object performance
- partially occluded cases
- long-shot and high-resolution cases

If misses remain systematic, improve or add the detector rather than hiding the problem with UI changes.

## Packaging

- [ ] Build from a clean checkout
- [ ] Do not commit downloaded model weights
- [ ] Do not commit `.venv`, `output`, `review`, or runtime logs
- [ ] Include `README.md`, `CHANGELOG.md`, `VALIDATION.md`, `ATTRIBUTION.md`, and `LICENSE`
- [ ] Record the archive SHA-256 when publishing a binary/source archive
