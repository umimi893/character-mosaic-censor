# Development benchmark

This benchmark is intentionally isolated from the production application so it can be removed before the public release by deleting:

- `benchmark_run.bat`
- `tools/benchmark_run.py`
- this document

No GitHub Actions are used.

## Run on Windows / RTX 5090

1. Switch to the `dev/benchmark-v0.4` branch and update the working copy.
2. Make sure the normal GPU installation already works (`install_gpu.bat` / `run.bat`).
3. Double-click `benchmark_run.bat`.
4. Paste the folder containing representative real images.
5. Use `500` images for the first measurement. Use `0` only when you deliberately want to test the whole folder.
6. Do not run Stable Diffusion, games, video encoding, or another heavy GPU workload during the measurement.

The benchmark creates a temporary `_cmc_benchmark_<timestamp>` folder inside the selected input folder so disk behavior is representative. That temporary folder is deleted automatically after a successful run.

## Results

Results are written under:

`benchmark_results/`

Two files are produced:

- `benchmark_YYYYMMDD_HHMMSS.html` — human-readable dashboard. It opens automatically on Windows.
- `benchmark_YYYYMMDD_HHMMSS.json` — detailed machine-readable data. **Send this JSON file back to ChatGPT for analysis.**

The JSON contains throughput, median/P90/P95/P99, Review/error rates, per-image elapsed time, and—when supported by the tested build—stage timings and Adaptive Detection pass paths. Source locations are stored relative to the selected input folder; absolute local paths are not included.

## Recommended first test

Use 300–500 representative images containing a realistic mixture of resolutions and difficult/easy cases. Avoid selecting only the easiest images because the main goal is to measure the real end-to-end workload.
