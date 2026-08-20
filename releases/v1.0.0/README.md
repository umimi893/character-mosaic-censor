# Character Mosaic Censor v1.0.0

This directory contains the frozen v1.0.0 source package.

- `character-mosaic-censor-1.0.0-source.zip` — canonical v1.0.0 source snapshot
- `SHA256SUMS.txt` — integrity checksum

Validation before publishing:

- 83/83 pytest tests passed
- Python compileall passed
- wheel build passed locally
- development benchmark UI is disabled in the public build

Windows quick start after extracting the source archive:

1. Run `START_HERE.bat`.
2. On first launch it runs the GPU setup and then opens the GUI.
3. Windows may show an unsigned batch-file publisher warning; choose Run if you trust this repository copy.

The repository `main` branch is intentionally left untouched as a rollback point. The release branch is `release/v1.0.0`.
