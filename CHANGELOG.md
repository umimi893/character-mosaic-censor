# Changelog

## 1.1.0

Automatic anatomy-aware false-positive suppression.

- Added a conservative DWPose-based sanity check after the normal censor detector.
- Added automatic suppression for candidates that are reliably placed near a knee or armpit while clearly separated from the pelvis.
- Kept recall-first fail-open behavior: missing people, missing hips, weak/ambiguous poses, overlapping-person ambiguity, or helper-model errors keep the original detector candidate.
- Added automatic run-wide fallback to the original detector when the anatomy helper cannot load, so a missing helper model does not stop a batch or retry on every image.
- Added `CMC_ANATOMY_FILTER=0` as an emergency disable switch without adding another normal-user UI setting.
- Added per-image JSONL diagnostics for anatomy-filter status, suppressed detector boxes, and suppression reasons.
- Added GUI counters showing how many obvious body-position false-positive candidates were removed.
- Added regression coverage for pelvis protection, knee/armpit suppression, ambiguous multi-person scenes, helper failures, simple detector signatures, and the emergency override.

## 1.0.1

Windows GUI hotfix release.

- Fixed `pythonw.exe` inference crashes caused by missing `sys.stdout` / `sys.stderr` when third-party inference code writes progress output.
- Fixed the output-folder follow behavior so `<input>\_censored` updates automatically when the input folder changes unless the user explicitly locks the output path.
- Added an `Output folder lock` control and safe migration for previously customized output paths.
- Fixed an initialization-order regression in the output-lock UI.
- Forced a complete dark application palette so Windows light mode cannot leak white native backgrounds behind light text.
- Added regression coverage for console-less standard streams, output path migration, and OS-theme background leaks.

## 1.0.0

First public-ready release.

- Promoted the project metadata from beta to stable.
- Added a single `START_HERE.bat` entry point for first-time Windows setup and launch.
- Added visible startup error reporting through a Windows dialog and `startup_error.log` instead of failing silently under `pythonw.exe`.
- Kept the Japanese / English desktop UI, live preview, tiled/retry detection, Review workflow, manual-review quarantine, GPU diagnostics, CLI, and safe file-writing behavior developed during the 0.x series.
- Removed internal release artifacts and development-only release notes from the repository root.
- Reworked the README for end users and public distribution.

## 0.4.0

- Added Japanese / English live UI switching and persisted language selection.
- Improved folder UX and settings persistence.
- Added expected-person-count review handling and separated editable manual-review originals from detector-reference images.
- Added retry transforms for zero-detection cases and improved large-image detection.
- Improved GPU/runtime setup and local regression coverage.

## 0.3.1

- Added repository metadata, licensing/attribution documents, packaging metadata, architecture documentation, and CLI/GUI entry points.
- Split pipeline responsibilities and improved local validation.

## 0.3.0

- Added recall-oriented full/tiled/TTA detection merging.
- Added atomic file writes, metadata preservation, persistent Review manifests, crash-tolerant JSONL logs, diagnostics, and GUI performance improvements.
