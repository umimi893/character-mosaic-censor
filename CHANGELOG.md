# Changelog

## 1.4.0

False-positive geometry v2 and persistent automatic learning memory.

- Added pose-aligned hard-negative geometry for the false positives seen most often in real use: upper back/shoulder-blade area, torso, armpits, thighs, and lower legs.
- Replaced the broad idea of “near the pelvis” with an additional directional groin-positive zone below the hip line, so lower-back/waist false positives are less likely to be protected just because they are close to the hips.
- Kept geometry v2 fail-open when a candidate cannot be assigned to a reliable person, when matched people disagree, or when another person's pelvis protects a close-contact candidate.
- Expanded Body analysis visualization with torso-v2, armpit, thigh, and lower-leg regions and human-readable evidence labels.
- Added a persistent local SQLite Experience Store for candidate evidence. The desktop GUI remembers compact candidate crops/evidence by default without copying original images.
- Added conservative pseudo-label tiers: strong known hard-negatives become GOLD, weaker evidence becomes SILVER, and conflicting/ambiguous examples go to QUARANTINE instead of being forced into training data.
- Added resumable legacy-corpus mining for mixed PNG/JPEG/WebP folders and ZIP archives. Corrupt files, exact duplicates, previously processed material, and unsupported files are skipped automatically; source files/archives are never modified.
- Added an idle-GPU gate for corpus mining, configurable from the GUI, so large legacy libraries can be processed opportunistically while the GPU is otherwise unused.
- Added a standalone `character-mosaic-mine` CLI for unattended corpus mining.
- Added conservative Negative Memory: repeated near-identical GOLD false positives can suppress an unprotected future candidate, while pelvis/groin-positive evidence disables this memory veto. Memory-generated suppressions are not promoted back into GOLD, preventing recursive self-reinforcement.
- Kept core/CLI processing side-effect-free by default; automatic experience capture is enabled by default only in the desktop GUI and can be disabled there.
- Disabled expensive flip/rotation retries for zero-result images during legacy-corpus mining, because they cannot contribute hard-negative candidates; normal production detection behavior is unchanged.
- Added no mandatory ML dependency for v1.4. Semantic hair segmentation and a learned verifier remain optional future consumers of the accumulated Experience Store rather than destabilizing the current ONNX/PySide installation.
- Added regression coverage for back/armpit/thigh/leg suppression, directional groin protection, unmatched/multi-person fail-open behavior, Experience Store deduplication, pseudo-label quarantine, Negative Memory protection, ZIP mining, corrupt-file handling, and resume behavior.

## 1.3.0

Rerun workflow, single-image testing, and stronger body-region suppression.

- Made GUI reruns overwrite existing generated outputs by default, including a one-time migration for older settings that silently persisted the former `overwrite=false` default.
- Re-running the same source now refreshes output/Review/manual-review artifacts while keeping JSONL logs as separate history files.
- Added drag-and-drop single-image processing directly on the preview area. A dropped image uses the normal detector, body analysis, logging, Review, and save pipeline instead of a separate test implementation.
- When the output folder is not locked, a dropped image uses `<image folder>/_censored`; an explicitly locked output folder is respected.
- Added a final body-reasoning layer that derives a conservative shoulder-to-hip torso/back BBox from reliable pose points.
- Strong torso/back overlap can now suppress waist/back/torso false positives, while nearly-horizontal poses skip this axis-aligned check for safety.
- Changed final body-analysis `REVIEW` decisions without pelvis evidence into `SUPPRESS`, reflecting real-use feedback that these boxes were reliable false positives.
- Preserved cross-person pelvis protection: another character's reliable pelvis evidence still wins over face/head/torso overlap for close-contact and oral compositions.
- Added a violet torso/back overlay and Japanese evidence labels in the Body analysis preview.
- Added regression coverage for overwrite migration, stale Review/manual-review cleanup, preview image drops, torso/back suppression, horizontal-pose safety, and cross-person pelvis protection.

## 1.2.0

Body-region reasoning and visual diagnostics.

- Expanded the anatomy layer from knee/armpit checks into a multi-signal body-region map using person, head, face, eye, and DWPose body information.
- Added candidate-level `KEEP`, `REVIEW`, and `SUPPRESS` evidence instead of treating one body part as an unconditional exclusion zone.
- Added cross-person pelvis protection so a candidate overlapping one character's face is still kept when another character's pelvis provides strong positive evidence.
- Added conservative face/head handling for oral and close-contact compositions: face/head overlap alone routes to Review and never auto-suppresses.
- Added strong eye+face+head confirmation for obvious facial false positives.
- Added a dedicated **Body analysis** preview mode with person/head/face/eye BBoxes, skeleton lines, pelvis/knee/armpit regions, and decision-colored candidate boxes.
- Candidate boxes in Body analysis are clickable and show the positive/negative evidence used for the decision.
- Added fail-open partial-helper handling: a failed auxiliary model is disabled for the rest of the batch while remaining body signals continue when possible.
- Expanded JSONL diagnostics with body regions, pose points/edges, and per-candidate evidence.
- Body-analysis Review decisions now participate in the normal Review workflow.
- Bumped the public application version to 1.2.0; `START_HERE.bat` continues to read the version dynamically from `pyproject.toml`.

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
