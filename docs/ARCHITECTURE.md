# Architecture

Character Mosaic Censor is split into a detection/pipeline core and a PySide6 desktop shell. UI classes do not own image-processing rules, so detector/body reasoning, storage, mining, and future learned verification can evolve independently.

## Production processing flow

```text
MainWindow
  └─ BatchWorker (QThread)
      └─ BatchProcessor
          └─ detector chain
              ├─ AnimeCensorDetector
              │   ├─ full-frame inference
              │   ├─ optional 2x2 / 3x3 overlapping tiles
              │   ├─ zero-result flip / rotation retries
              │   └─ cross-pass box union/merge
              ├─ BodyReasoningDetector
              │   └─ person/head/face/eye + DWPose evidence
              ├─ GeometryV2Detector
              │   └─ torso/back/armpit/thigh/leg + directional groin geometry
              └─ NegativeMemoryDetector (GUI learning enabled)
                  └─ repeated GOLD hard-negative similarity veto
          ├─ censor image operation
          ├─ review/manual-review state
          ├─ JSONL run log
          └─ best-effort Experience Store capture
```

The base censor detector remains the only source of target candidates. Later layers may suppress a candidate but do not invent a new censor target.

## Learning / corpus-mining flow

```text
Noisy legacy folders / ZIP archives
  └─ CorpusMinerWorker (QThread) or miner CLI
      └─ CorpusMiner
          ├─ resume/signature check
          ├─ image decode / size validation
          ├─ SHA-256 exact deduplication
          ├─ AnimeCensorDetector (tiling enabled, zero-result flip TTA disabled)
          ├─ BodyReasoningDetector
          ├─ GeometryV2Detector
          ├─ pseudo-label: GOLD / SILVER / QUARANTINE
          └─ ExperienceStore (SQLite + compact candidate crops)
```

Legacy material is treated as a **mine**, not as trusted ground truth. Ambiguous examples are quarantined rather than forced into a positive/negative label.

The miner deliberately does not use `NegativeMemoryDetector` to manufacture its own trusted examples. This breaks a potentially dangerous feedback loop: memory-derived suppressions cannot recursively become GOLD evidence.

## Module responsibilities

| Module | Responsibility |
|---|---|
| `detector.py` | Detector abstraction, dghs-imgutils censor adapter, tiling/TTA, merge logic, runtime diagnostics |
| `anatomy_filter.py` | Person/head/face/eye/DWPose body map and lower-level candidate evidence |
| `body_reasoning.py` | v1.3 final body reasoning, torso/back region, pelvis/cross-person safety |
| `body_geometry.py` | v1.4 pose-aligned torso/armpit/thigh/lower-leg hard-negative geometry and directional groin protection |
| `negative_memory.py` | Conservative repeated-GOLD visual false-positive memory |
| `experience_store.py` | SQLite schema, source/candidate records, fingerprints, pseudo-label tiers, mining-run state |
| `experience_recorder.py` | Best-effort capture from ordinary GUI processing |
| `corpus_miner.py` | Noisy-folder/ZIP discovery, resume, decode validation, exact dedupe, idle-GPU gate, candidate mining |
| `miner_cli.py` | Standalone unattended corpus-mining CLI |
| `pipeline.py` | Stable public facade and production detector-layer composition |
| `pipeline_processor.py` | Folder discovery, path safety, and per-image orchestration |
| `pipeline_config.py` | Validated pipeline settings; core learning is opt-in |
| `pipeline_logging.py` | Crash-tolerant JSONL run logging |
| `pipeline_review.py` | Persistent Review manifest and HTML |
| `pipeline_storage.py` | Atomic file writes, metadata preservation, image discovery |
| `image_ops.py` | EXIF normalization, box expansion, Mosaic/Blur/Black operations, review overlays |
| `types.py` | Immutable transfer objects shared across detector, pipeline, workers, and UI |
| `workers/batch_worker.py` | Production QThread worker and cooperative cancellation |
| `workers/corpus_miner_worker.py` | Background mining worker and cooperative cancellation |
| `ui/learning_dialog.py` | Corpus roots, mining options, progress, and Experience Store statistics |
| `ui/*` | Main window, preview rendering, controls, settings and UX layers |
| `cli.py` | Headless normal batch entry point using the same pipeline |

## Detection strategy

The default detector is `dghs-imgutils` `detect_censors` using the standard `s` model. Production processing favors recall.

1. Run a full-frame pass.
2. For sufficiently large images, run overlapping tiled passes.
3. If production passes find no target, retry with horizontal/vertical flips and 90/180/270-degree rotations.
4. Convert transformed detections back to original coordinates.
5. Merge strongly overlapping detections using IoU/intersection-over-smaller-box rules.
6. Preserve the union of matching boxes rather than dropping a wider alternate box.
7. Pass candidates through body/anatomy reasoning and geometry-v2 suppression.
8. If GUI learning is enabled, allow only conservative repeated-GOLD Negative Memory to veto an otherwise kept unprotected candidate.
9. Expand retained censor boxes by configured fixed/relative margins.

The desktop GUI treats the configured people value as a maximum plausible final detection count for manual over-detection quarantine. Zero detections are a valid no-target result.

### Corpus-miner detector difference

The corpus miner keeps full-frame and large-image tiled inference but disables expensive flip/rotation retries after a zero-result image. A zero-result image contributes no hard-negative candidate, so those retries only waste mining time; this does not change production detection behavior.

## Geometry-v2 safety model

Geometry v2 is an aggressive false-positive layer only when pose association is reliable.

- Unmatched candidates fail open.
- Missing body landmarks remove the relevant geometry rule.
- Multi-person candidates require hard-negative agreement from all matched people.
- Pelvis evidence from another person protects close-contact/oral compositions.
- The directional groin region is evaluated before torso/armpit/leg suppression.
- Limb negatives are tubes around actual pose bones, not global axis-aligned rectangles.

The visual preview exposes derived regions and candidate evidence so these decisions remain inspectable.

## Experience Store and self-improvement

The default desktop GUI enables candidate experience capture; core/CLI callers do not write learning data unless they opt in.

The default store is under:

```text
%LOCALAPPDATA%\CharacterMosaicCensor\learning\
```

SQLite stores source signatures, exact SHA-256 hashes, candidate BBoxes, detector/evidence metadata, pseudo-label tier, suppression reason, perceptual fingerprint, and optional compact crop path. Original source images are not copied by the recorder/miner.

Pseudo-label policy is intentionally conservative:

- `GOLD negative`: suppression has a recognized high-confidence hard-negative reason.
- `SILVER`: potentially useful but not trusted as strongly.
- `QUARANTINE`: evidence conflicts or is otherwise ambiguous.

`NegativeMemoryDetector` uses only repeated near-identical **GOLD** negative fingerprints. A pelvis/groin-positive signal disables this veto. A memory-derived suppression itself maps only to weaker evidence, preventing recursive trust amplification.

Version 1.4 intentionally does not auto-train and auto-promote a deep verifier entirely from pseudo-labels. The Experience Store is the stable data boundary for a future verifier, semantic hair provider, or detector fine-tune once a trustworthy evaluation strategy exists.

## Threading model

Qt UI objects remain on the main thread. Production inference runs in `BatchWorker`; corpus mining runs in `CorpusMinerWorker`. Both are moved to dedicated `QThread` instances and communicate through signals.

Cancellation is cooperative. Production processing does not write an image as finished when its complete inference has not finished. Mining stores completed source records and resumes past them on later runs.

## File-safety model

- Input, output, and review paths are normalized before processing.
- Output/review trees are excluded from recursive input discovery.
- Dangerous parent/child path combinations are rejected.
- Modified files are written to temporary siblings and installed with `os.replace`.
- Images without detections are byte-copied when possible.
- Review state is persisted in `manifest.json` and rebuilt into `index.html`.
- JSONL logs are flushed after each image.
- Corpus mining never modifies source images or ZIP files.
- Corrupt/unreadable corpus files are recorded/skipped rather than aborting the mining run.
- Learning-store failures are isolated from normal censor-output success.

## Metadata behavior

When an image must be re-encoded, the pipeline preserves supported metadata where practical, including PNG generation text, ICC profiles, DPI, EXIF, and WebP XMP. EXIF Orientation is removed after pixel normalization to prevent double rotation.

## Extension points

`Detector` remains the boundary for replacement candidate generators:

```python
class Detector(ABC):
    @abstractmethod
    def detect(self, image, progress=None, stop_requested=None):
        ...
```

The Experience Store is a second extension boundary. Future learned components should consume stored candidate examples/evidence rather than reaching into GUI state or modifying source libraries directly.
