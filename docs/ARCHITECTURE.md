# Architecture

Character Mosaic Censor is intentionally split into a small detection/pipeline core and a PySide6 desktop shell. The UI never owns image-processing rules, which keeps the batch engine testable and makes future detector replacement possible without rewriting the application.

## High-level flow

```text
MainWindow
  └─ BatchWorker (QThread)
      └─ BatchProcessor
          ├─ AnimeCensorDetector
          │   ├─ full-frame inference
          │   ├─ optional 2x2 / 3x3 overlapping tiles
          │   ├─ optional horizontal-flip TTA
          │   └─ cross-pass box union/merge
          ├─ censor image operation
          ├─ review manifest / HTML
          └─ JSONL run log
```

## Module responsibilities

| Module | Responsibility |
|---|---|
| `detector.py` | Detector abstraction, dghs-imgutils adapter, tiling/TTA, merge logic, runtime diagnostics |
| `pipeline.py` | Stable public facade for the batch-processing API |
| `pipeline_processor.py` | Folder discovery, path safety, and per-image orchestration |
| `pipeline_config.py` | Validated pipeline settings |
| `pipeline_logging.py` | Crash-tolerant JSONL run logging |
| `pipeline_review.py` | Persistent Review manifest and HTML |
| `pipeline_storage.py` | Atomic file writes, metadata preservation, image discovery |
| `image_ops.py` | EXIF normalization, box expansion, Mosaic/Blur/Black operations, review overlays |
| `types.py` | Immutable transfer objects used across detector, pipeline, worker, and UI |
| `workers/batch_worker.py` | QThread worker, cooperative cancellation, GUI-safe signal emission |
| `ui/*` | Main window, preview rendering, controls, settings dialog |
| `cli.py` | Headless batch entry point using the same pipeline |

## Detection strategy

The default detector is `dghs-imgutils` `detect_censors` using the standard `s` model. The application favors recall over strict precision.

1. Run a full-frame pass.
2. For sufficiently large images, run overlapping tiled passes.
3. Optionally repeat each pass on a horizontally flipped image.
4. Convert every detection back into the original image coordinate system.
5. Merge detections that overlap strongly by IoU or intersection-over-smaller-box.
6. Use the union of matching boxes rather than discarding the wider alternative.
7. Expand the final censor area by fixed pixels plus a ratio of the detected box size.

This intentionally makes censor coverage conservative.

## Threading model

Qt UI objects remain on the main thread. Batch inference runs inside `BatchWorker`, moved to a dedicated `QThread`.

The worker communicates through signals only. Cancellation is cooperative: the current detector pass is allowed to return, but an image whose complete pass set has not finished is not written as a finished output.

## File-safety model

- Input, output, and review paths are normalized before processing.
- Output/review trees are excluded from recursive input discovery.
- Dangerous parent/child path combinations are rejected.
- Modified files are written to temporary siblings and installed with `os.replace`.
- Images without detections are byte-copied when possible, avoiding unnecessary recompression.
- Review state is persisted in `manifest.json` and rebuilt into `index.html`.
- JSONL logs are flushed after each image to keep completed work inspectable after a crash.

## Metadata behavior

When an image must be re-encoded, the pipeline preserves supported metadata where practical, including PNG text fields commonly used for generation parameters, ICC profiles, DPI, EXIF, and WebP XMP. EXIF Orientation is removed after pixel normalization to prevent double rotation.

## Extension points

`Detector` is the intended boundary for future models:

```python
class Detector(ABC):
    @abstractmethod
    def detect(self, image, progress=None, stop_requested=None):
        ...
```

A future `CustomYoloDetector` can be added without changing the batch pipeline or desktop layout as long as it returns the shared `Detection` type.
