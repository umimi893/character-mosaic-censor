from __future__ import annotations

from pathlib import Path

from PIL import Image

from . import __version__
from .experience_store import (
    ExperienceStore,
    candidate_crop,
    candidate_fingerprint,
    classify_pseudo_label,
    default_learning_root,
    sha256_file,
    suppression_reason_map,
)
from .image_ops import normalize_image
from .types import ProcessResult


def record_process_experience(
    source: Path,
    result: ProcessResult,
    *,
    store: ExperienceStore | None = None,
    save_crops: bool = True,
) -> None:
    """Persist one normal processing result as future learning experience.

    This function is intentionally best-effort at its call sites: learning must
    never make normal censoring fail. Original source files are not copied.
    """
    source = Path(source)
    if not source.is_file() or result.error or result.cancelled or result.skipped:
        return

    store = store or ExperienceStore()
    stat = source.stat()
    source_key = f"file://{source.resolve()}"
    signature = f"{stat.st_size}:{stat.st_mtime_ns}:v{__version__}"
    if store.source_seen(source_key, signature):
        return

    digest = sha256_file(source)
    duplicate_of = store.duplicate_source_id(digest, excluding_key=source_key)

    with Image.open(source) as opened:
        image = normalize_image(opened.copy())

    source_id = store.upsert_source(
        source_key=source_key,
        container_path=str(source.resolve()),
        member_path=None,
        signature=signature,
        sha256=digest,
        size_bytes=stat.st_size,
        width=image.width,
        height=image.height,
        status="duplicate" if duplicate_of else "processed",
        duplicate_of=duplicate_of,
    )
    if duplicate_of:
        return

    suppression = suppression_reason_map(result)
    for evidence in result.candidate_evidence:
        reason = suppression.get(evidence.detection)
        pseudo_label, tier, negative_kind = classify_pseudo_label(evidence, reason)
        crop = candidate_crop(image, evidence.detection)
        fingerprint = candidate_fingerprint(crop)
        crop_path: str | None = None
        if save_crops and tier in {"gold", "silver"}:
            crop_path = str(_save_crop(crop, fingerprint, pseudo_label, negative_kind))
        store.record_candidate(
            source_id,
            evidence,
            pseudo_label=pseudo_label,
            quality_tier=tier,
            negative_kind=negative_kind,
            suppression_reason=reason,
            fingerprint=fingerprint,
            crop_path=crop_path,
            app_version=__version__,
        )


def _save_crop(image: Image.Image, fingerprint: str, pseudo_label: str, negative_kind: str | None) -> Path:
    root = default_learning_root() / "crops" / pseudo_label / (negative_kind or "generic")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{fingerprint}.webp"
    if not path.exists():
        image.save(path, "WEBP", quality=86, method=4)
    return path
