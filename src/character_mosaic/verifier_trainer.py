from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .image_ops import normalize_image
from .semantic_probe import candidate_context_crop
from .types import Detection
from .verifier_model import (
    VerifierKnnModel,
    binary_counts,
    default_verifier_dir,
    normalize_embedding,
    write_training_report,
)
from .verifier_store import VerifierLabelSample, VerifierStore


ProgressCallback = Callable[[int, int, VerifierLabelSample, str], None]


@dataclass(frozen=True)
class TrainingResult:
    model_path: Path
    report_path: Path
    report: dict


def stable_source_id(path: str) -> int:
    digest = hashlib.blake2b(str(path).encode("utf-8", errors="replace"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) & ((1 << 63) - 1)


def _cache_namespace(model_name: str, crop_scale: float, min_crop_side: int) -> str:
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model_name))
    scale = str(round(float(crop_scale), 3)).replace(".", "p")
    return f"{safe_model}_s{scale}_m{int(min_crop_side)}"


def _load_context_crop(
    sample: VerifierLabelSample,
    *,
    crop_scale: float,
    min_crop_side: int,
) -> Image.Image:
    source = Path(sample.source_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"source image unavailable: {source}")
    with Image.open(source) as opened:
        image = normalize_image(opened.copy()).convert("RGB")
    detection = Detection(sample.box, "pussy", sample.detector_score, sample.detector_source)
    crop, _ = candidate_context_crop(
        image,
        detection,
        scale=crop_scale,
        min_side=min_crop_side,
    )
    return crop.convert("RGB")


def _embedding_for_sample(
    sample: VerifierLabelSample,
    *,
    model_name: str,
    crop_scale: float,
    min_crop_side: int,
    cache_root: Path,
    rebuild_cache: bool,
    embedder: Callable | None,
) -> tuple[np.ndarray, str]:
    cache_dir = cache_root / _cache_namespace(model_name, crop_scale, min_crop_side)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{sample.fingerprint}.npy"
    if cache_path.is_file() and not rebuild_cache:
        try:
            return normalize_embedding(np.load(cache_path, allow_pickle=False)), "cache"
        except Exception:
            cache_path.unlink(missing_ok=True)

    crop = _load_context_crop(
        sample,
        crop_scale=crop_scale,
        min_crop_side=min_crop_side,
    )
    if embedder is None:
        from imgutils.tagging import get_wd14_tags

        embedding = get_wd14_tags(crop, model_name=model_name, fmt="embedding")
    else:
        embedding = embedder(crop, model_name=model_name, fmt="embedding")
    vector = normalize_embedding(embedding)
    np.save(cache_path, vector.astype(np.float32), allow_pickle=False)
    return vector, "computed"


def cross_validated_scores(
    embeddings,
    labels,
    source_ids,
    *,
    k: int,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score each row while excluding its source; return negative similarity.

    The second returned array is intentionally the nearest *negative* reference
    similarity. Runtime suppression uses the same value as its OOD guard. Rows
    without the same minimum per-class neighbour support required at runtime are
    marked invalid for policy selection.
    """

    matrix = np.asarray(embeddings, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int8)
    groups = np.asarray(source_ids, dtype=np.int64)
    probe_model = VerifierKnnModel(
        embeddings=matrix,
        labels=y,
        source_ids=groups,
        k=k,
        temperature=temperature,
        suppress_threshold=0.0,
        similarity_floor=-1.0,
    )
    scores = np.empty(len(y), dtype=np.float32)
    negative_similarities = np.empty(len(y), dtype=np.float32)
    neighbors = np.empty(len(y), dtype=np.int32)
    required_neighbors = min(3, max(1, int(k)))
    for index, embedding in enumerate(probe_model.embeddings):
        result = probe_model.score(embedding, exclude_source_id=int(groups[index]))
        class_neighbors = min(result.positive_neighbors, result.negative_neighbors)
        scores[index] = result.positive_score
        neighbors[index] = class_neighbors
        negative_similarities[index] = (
            result.negative_similarity if class_neighbors >= required_neighbors else -1.0
        )
    return scores, negative_similarities, neighbors


def choose_conservative_policy(
    scores,
    negative_similarities,
    labels,
    *,
    max_suppress_threshold: float = 0.35,
) -> dict:
    """Choose a one-sided veto with zero observed positive suppressions.

    The learned score is class-balanced, but negative training coverage is still
    much smaller than positive coverage. Therefore a candidate may be vetoed
    only when it is both below the positive-score threshold and sufficiently
    similar to a known human-labelled negative example.
    """

    p = np.asarray(scores, dtype=np.float64)
    neg_sim = np.asarray(negative_similarities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int8)
    valid = np.isfinite(p) & np.isfinite(neg_sim) & (neg_sim > -0.999)
    positives = valid & (y == 1)
    negatives = valid & (y == 0)
    if not np.any(positives) or not np.any(negatives):
        raise ValueError("both positive and negative validation rows are required")

    # Runtime uses score < threshold. Pinning the cutoff to the smallest
    # observed positive score preserves every validation positive.
    threshold = min(float(max_suppress_threshold), float(np.min(p[positives])))
    negative_candidates = negatives & (p < threshold)
    if np.any(negative_candidates):
        floor = float(np.quantile(neg_sim[negative_candidates], 0.10)) - 0.01
        floor = max(-1.0, min(1.0, floor))
    else:
        floor = 1.0

    suppressed = valid & (p < threshold) & (neg_sim >= floor)
    positive_count = int(np.sum(positives))
    negative_count = int(np.sum(negatives))
    false_suppressed = int(np.sum(suppressed & positives))
    negatives_suppressed = int(np.sum(suppressed & negatives))
    positive_recall = 1.0 - false_suppressed / max(1, positive_count)
    negative_suppression_rate = negatives_suppressed / max(1, negative_count)
    suppressed_total = int(np.sum(suppressed))
    negative_precision = negatives_suppressed / max(1, suppressed_total)

    return {
        "suppress_threshold": threshold,
        "similarity_floor": floor,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_false_suppressed": false_suppressed,
        "positive_recall": positive_recall,
        "negative_suppressed": negatives_suppressed,
        "negative_suppression_rate": negative_suppression_rate,
        "negative_precision_among_suppressed": negative_precision,
        "valid_rows": int(np.sum(valid)),
    }


def train_verifier(
    *,
    store: VerifierStore | None = None,
    output_dir: Path | None = None,
    model_name: str = "SwinV2_v3",
    crop_scale: float = 3.8,
    min_crop_side: int = 320,
    k: int = 9,
    temperature: float = 0.06,
    max_suppress_threshold: float = 0.35,
    max_samples: int | None = None,
    rebuild_cache: bool = False,
    progress: ProgressCallback | None = None,
    embedder: Callable | None = None,
) -> TrainingResult:
    store = store or VerifierStore()
    output_dir = Path(output_dir or default_verifier_dir())
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = output_dir / "embedding_cache"

    label_stats = store.stats()
    coverage = store.coverage_stats()
    samples = store.labeled_samples(labels=("positive", "negative"), limit=max_samples)
    if len(samples) < 4:
        raise ValueError("Verifier学習には本物/誤検出ラベルが最低数件必要です。")

    embeddings: list[np.ndarray] = []
    labels: list[int] = []
    source_ids: list[int] = []
    fingerprints: list[str] = []
    failures: list[dict[str, str]] = []
    cache_hits = 0
    computed = 0

    total = len(samples)
    for index, sample in enumerate(samples, 1):
        try:
            embedding, state = _embedding_for_sample(
                sample,
                model_name=model_name,
                crop_scale=crop_scale,
                min_crop_side=min_crop_side,
                cache_root=cache_root,
                rebuild_cache=rebuild_cache,
                embedder=embedder,
            )
            cache_hits += int(state == "cache")
            computed += int(state == "computed")
            embeddings.append(embedding)
            labels.append(1 if sample.label == "positive" else 0)
            source_ids.append(stable_source_id(sample.source_path))
            fingerprints.append(sample.fingerprint)
            if progress:
                progress(index, total, sample, state)
        except Exception as exc:
            failures.append(
                {
                    "fingerprint": sample.fingerprint,
                    "source": sample.source_path,
                    "label": sample.label,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if progress:
                progress(index, total, sample, f"error:{type(exc).__name__}")

    if len(embeddings) < 4:
        raise ValueError("画像を読み込めた教師データが不足しています。")
    matrix = np.stack(embeddings).astype(np.float32)
    y = np.asarray(labels, dtype=np.int8)
    groups = np.asarray(source_ids, dtype=np.int64)
    counts = binary_counts(y)
    if counts["positive"] < 2 or counts["negative"] < 2:
        raise ValueError("読み込み可能な本物/誤検出ラベルがそれぞれ2件以上必要です。")

    cv_scores, cv_negative_similarities, cv_neighbors = cross_validated_scores(
        matrix,
        y,
        groups,
        k=k,
        temperature=temperature,
    )
    policy = choose_conservative_policy(
        cv_scores,
        cv_negative_similarities,
        y,
        max_suppress_threshold=max_suppress_threshold,
    )

    model = VerifierKnnModel(
        embeddings=matrix,
        labels=y,
        source_ids=groups,
        k=k,
        temperature=temperature,
        suppress_threshold=float(policy["suppress_threshold"]),
        similarity_floor=float(policy["similarity_floor"]),
        model_name=model_name,
        crop_scale=crop_scale,
        min_crop_side=min_crop_side,
    )
    model_path = model.save(output_dir / "model.npz")

    enough_data = counts["positive"] >= 20 and counts["negative"] >= 20
    activation_recommended = bool(
        enough_data
        and policy["positive_recall"] >= 0.995
        and policy["negative_precision_among_suppressed"] >= 0.98
        and policy["negative_suppression_rate"] >= 0.20
    )
    report = {
        "schema": 2,
        "model_type": "wd14_embedding_class_balanced_distinct_source_knn",
        "model_name": model_name,
        "crop_scale": float(crop_scale),
        "min_crop_side": int(min_crop_side),
        "k_per_class": int(k),
        "temperature": float(temperature),
        "coverage": coverage,
        "manual_labels": label_stats,
        "usable_training_rows": counts,
        "embedding_cache_hits": cache_hits,
        "embeddings_computed": computed,
        "embedding_failures": failures,
        "cross_validation": {
            **policy,
            "rows_without_required_class_neighbors": int(np.sum(cv_neighbors < min(3, max(1, int(k))))),
            "mean_negative_similarity": float(np.mean(cv_negative_similarities[cv_negative_similarities > -0.999]))
            if np.any(cv_negative_similarities > -0.999)
            else -1.0,
        },
        "activation_recommended": activation_recommended,
        "activation_rule": (
            "candidate may be suppressed only when class-balanced positive_score is below "
            "suppress_threshold and nearest-negative similarity is above similarity_floor; "
            "at least three distinct-source neighbors per class and recall safety gates remain required"
        ),
        "fingerprints": fingerprints,
        "model_path": str(model_path),
    }
    report_path = write_training_report(report, output_dir / "training_report.json")
    return TrainingResult(model_path=model_path, report_path=report_path, report=report)
