from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .experience_store import default_learning_root


_SCORE_VERSION = 2


def default_verifier_dir() -> Path:
    return default_learning_root() / "verifier"


def default_verifier_model_path() -> Path:
    return default_verifier_dir() / "model.npz"


def default_verifier_report_path() -> Path:
    return default_verifier_dir() / "training_report.json"


def normalize_embedding(value) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("verifier embedding has zero/invalid norm")
    return vector / norm


def normalize_embedding_rows(values) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("verifier embeddings must be a non-empty 2D matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-12):
        raise ValueError("verifier embeddings contain zero/invalid vectors")
    return matrix / norms


@dataclass(frozen=True)
class VerifierScore:
    positive_score: float
    max_similarity: float
    neighbors: int
    positive_similarity: float = -1.0
    negative_similarity: float = -1.0
    positive_neighbors: int = 0
    negative_neighbors: int = 0


@dataclass
class VerifierKnnModel:
    """One-sided visual verifier backed by human-labelled WD14 embeddings.

    Version 2 deliberately selects neighbours separately for the positive and
    negative classes. The user corpus is naturally very imbalanced (many true
    targets, relatively few false positives), so a single global top-k lets the
    denser positive class dominate even for unseen knee/armpit/skin creases.

    Each class gets the same prior weight here: up to ``k`` distinct source
    images are selected per class, then their temperature-weighted similarity
    evidence is compared. Runtime suppression also requires direct similarity
    to known negative examples, not merely similarity to either class.
    """

    embeddings: np.ndarray
    labels: np.ndarray
    source_ids: np.ndarray
    k: int = 9
    temperature: float = 0.06
    suppress_threshold: float = 0.20
    similarity_floor: float = 0.0
    model_name: str = "SwinV2_v3"
    crop_scale: float = 3.8
    min_crop_side: int = 320

    def __post_init__(self) -> None:
        self.embeddings = normalize_embedding_rows(self.embeddings)
        self.labels = np.asarray(self.labels, dtype=np.int8).reshape(-1)
        self.source_ids = np.asarray(self.source_ids, dtype=np.int64).reshape(-1)
        if len(self.embeddings) != len(self.labels) or len(self.labels) != len(self.source_ids):
            raise ValueError("verifier model arrays have inconsistent row counts")
        if not set(int(value) for value in np.unique(self.labels)).issubset({0, 1}):
            raise ValueError("verifier labels must be binary")
        self.k = max(1, int(self.k))
        self.temperature = max(1e-4, float(self.temperature))
        self.suppress_threshold = float(self.suppress_threshold)
        self.similarity_floor = float(self.similarity_floor)

    def _select_class_neighbors(
        self,
        similarities: np.ndarray,
        *,
        label: int,
        exclude_source_id: int | None,
    ) -> list[int]:
        order = np.argsort(similarities)[::-1]
        selected: list[int] = []
        seen_sources: set[int] = set()
        for index in order:
            index = int(index)
            if int(self.labels[index]) != int(label):
                continue
            source_id = int(self.source_ids[index])
            if exclude_source_id is not None and source_id == int(exclude_source_id):
                continue
            # One source image contributes at most one vote to each class. A
            # source may legitimately contain both a true target and a false
            # positive candidate, so the de-duplication is class-local.
            if source_id in seen_sources:
                continue
            seen_sources.add(source_id)
            selected.append(index)
            if len(selected) >= self.k:
                break
        return selected

    def score(self, embedding, *, exclude_source_id: int | None = None) -> VerifierScore:
        query = normalize_embedding(embedding)
        if query.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                f"embedding width mismatch: got {query.shape[0]}, expected {self.embeddings.shape[1]}"
            )
        similarities = self.embeddings @ query
        positive_idx = self._select_class_neighbors(
            similarities,
            label=1,
            exclude_source_id=exclude_source_id,
        )
        negative_idx = self._select_class_neighbors(
            similarities,
            label=0,
            exclude_source_id=exclude_source_id,
        )

        pos_n = len(positive_idx)
        neg_n = len(negative_idx)
        if not positive_idx or not negative_idx:
            selected = positive_idx + negative_idx
            max_similarity = float(np.max(similarities[selected])) if selected else -1.0
            return VerifierScore(
                positive_score=0.5,
                max_similarity=max_similarity,
                neighbors=len(selected),
                positive_similarity=(float(np.max(similarities[positive_idx])) if positive_idx else -1.0),
                negative_similarity=(float(np.max(similarities[negative_idx])) if negative_idx else -1.0),
                positive_neighbors=pos_n,
                negative_neighbors=neg_n,
            )

        pos_sims = similarities[positive_idx].astype(np.float64)
        neg_sims = similarities[negative_idx].astype(np.float64)
        pos_max = float(np.max(pos_sims))
        neg_max = float(np.max(neg_sims))
        global_max = max(pos_max, neg_max)

        # Equal class priors: average evidence inside each class instead of
        # summing all neighbours together. This removes the 30:1-ish corpus
        # imbalance from the score while preserving visual similarity strength.
        pos_weights = np.exp(np.clip((pos_sims - global_max) / self.temperature, -50.0, 0.0))
        neg_weights = np.exp(np.clip((neg_sims - global_max) / self.temperature, -50.0, 0.0))
        pos_evidence = float(np.mean(pos_weights))
        neg_evidence = float(np.mean(neg_weights))
        denom = pos_evidence + neg_evidence
        positive_score = pos_evidence / max(denom, 1e-12)

        return VerifierScore(
            positive_score=max(0.0, min(1.0, float(positive_score))),
            max_similarity=global_max,
            neighbors=pos_n + neg_n,
            positive_similarity=pos_max,
            negative_similarity=neg_max,
            positive_neighbors=pos_n,
            negative_neighbors=neg_n,
        )

    def should_suppress(self, embedding) -> tuple[bool, VerifierScore]:
        score = self.score(embedding)
        required_neighbors = min(3, self.k)
        decision = (
            score.positive_neighbors >= required_neighbors
            and score.negative_neighbors >= required_neighbors
            and score.negative_similarity >= self.similarity_floor
            and score.positive_score < self.suppress_threshold
        )
        return bool(decision), score

    def save(self, path: Path | None = None) -> Path:
        path = Path(path or default_verifier_model_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            score_version=np.asarray([_SCORE_VERSION], dtype=np.int32),
            embeddings=self.embeddings.astype(np.float32),
            labels=self.labels.astype(np.int8),
            source_ids=self.source_ids.astype(np.int64),
            k=np.asarray([self.k], dtype=np.int32),
            temperature=np.asarray([self.temperature], dtype=np.float32),
            suppress_threshold=np.asarray([self.suppress_threshold], dtype=np.float32),
            similarity_floor=np.asarray([self.similarity_floor], dtype=np.float32),
            model_name=np.asarray([self.model_name]),
            crop_scale=np.asarray([self.crop_scale], dtype=np.float32),
            min_crop_side=np.asarray([self.min_crop_side], dtype=np.int32),
        )
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "VerifierKnnModel":
        path = Path(path or default_verifier_model_path())
        with np.load(path, allow_pickle=False) as data:
            if "score_version" not in data.files or int(data["score_version"][0]) != _SCORE_VERSION:
                raise ValueError("legacy verifier model requires retraining for class-balanced scoring")
            return cls(
                embeddings=np.asarray(data["embeddings"], dtype=np.float32),
                labels=np.asarray(data["labels"], dtype=np.int8),
                source_ids=np.asarray(data["source_ids"], dtype=np.int64),
                k=int(data["k"][0]),
                temperature=float(data["temperature"][0]),
                suppress_threshold=float(data["suppress_threshold"][0]),
                similarity_floor=float(data["similarity_floor"][0]),
                model_name=str(data["model_name"][0]),
                crop_scale=float(data["crop_scale"][0]),
                min_crop_side=int(data["min_crop_side"][0]),
            )


def write_training_report(report: dict, path: Path | None = None) -> Path:
    path = Path(path or default_verifier_report_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def binary_counts(labels: Iterable[int]) -> dict[str, int]:
    values = [int(value) for value in labels]
    return {
        "positive": sum(value == 1 for value in values),
        "negative": sum(value == 0 for value in values),
        "total": len(values),
    }
