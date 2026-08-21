from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .experience_store import default_learning_root


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


@dataclass
class VerifierKnnModel:
    """Small one-sided visual verifier backed by labelled WD14 embeddings.

    The model is intentionally non-parametric: every human-labelled example is
    retained as a reference embedding.  At runtime it can veto a detector
    candidate only when the candidate looks sufficiently similar to the human
    labelled corpus *and* its local neighbours are strongly negative.
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

    def score(self, embedding, *, exclude_source_id: int | None = None) -> VerifierScore:
        query = normalize_embedding(embedding)
        if query.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                f"embedding width mismatch: got {query.shape[0]}, expected {self.embeddings.shape[1]}"
            )
        similarities = self.embeddings @ query
        order = np.argsort(similarities)[::-1]
        selected: list[int] = []
        seen_sources: set[int] = set()
        for index in order:
            source_id = int(self.source_ids[index])
            if exclude_source_id is not None and source_id == int(exclude_source_id):
                continue
            # One reference image gets one vote.  This prevents multiple crops
            # from a single source image from manufacturing confidence.
            if source_id in seen_sources:
                continue
            seen_sources.add(source_id)
            selected.append(int(index))
            if len(selected) >= min(self.k, len(self.embeddings)):
                break

        if not selected:
            return VerifierScore(positive_score=0.5, max_similarity=-1.0, neighbors=0)

        sims = similarities[selected].astype(np.float64)
        labels = self.labels[selected].astype(np.float64)
        max_similarity = float(np.max(sims))
        shifted = (sims - max_similarity) / self.temperature
        weights = np.exp(np.clip(shifted, -50.0, 0.0))
        denom = float(np.sum(weights))
        positive_score = float(np.sum(weights * labels) / max(denom, 1e-12))
        return VerifierScore(
            positive_score=max(0.0, min(1.0, positive_score)),
            max_similarity=max_similarity,
            neighbors=len(selected),
        )

    def should_suppress(self, embedding) -> tuple[bool, VerifierScore]:
        score = self.score(embedding)
        decision = (
            score.neighbors > 0
            and score.max_similarity >= self.similarity_floor
            and score.positive_score < self.suppress_threshold
        )
        return bool(decision), score

    def save(self, path: Path | None = None) -> Path:
        path = Path(path or default_verifier_model_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
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
