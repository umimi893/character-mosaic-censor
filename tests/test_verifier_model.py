from __future__ import annotations

import numpy as np
import pytest

from character_mosaic.verifier_model import VerifierKnnModel


def _model():
    embeddings = np.asarray([
        [1.00, 0.00, 0.00],
        [0.98, 0.08, 0.00],
        [0.95, 0.12, 0.02],
        [0.00, 1.00, 0.00],
        [0.08, 0.98, 0.00],
        [0.12, 0.95, 0.02],
    ], dtype=np.float32)
    labels = np.asarray([1, 1, 1, 0, 0, 0], dtype=np.int8)
    source_ids = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int64)
    return VerifierKnnModel(
        embeddings=embeddings,
        labels=labels,
        source_ids=source_ids,
        k=3,
        support_k=3,
        temperature=0.05,
        margin_threshold=0.02,
        negative_similarity_floor=0.70,
    )


def test_knn_verifier_separates_synthetic_visual_clusters():
    model = _model()
    positive = model.score(np.asarray([1.0, 0.02, 0.0], dtype=np.float32))
    negative = model.score(np.asarray([0.02, 1.0, 0.0], dtype=np.float32))
    assert positive.margin < 0.0
    assert negative.margin > 0.0
    assert positive.positive_score > 0.90
    assert negative.positive_score < 0.10
    assert positive.positive_neighbors == 3
    assert positive.negative_neighbors == 3
    assert negative.negative_support > negative.positive_support
    assert model.should_suppress(np.asarray([0.02, 1.0, 0.0], dtype=np.float32))[0]
    assert not model.should_suppress(np.asarray([1.0, 0.02, 0.0], dtype=np.float32))[0]


def test_knn_verifier_excludes_same_source_from_both_classes():
    model = _model()
    score = model.score(model.embeddings[0], exclude_source_id=1)
    assert score.positive_neighbors == 2
    assert score.negative_neighbors == 3
    assert score.neighbors == 5
    assert score.margin < 0.0


def test_class_margin_handles_dense_positive_corpus_without_count_bias():
    positive = np.asarray(
        [[0.78 + i * 0.001, 0.62 - i * 0.001] for i in range(30)],
        dtype=np.float32,
    )
    negative = np.asarray([
        [0.05, 1.00],
        [0.10, 0.99],
        [0.15, 0.98],
    ], dtype=np.float32)
    embeddings = np.vstack([positive, negative])
    labels = np.asarray([1] * len(positive) + [0] * len(negative), dtype=np.int8)
    source_ids = np.arange(1, len(labels) + 1, dtype=np.int64)
    model = VerifierKnnModel(
        embeddings=embeddings,
        labels=labels,
        source_ids=source_ids,
        k=3,
        support_k=3,
        temperature=0.06,
        margin_threshold=0.02,
        negative_similarity_floor=0.70,
    )

    score = model.score(np.asarray([0.08, 1.0], dtype=np.float32))

    assert score.positive_neighbors == 3
    assert score.negative_neighbors == 3
    assert score.margin > 0.20
    assert model.should_suppress(np.asarray([0.08, 1.0], dtype=np.float32))[0]


def test_margin_requires_negative_support_to_beat_positive_support():
    model = _model()
    score = model.score(np.asarray([0.70, 0.70, 0.0], dtype=np.float32))
    if score.margin <= model.margin_threshold:
        assert not model.should_suppress(np.asarray([0.70, 0.70, 0.0], dtype=np.float32))[0]


def test_knn_verifier_round_trips_model_file(tmp_path):
    model = _model()
    path = model.save(tmp_path / "model.npz")
    restored = VerifierKnnModel.load(path)
    before = model.score(np.asarray([0.02, 1.0, 0.0], dtype=np.float32))
    after = restored.score(np.asarray([0.02, 1.0, 0.0], dtype=np.float32))
    assert abs(before.margin - after.margin) < 1e-6
    assert restored.k == model.k
    assert restored.support_k == model.support_k
    assert abs(restored.margin_threshold - model.margin_threshold) < 1e-6
    assert abs(restored.negative_similarity_floor - model.negative_similarity_floor) < 1e-6


def test_legacy_verifier_model_is_rejected_until_retrained(tmp_path):
    path = tmp_path / "legacy.npz"
    model = _model()
    np.savez_compressed(
        path,
        score_version=np.asarray([2]),
        embeddings=model.embeddings,
        labels=model.labels,
        source_ids=model.source_ids,
        k=np.asarray([model.k]),
        temperature=np.asarray([model.temperature]),
        suppress_threshold=np.asarray([0.20]),
        similarity_floor=np.asarray([0.70]),
        model_name=np.asarray([model.model_name]),
        crop_scale=np.asarray([model.crop_scale]),
        min_crop_side=np.asarray([model.min_crop_side]),
    )

    with pytest.raises(ValueError, match="requires retraining"):
        VerifierKnnModel.load(path)
