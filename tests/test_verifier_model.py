from __future__ import annotations

import numpy as np

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
        temperature=0.05,
        suppress_threshold=0.20,
        similarity_floor=0.70,
    )


def test_knn_verifier_separates_synthetic_visual_clusters():
    model = _model()
    positive = model.score(np.asarray([1.0, 0.02, 0.0], dtype=np.float32))
    negative = model.score(np.asarray([0.02, 1.0, 0.0], dtype=np.float32))
    assert positive.positive_score > 0.90
    assert negative.positive_score < 0.10
    assert model.should_suppress(np.asarray([0.02, 1.0, 0.0], dtype=np.float32))[0]
    assert not model.should_suppress(np.asarray([1.0, 0.02, 0.0], dtype=np.float32))[0]


def test_knn_verifier_excludes_same_source_from_validation_score():
    model = _model()
    score = model.score(model.embeddings[0], exclude_source_id=1)
    assert score.neighbors == 3
    assert score.positive_score > 0.80


def test_knn_verifier_round_trips_model_file(tmp_path):
    model = _model()
    path = model.save(tmp_path / "model.npz")
    restored = VerifierKnnModel.load(path)
    before = model.score(np.asarray([0.02, 1.0, 0.0], dtype=np.float32))
    after = restored.score(np.asarray([0.02, 1.0, 0.0], dtype=np.float32))
    assert abs(before.positive_score - after.positive_score) < 1e-6
    assert restored.k == model.k
    assert abs(restored.suppress_threshold - model.suppress_threshold) < 1e-6
