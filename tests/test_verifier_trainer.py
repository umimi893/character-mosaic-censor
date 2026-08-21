from __future__ import annotations

import numpy as np

from character_mosaic.verifier_trainer import (
    choose_conservative_policy,
    cross_validated_scores,
    stable_source_id,
)


def test_conservative_policy_preserves_all_observed_positives():
    scores = np.asarray([0.82, 0.76, 0.71, 0.02, 0.08, 0.18], dtype=np.float32)
    similarities = np.asarray([0.91, 0.89, 0.88, 0.94, 0.90, 0.86], dtype=np.float32)
    labels = np.asarray([1, 1, 1, 0, 0, 0], dtype=np.int8)
    policy = choose_conservative_policy(scores, similarities, labels, max_suppress_threshold=0.35)
    assert policy["suppress_threshold"] == 0.35
    assert policy["positive_false_suppressed"] == 0
    assert policy["positive_recall"] == 1.0
    assert policy["negative_suppression_rate"] > 0.0


def test_cross_validation_never_votes_from_same_source():
    embeddings = np.asarray([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.1, 0.9],
    ], dtype=np.float32)
    labels = np.asarray([1, 0, 1, 0, 1, 0], dtype=np.int8)
    source_ids = np.asarray([10, 10, 11, 20, 20, 21], dtype=np.int64)
    scores, similarities, neighbors = cross_validated_scores(
        embeddings,
        labels,
        source_ids,
        k=3,
        temperature=0.05,
    )
    assert np.all(neighbors > 0)
    assert np.all(np.isfinite(scores))
    assert np.all(similarities <= 1.00001)


def test_stable_source_id_is_repeatable_and_path_sensitive():
    assert stable_source_id("A:/one.png") == stable_source_id("A:/one.png")
    assert stable_source_id("A:/one.png") != stable_source_id("A:/two.png")
